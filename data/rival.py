"""Retrieve competitors and estimate scenario probabilities from campaign evidence."""

import math
import re
from dataclasses import dataclass, replace
from datetime import date

import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.feature_extraction.text import TfidfVectorizer

from contracts.models import CampaignSpec, RivalScenario, Strategy
from data.etl.config import MANUFACTURERS
from data.loaders import brand_display_name, load_brands, load_library, load_rival_prior

REACTIONS = ("none", "match", "undercut", "non_price")
UNDERCUT = REACTIONS.index("undercut")
PRIOR_STRENGTH = 5
NEIGHBOURS = 10


@dataclass(frozen=True)
class RivalInput:
    """The few CampaignSpec v0.1 fields that rival retrieval reads, flattened once."""

    discount_pct: float | None
    mechanic: str
    start: str | None
    end: str | None
    event: str
    text: str
    target_segments: list[str]
    channels: list[str]


def spec_inputs(spec: CampaignSpec) -> RivalInput:
    """Flatten a nested CampaignSpec v0.1 into the inputs rival retrieval needs.

    The discount is the deepest percentage mechanic (percent_off, or any mechanic with
    value_unit "pct"). Fixed-baht mechanics carry no percentage, so they count as no
    price discount for rival reactions.
    """
    mechanics = spec.offer.mechanics
    percentages = [m.percentage for m in mechanics if m.percentage is not None]
    discount = max(percentages) if percentages else None
    priced = [m for m in mechanics if m.percentage is not None]
    mechanic = (priced or mechanics)[0].type if mechanics else ""
    message = spec.offer.message
    event = spec.timing.event if spec.timing.event not in (None, "none") else ""
    text = " ".join(
        part
        for part in (
            spec.name,
            event,
            mechanic.replace("_", " "),
            spec.audience.description,
            message.headline if message else None,
            message.key_message if message else None,
        )
        if part
    )
    return RivalInput(
        discount_pct=discount,
        mechanic=mechanic,
        start=spec.timing.start_date.isoformat() if spec.timing.start_date else None,
        end=spec.timing.end_date.isoformat() if spec.timing.end_date else None,
        event=event,
        text=text,
        target_segments=list(spec.audience.include_hints),
        channels=list(spec.channels),
    )


def _tags(value: str | list[str]) -> set[str]:
    parts = value.split("|") if isinstance(value, str) else value
    return {part.strip().casefold() for part in parts if part.strip()}


def _overlap(left: set[str], right: set[str]) -> float:
    return len(left & right) / len(left | right) if left or right else 0.0


def _competitors(spec: RivalInput, brands: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    ours = brands.loc[brands["is_ours"]].iloc[0]
    tiers = {"low": 0, "mid": 1, "high": 2}
    target = _tags(spec.target_segments or ours.target_segments)
    channels = _tags(spec.channels or ours.channels)
    rivals = brands.loc[~brands["is_ours"] & brands["brand"].isin(map(str, MANUFACTURERS))].copy()
    scores = []
    for brand in rivals.itertuples():
        tier_distance = abs(tiers[ours.price_tier] - tiers[brand.price_tier])
        scores.append(
            0.35 * (brand.category == ours.category)
            + 0.25 * (1 - tier_distance / 2)
            + 0.20 * _overlap(target, _tags(brand.target_segments))
            + 0.15 * _overlap(channels, _tags(brand.channels))
            + 0.05 * _overlap(_tags(ours.geo), _tags(brand.geo))
        )
    rivals["score"] = scores
    ranked = rivals.sort_values(["score", "market_share", "brand"], ascending=[False, False, True])
    return ours, ranked


def _season(spec: RivalInput, start: pd.Timestamp, end: pd.Timestamp) -> str:
    if spec.event:
        return spec.event
    match = re.search(r"\b(1[0-2]|[1-9])\.\1\b", spec.text)
    if match:
        return match.group()
    if "payday" in spec.text.casefold():
        return "payday"
    if pd.notna(start) and pd.notna(end):
        for year in range(start.year, end.year + 1):
            for month in range(1, 13):
                if start.date() <= date(year, month, month) <= end.date():
                    return f"{month}.{month}"
    return ""


def _days(rows: pd.DataFrame) -> pd.Series:
    start = pd.to_datetime(rows["start_date"], errors="coerce")
    end = pd.to_datetime(rows["end_date"], errors="coerce")
    days = (end - start).dt.days + 1
    return days.where(days.gt(0))


MECHANIC_ALIASES = {
    "percent_off": "pct_off",
    "percentage_discount": "pct_off",
    "free_shipping": "free_ship",
    "coins_cashback": "cashback",
}


def _mechanic(name: str) -> str:
    name = str(name).strip().casefold()
    if "voucher" in name:
        return "voucher"
    return MECHANIC_ALIASES.get(name, name)


def _text_scores(goal: str, descriptions: pd.Series) -> np.ndarray:
    if not goal.strip() or not descriptions.str.strip().any():
        return np.zeros(len(descriptions))
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
    matrix = vectorizer.fit_transform(descriptions)
    return (matrix @ vectorizer.transform([goal]).T).toarray().ravel()


def _campaign_features(
    spec: RivalInput, category: str, library: pd.DataFrame, candidates: pd.DataFrame
) -> pd.DataFrame:
    outcomes = candidates.sort_values("campaign_id").reset_index(drop=True)
    if outcomes.empty:
        return outcomes
    by_id = library.set_index("campaign_id")
    parent_ids = outcomes["reaction_to_campaign_id"]
    trigger_ids = parent_ids.where(parent_ids.isin(by_id.index), outcomes["campaign_id"])
    triggers = by_id.loc[trigger_ids].reset_index(drop=True)

    start = pd.to_datetime(spec.start, errors="coerce")
    end = pd.to_datetime(spec.end, errors="coerce")
    duration = (end - start).days + 1 if pd.notna(start) and pd.notna(end) else 0
    hist_days = _days(triggers)
    duration_sim = np.minimum(hist_days, duration) / np.maximum(hist_days, duration)
    season = _season(spec, start, end)
    mechanic = _mechanic(spec.mechanic or "")
    feature_sum = (
        (triggers["mechanic"].map(_mechanic).eq(mechanic) & bool(mechanic)).astype(float)
        + duration_sim.fillna(0).clip(0, 1)
        + (triggers["season"].str.casefold().eq(season) & bool(season))
        + triggers["category"].str.casefold().eq(category.casefold())
    )
    outcomes["fixed_score"] = 0.1 * feature_sum + 0.5 * _text_scores(
        spec.text, triggers["description"]
    )
    # The hand-edited campaign library stores fractions. All rival depths use percent.
    outcomes["trigger_depth"] = 100 * pd.to_numeric(triggers["discount_depth"], errors="coerce")
    outcomes["response_depth"] = 100 * pd.to_numeric(outcomes["discount_depth"], errors="coerce")
    outcomes["duration_days"] = _days(outcomes)
    return outcomes


def _discount_level(percent: float) -> int:
    return max(1, int(np.floor(np.clip(percent, 0, 100) + 0.5)))


def _similarity(fixed: np.ndarray, depth: np.ndarray, percent: float) -> np.ndarray:
    depth_sim = np.where(np.isfinite(depth), np.maximum(0, 1 - np.abs(percent - depth) / 100), 0)
    return fixed + 0.1 * depth_sim


def _posterior(
    evidence: pd.DataFrame, percent: float, prior: dict
) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    alpha0 = PRIOR_STRENGTH * np.array([prior["p"][reaction] for reaction in REACTIONS])
    if evidence.empty:
        return alpha0, evidence.assign(sim=pd.Series(dtype=float)), []
    fixed = evidence["fixed_score"].to_numpy()
    depth = evidence["trigger_depth"].to_numpy(dtype=float)
    labels = evidence["reaction_type"].to_numpy()
    # Evaluate the same neighbours and prior on a percentage-point grid. A shallower
    # offer supplies a lower bound on undercut risk when sparse analogues reverse it.
    level = _discount_level(percent)
    peak_risk = -1.0
    peak_rows = []
    for point in range(1, level + 1):
        sim = _similarity(fixed, depth, point)
        nearest = np.argsort(-sim, kind="stable")[:NEIGHBOURS]
        known = nearest[(labels[nearest] != "") & (sim[nearest] > 0)]
        counts = np.array([sim[known][labels[known] == label].sum() for label in REACTIONS])
        alpha = alpha0 + counts
        risk = alpha[UNDERCUT] / alpha.sum()
        if risk >= peak_risk:
            peak_risk = risk
            peak_rows = known
    neighbours = evidence.iloc[known].assign(sim=sim[known])
    ids = neighbours["campaign_id"].tolist()
    if risk < peak_risk:
        # Preserve the current neighbours' relative odds for the other outcomes and
        # the posterior's total concentration when enforcing the undercut risk floor.
        total = alpha.sum()
        alpha *= (1 - peak_risk) / (1 - risk)
        alpha[UNDERCUT] = total * peak_risk
        ids = list(dict.fromkeys([*ids, *evidence.iloc[peak_rows]["campaign_id"]]))
    return alpha, neighbours, ids


def _quantile(rows: pd.DataFrame, field: str, q: float, fallback: float) -> float:
    if rows.empty:
        return fallback
    values = pd.to_numeric(rows[field], errors="coerce")
    valid = values.notna() & values.ge(0) & rows["sim"].gt(0)
    if not valid.any():
        return fallback
    order = values.loc[valid].sort_values(kind="stable")
    weights = rows.loc[order.index, "sim"].cumsum()
    position = weights.searchsorted(q * weights.iloc[-1])
    return float(order.iloc[position])


def _interval(alpha: np.ndarray, index: int) -> tuple[float, float]:
    a = alpha[index]
    b = alpha.sum() - a
    if a <= 0:
        return 0.0, 0.0
    if b <= 0:
        return 1.0, 1.0
    low, high = beta.ppf([0.05, 0.95], a, b)
    return float(low), float(high)


def strategy_inputs(spec: CampaignSpec, strategy: Strategy) -> RivalInput:
    """The brief's inputs with one strategy's offer, so each plan gets its own S1-S3.

    The baseline (or any zero discount) yields no price discount, so every rival is `none`.
    """
    base = spec_inputs(spec)
    return replace(
        base,
        discount_pct=None if strategy.is_baseline else float(strategy.discount_pct),
        mechanic="" if strategy.mechanic == "none" else strategy.mechanic,
        target_segments=list(strategy.target_segments) or base.target_segments,
        channels=list(strategy.channel_mix) or base.channels,
    )


def similar_campaigns(
    spec: CampaignSpec, strategy: Strategy | None = None, k: int = 5
) -> list[dict]:
    """Return up to k sourced campaigns for Competitive Outlook, with discounts in percent."""
    if k <= 0:
        return []
    library = load_library()
    if library.empty:
        return []
    inputs = spec_inputs(spec) if strategy is None else strategy_inputs(spec, strategy)
    campaigns = _campaign_features(inputs, spec.brand.category or "", library, library)
    percent = _discount_level(inputs.discount_pct) if inputs.discount_pct else 0
    campaigns["similarity"] = _similarity(
        campaigns["fixed_score"].to_numpy(), campaigns["trigger_depth"].to_numpy(), percent
    )
    campaigns["discount_depth"] = campaigns["response_depth"]
    sourced = campaigns["source_url"].fillna("").str.strip().ne("")
    ranked = campaigns.loc[sourced & campaigns["discount_depth"].between(0, 100)]
    if ranked.empty:
        return []
    columns = [
        "campaign_id", "brand", "brand_name", "platform", "mechanic", "discount_depth",
        "description", "source_url", "confidence", "similarity",
    ]
    top = ranked.nlargest(k, "similarity")
    top["brand_name"] = top["brand"].map(brand_display_name)
    return top[columns].to_dict("records")


def get_rival_scenarios(
    spec: CampaignSpec, strategy: Strategy | None = None
) -> list[RivalScenario]:
    """S1-S3 per top rival. Pass a strategy to price the rival response to that plan."""
    spec = spec_inputs(spec) if strategy is None else strategy_inputs(spec, strategy)
    ours, rivals = _competitors(spec, load_brands())
    prior = load_rival_prior()
    library = load_library()
    scenarios = []
    for rival in rivals.itertuples():
        if not spec.discount_pct or spec.discount_pct <= 0:
            scenarios.extend(
                RivalScenario(
                    rival=rival.brand,
                    scenario=scenario,
                    reaction="none",
                    depth=0,
                    delay_weeks=0,
                    duration_weeks=0,
                    likelihood=1,
                    ci_low=1,
                    ci_high=1,
                    evidence_ids=[],
                    rival_score=rival.score,
                )
                for scenario in ("S1", "S2", "S3")
            )
            continue
        specific = library["brand"].eq(rival.brand)
        candidates = library.loc[specific] if specific.any() else library
        evidence = _campaign_features(spec, ours.category, library, candidates)
        alpha, neighbours, ids = _posterior(evidence, spec.discount_pct, prior)
        modal_active_index = 1 + int(alpha[1:].argmax())
        for scenario, index in (("S1", 0), ("S2", modal_active_index), ("S3", UNDERCUT)):
            reaction = REACTIONS[index]
            depth = delay = duration = 0
            if reaction != "none":
                rows = neighbours.loc[neighbours["reaction_type"].eq(reaction)]
                aggressive = scenario == "S3"
                q = 0.9 if aggressive else 0.5
                if reaction != "non_price":
                    raw = _quantile(
                        rows, "response_depth", q, prior["depth_p90" if aggressive else "depth_p50"]
                    )
                    depth = min(raw, max(0, 100 - rival.margin_floor_pct))
                    if aggressive:
                        depth = min(depth, spec.discount_pct + 10)
                lag = prior["delay_weeks_min" if aggressive else "delay_weeks_p50"]
                delay = math.ceil(
                    _quantile(rows, "reaction_delay_days", 0 if aggressive else 0.5, lag * 7) / 7
                )
                duration = max(
                    1,
                    math.ceil(
                        _quantile(rows, "duration_days", 0.5, prior["duration_weeks"] * 7) / 7
                    ),
                )
            low, high = _interval(alpha, index)
            scenarios.append(
                RivalScenario(
                    rival=rival.brand,
                    scenario=scenario,
                    reaction=reaction,
                    depth=depth,
                    delay_weeks=delay,
                    duration_weeks=duration,
                    likelihood=alpha[index] / alpha.sum(),
                    ci_low=low,
                    ci_high=high,
                    evidence_ids=ids,
                    rival_score=rival.score,
                )
            )
    return scenarios
