"""Back-test: replay Complete Journey campaign 18 through the shopper model.

The historical campaign (data/processed/backtest_campaign.json) is simulated against a
paired no-campaign baseline on the same seeds, and the simulated lift in *category*
sales (us + rival + other) over the campaign weeks is compared with the observed lift
range by data.backtest.evaluate_backtest().

The discount has two parts, as agreed with Gan (depths measured in backtest_campaign.json):
- shelf markdown (shelf_depth_fraction): every shopper sees it, reach shelf_reach;
- coupon (coupon_depth_fraction on redemptions): only the treated share holds it.
The observed redemption rate is an outcome, so it is never used as reach. The simulated
redemption rate is reported next to it as an extra check.

Run `python -m engine.backtest` for a JSON report. `sweep()` evaluates several
ShopperParams overrides for calibration (values go to shopper_params.json).
"""

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np

from data.backtest import evaluate_backtest
from data.loaders import load_backtest
from engine import shopper

DEFAULT_SEEDS = (0, 1, 2, 3, 4)
DEFAULT_AGENTS = 5000
PANEL_HOUSEHOLDS = 2500  # Complete Journey panel size, used only if treated_share is absent


@dataclass(frozen=True)
class BacktestResult:
    campaign_id: str
    weeks: int
    discount_pct: float  # the record's combined median line depth, for reference
    shelf_discount_pct: float
    shelf_reach: float
    coupon_discount_pct: float
    treated_share: float
    simulated_lift_pct: float
    lift_by_seed: list[float]
    simulated_redemption_rate: float | None
    observed_redemption_rate: float | None
    evaluation: dict[str, Any]
    labels: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CampaignInputs:
    weeks: int
    shelf_pct: float  # shelf markdown every shopper sees
    shelf_reach: float
    coupon_pct: float  # extra markdown when a coupon holder buys
    treated_share: float  # share of households holding the coupon
    labels: dict[str, str]


def campaign_inputs(record: dict[str, Any]) -> CampaignInputs:
    """Offer inputs for the campaign record, preferring Gan's measured depths.

    mechanics.all_tissue_purchases.shelf_depth_fraction is the shelf markdown on all
    tissue purchases; matched_campaign_redemptions.coupon_depth_fraction is the extra
    markdown on purchases where a campaign coupon was redeemed. Without them the single
    discount_depth is split by shelf_share (default: all shelf).
    """
    labels: dict[str, str] = {}
    weeks = int(record["end_week"]) - int(record["start_week"]) + 1
    mechanics = record.get("mechanics") or {}
    purchases = mechanics.get("all_tissue_purchases") or {}
    matched = mechanics.get("matched_campaign_redemptions") or {}
    total = 100 * float(record["discount_depth"])
    if "shelf_depth_fraction" in purchases:
        shelf = 100 * float(purchases["shelf_depth_fraction"])
        coupon = 100 * float(
            matched.get("coupon_depth_fraction", purchases.get("coupon_depth_fraction", 0.0))
        )
        labels["shelf_pct"] = labels["coupon_pct"] = "data"
    elif "shelf_share" in record:
        share = float(np.clip(record["shelf_share"], 0, 1))
        shelf, coupon = total * share, total * (1 - share)
        labels["shelf_pct"] = labels["coupon_pct"] = "data"
    else:
        shelf, coupon = total, 0.0
        labels["shelf_pct"] = labels["coupon_pct"] = "default"
    if "shelf_reach" in record:
        reach, labels["shelf_reach"] = float(record["shelf_reach"]), "data"
    else:
        reach, labels["shelf_reach"] = 1.0, "default"
    if "treated_share" in record:
        treated, labels["treated_share"] = float(record["treated_share"]), "data"
    else:
        treated = float(record.get("treated_households", 0)) / PANEL_HOUSEHOLDS
        labels["treated_share"] = "default"
    return CampaignInputs(
        weeks=weeks,
        shelf_pct=float(np.clip(shelf, 0, 100)),
        shelf_reach=float(np.clip(reach, 0, 1)),
        coupon_pct=float(np.clip(coupon, 0, 100)),
        treated_share=float(np.clip(treated, 0, 1)),
        labels=labels,
    )


def _category_sales(stats: Sequence[shopper.WeekStats]) -> float:
    return float(sum(s.revenue_us + s.revenue_rival + s.revenue_other for s in stats))


def _run(
    population: shopper.Population,
    offer: shopper.WeekOffer,
    weeks: int,
    seed: int,
    params: shopper.ShopperParams,
) -> list[shopper.WeekStats]:
    state = shopper.initial_state(population, seed)
    out = []
    for week in range(1, weeks + 1):
        stats, state = shopper.step(population, state, offer, seed, week, params)
        out.append(stats)
    return out


def run_backtest(
    seeds: Sequence[int] = DEFAULT_SEEDS,
    n_agents: int = DEFAULT_AGENTS,
    params: shopper.ShopperParams | None = None,
    record: dict[str, Any] | None = None,
) -> BacktestResult:
    record = record or load_backtest()
    params = params or shopper.default_params()
    # The whole panel is measured, so no holdout is kept out of the offer here.
    params = replace(params, holdout_share=0.0)
    inputs = campaign_inputs(record)
    weeks = inputs.weeks
    population = shopper.build_population(n_agents, seed=0, params=params)
    targets = tuple(record.get("target_segments") or ())
    offer = shopper.WeekOffer(
        active=True,
        our_discount_pct=inputs.shelf_pct,
        reach=inputs.shelf_reach,
        target_segments=targets,
        coupon_discount_pct=inputs.coupon_pct,
        coupon_reach=inputs.treated_share,
    )
    lifts = []
    redemptions = holders = 0.0
    for seed in seeds:
        base = _run(population, shopper.WeekOffer(), weeks, seed, params)
        plan = _run(population, offer, weeks, seed, params)
        base_sales = _category_sales(base)
        lifts.append(100 * (_category_sales(plan) - base_sales) / base_sales if base_sales else 0.0)
        redemptions += sum(s.coupon_redemptions for s in plan)
        holders += max((s.coupon_holders for s in plan), default=0)
    lift = float(np.mean(lifts))
    observed = (record.get("observed_redemption_rate")
                or (record.get("actual") or {}).get("redemption_rate"))
    return BacktestResult(
        campaign_id=str(record["campaign_id"]),
        weeks=weeks,
        discount_pct=round(100 * float(record["discount_depth"]), 4),
        shelf_discount_pct=round(inputs.shelf_pct, 4),
        shelf_reach=inputs.shelf_reach,
        coupon_discount_pct=round(inputs.coupon_pct, 4),
        treated_share=inputs.treated_share,
        simulated_lift_pct=round(lift, 4),
        lift_by_seed=[round(v, 4) for v in lifts],
        # Coupon purchases per holder. The observed rate counts households with at least
        # one redemption, so the simulated value is an upper bound on that rate.
        simulated_redemption_rate=round(redemptions / holders, 6) if holders else None,
        observed_redemption_rate=float(observed) if observed is not None else None,
        evaluation=evaluate_backtest(lift),
        labels={**inputs.labels, "params": params.label},
    )


def sweep(
    overrides: Sequence[dict[str, float]],
    seeds: Sequence[int] = DEFAULT_SEEDS,
    n_agents: int = DEFAULT_AGENTS,
) -> list[dict[str, Any]]:
    """Back-test each ShopperParams override (e.g. {"beta_price": 3.0}) for calibration."""
    base = shopper.default_params()
    rows = []
    for change in overrides:
        result = run_backtest(seeds, n_agents, replace(base, **change))
        rows.append({
            "params": change,
            "simulated_lift_pct": result.simulated_lift_pct,
            "label": result.evaluation["label"],
            "simulated_redemption_rate": result.simulated_redemption_rate,
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agents", type=int, default=DEFAULT_AGENTS)
    parser.add_argument("--seeds", type=int, default=len(DEFAULT_SEEDS))
    args = parser.parse_args()
    result = run_backtest(range(args.seeds), args.agents)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
