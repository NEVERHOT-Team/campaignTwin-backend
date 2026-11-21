from collections import defaultdict
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from scipy.stats import beta

from contracts.models import CampaignSpec
from data import rival
from data.etl.build_rival_prior import (
    DEFAULT_PROBABILITY,
    EVENT_COLUMNS,
    _price_tiers,
    base_rate,
    brand_catalog,
    reaction_events,
)
from mocks import load_mock


def make_spec(
    discount_pct=20,
    mechanic="voucher",
    start="2026-11-09",
    end="2026-11-15",
    name="11.11 tissue voucher",
    target_segments=None,
    channels=None,
):
    """Build a CampaignSpec v0.1 from the few fields rival retrieval reads."""
    raw = deepcopy(load_mock("campaign_spec")["spec"])
    unit = "pct" if mechanic in ("voucher", "percent_off") else None
    raw["status"] = "draft"
    raw["name"] = name
    raw["brand"]["category"] = "toilet_tissue"
    raw["audience"]["description"] = None
    raw["offer"]["mechanics"] = [
        {
            "type": mechanic,
            "value": discount_pct,
            "value_unit": unit if discount_pct is not None else None,
            "min_spend_thb": None,
            "cap_per_user": 1,
            "eligibility": "all",
            "duration_days": None,
        }
    ]
    raw["budget"]["max_discount_pct"] = None
    raw["timing"].update({"start_date": start, "end_date": end, "event": "none"})
    if target_segments is not None:
        raw["audience"]["include_hints"] = target_segments
    if channels is not None:
        raw["channels"] = channels
    return CampaignSpec.model_validate(raw)


def flat(spec):
    return rival.spec_inputs(spec)


@pytest.fixture
def spec():
    return make_spec()


def test_top_three_evidence_intervals_and_plausibility(spec):
    by_rival = defaultdict(list)
    library_ids = set(rival.load_library()["campaign_id"])
    brands = rival.load_brands().set_index("brand")
    for result in rival.get_rival_scenarios(spec):
        by_rival[result.rival].append(result)
        assert set(result.evidence_ids) <= library_ids
        assert 0 <= result.likelihood <= 1
        assert 0 <= result.ci_low <= result.ci_high <= 1
        assert 0 <= result.depth <= 100 - brands.at[result.rival, "margin_floor_pct"]
        assert 0 <= result.rival_score <= 1
    assert set(by_rival) == {"764", "595", "69"}
    for passive, modal, aggressive in by_rival.values():
        assert [passive.scenario, modal.scenario, aggressive.scenario] == ["S1", "S2", "S3"]
        assert passive.reaction == "none" and passive.depth == 0
        assert aggressive.reaction == "undercut"
        assert aggressive.depth <= flat(spec).discount_pct + 10
        assert modal.likelihood >= aggressive.likelihood


@pytest.mark.parametrize("discount,active", [(0.1, "match"), (20, "undercut"), (100, "non_price")])
def test_positive_discount_selects_most_likely_active_reaction(spec, monkeypatch, discount, active):
    probabilities = {"none": 0.70, "match": 0.05, "undercut": 0.05, "non_price": 0.05}
    probabilities[active] = 0.20
    prior = {**rival.load_rival_prior(), "p": probabilities}
    library = rival.load_library().iloc[:0]
    monkeypatch.setattr(rival, "load_rival_prior", lambda: prior)
    monkeypatch.setattr(rival, "load_library", lambda: library)

    rows = rival.get_rival_scenarios(make_spec(discount_pct=discount))
    for scenario, expected in [("S1", "none"), ("S2", active)]:
        selected = [row for row in rows if row.scenario == scenario]
        assert {row.rival for row in selected} == {"764", "595", "69"}
        for row in selected:
            assert row.reaction == expected
            assert row.likelihood == pytest.approx(probabilities[expected])


@pytest.mark.parametrize("discount", [None, 0])
def test_no_discount_is_deterministically_passive(spec, discount):
    results = rival.get_rival_scenarios(make_spec(discount_pct=discount))
    assert len(results) == 9
    for row in results:
        assert row.reaction == "none" and row.depth == row.delay_weeks == row.duration_weeks == 0
        assert row.ci_low == row.likelihood == row.ci_high == 1
        assert row.evidence_ids == []


def test_competitor_score_ranks_all_configured_rivals_even_below_threshold(spec):
    catalog = rival.load_brands().copy()
    catalog.loc[catalog["is_ours"], ["price_tier", "geo"]] = ["mid", "TH"]
    catalog.loc[~catalog["is_ours"], ["category", "target_segments", "channels", "geo"]] = [
        "other",
        "waiter",
        "other",
        "other",
    ]
    catalog.loc[
        catalog["brand"].eq("595"), ["category", "price_tier", "target_segments", "channels", "geo"]
    ] = ["toilet_tissue", "low", "explorer|loyal", "Shopee|LINE", "TH"]
    targeted = make_spec(target_segments=["loyal"], channels=["line"])
    _, selected = rival._competitors(flat(targeted), catalog)
    assert set(selected["brand"]) == {"764", "595", "69"}
    assert selected.iloc[0].brand == "595"
    assert selected.iloc[1:].score.lt(0.5).all()
    assert selected.iloc[0].score == pytest.approx(0.35 + 0.125 + 0.10 + 0.075 + 0.05)


def test_deeper_discount_does_not_lower_undercut(spec):
    previous = {}
    for discount in [1, 8, 12, 20, 25.5, 30, 40, 44, 50, 75, 100]:
        rows = rival.get_rival_scenarios(make_spec(discount_pct=discount))
        current = {row.rival: row.likelihood for row in rows if row.scenario == "S3"}
        assert all(risk >= previous.get(brand, 0) - 1e-12 for brand, risk in current.items())
        previous = current


def test_linked_campaign_votes_once_and_uses_responders_terms(spec, monkeypatch):
    template = rival.load_library().iloc[0].to_dict()
    trigger = {
        **template,
        "campaign_id": "launch",
        "brand": "751",
        "category": "toilet_tissue",
        "description": flat(spec).text,
        "mechanic": flat(spec).mechanic,
        "season": "11.11",
        "discount_depth": 0.20,
        "reaction_type": "",
        "reaction_to_campaign_id": "",
        "start_date": flat(spec).start,
        "end_date": flat(spec).end,
    }
    response = {
        **trigger,
        "campaign_id": "response",
        "brand": "764",
        "reaction_to_campaign_id": "launch",
        "reaction_type": "undercut",
        "start_date": "2026-11-12",
        "end_date": "2026-11-25",
        "reaction_delay_days": 3,
        "discount_depth": 0.99,
        "description": "A different response",
        "mechanic": "bundle",
    }
    unknown = {**response, "campaign_id": "unknown", "reaction_type": ""}
    monkeypatch.setattr(rival, "load_library", lambda: pd.DataFrame([trigger, response, unknown]))
    prior = rival.load_rival_prior()
    row = next(
        r for r in rival.get_rival_scenarios(spec) if r.rival == "764" and r.scenario == "S3"
    )
    assert row.evidence_ids == ["response"]
    alpha = 5 * prior["p"]["undercut"] + 1
    assert row.likelihood == pytest.approx(alpha / 6)
    assert [row.ci_low, row.ci_high] == pytest.approx(beta.ppf([0.05, 0.95], alpha, 6 - alpha))
    assert row.depth == 30  # 20% offer plus ten percentage points
    assert row.delay_weeks == 1 and row.duration_weeks == 2


def test_all_unknown_reactions_use_category_prior(spec, monkeypatch):
    library = rival.load_library().assign(reaction_type="")
    monkeypatch.setattr(rival, "load_library", lambda: library)
    prior = rival.load_rival_prior()
    for row in rival.get_rival_scenarios(spec):
        assert row.likelihood == pytest.approx(prior["p"][row.reaction])
        assert row.evidence_ids == []
        if row.scenario == "S3":
            assert row.depth == pytest.approx(min(prior["depth_p90"], flat(spec).discount_pct + 10))
            assert row.delay_weeks == prior["delay_weeks_min"]


def test_reaction_onsets_require_washout_and_positive_lag():
    index = pd.MultiIndex.from_product([range(1, 11), range(1, 13)], names=["brand", "week"])
    weekly = pd.DataFrame({"depth": 0.0, "featured": False}, index=index)
    weekly.loc[(1, 6), "depth"] = 20
    weekly.loc[(2, 9), "depth"] = 25  # boundary match, lag 3
    weekly.loc[(3, 7), "depth"] = 26  # new undercut, lag 1
    weekly.loc[(4, [6, 7]), "depth"] = 30  # started on event week
    weekly.loc[(5, [4, 8]), "depth"] = 30  # cutting two weeks before
    weekly.loc[(6, [5, 8]), "featured"] = True  # featured one week before
    weekly.loc[(7, [8, 9]), "featured"] = True  # new non-price display
    weekly.loc[(8, 10), "depth"] = 28  # too late
    weekly.loc[(9, [6, 7]), "featured"] = True  # existing display
    weekly.loc[(10, 7), ["featured", "depth"]] = [True, 3]  # small price drop
    events = reaction_events(weekly)
    triggered = events.loc[events["brand"].eq(1) & events["week"].eq(6)].set_index("rival")
    assert triggered["reaction_type"].to_dict() == {
        2: "match",
        3: "undercut",
        4: "none",
        5: "none",
        6: "none",
        7: "non_price",
        8: "none",
        9: "none",
        10: "none",
    }
    assert triggered.at[2, "lag_weeks"] == 3 and triggered.at[3, "lag_weeks"] == 1
    assert triggered.at[7, "duration_weeks"] == 2
    assert not triggered.at[5, "washout_clear"] and not triggered.at[6, "washout_clear"]
    assert events["lag_weeks"].dropna().between(1, 3).all()
    for row in events.dropna(subset=["background_week"]).itertuples():
        anchor = weekly.loc[(row.brand, row.background_week)]
        assert anchor.depth < 10 and not anchor.featured
        if pd.notna(row.background_reaction_week):
            assert row.background_washout_clear
            assert 1 <= row.background_reaction_week - row.background_week <= 3
    pd.testing.assert_frame_equal(reaction_events(weekly.sample(frac=1, random_state=7)), events)


def test_control_week_can_produce_a_reaction_absent_after_the_event():
    index = pd.MultiIndex.from_product([[1, 2], range(1, 15)], names=["brand", "week"])
    weekly = pd.DataFrame({"depth": 0.0, "featured": False}, index=index)
    weekly.loc[(1, slice(3, 11)), "depth"] = 20
    weekly.at[(1, 8), "depth"] = 0  # only focal non-event anchor with full windows
    weekly.at[(2, 10), "depth"] = 27
    audit = reaction_events(weekly).set_index(["brand", "week", "rival"])
    row = audit.loc[(1, 4, 2)]
    assert row.reaction_type == "none"
    assert row.background_week == 8 and row.background_reaction_week == 10
    assert row.background_reaction_type == "undercut" and row.background_washout_clear


@pytest.mark.parametrize(
    "n,strong,blend", [(19, False, True), (20, False, False), (20, True, True)]
)
def test_background_subtraction_and_both_blend_conditions(n, strong, blend):
    # One reaction per event. Background shares exceed event shares for non-price.
    responses = ["match"] * n if strong else ["undercut"] * 4 + ["match"] * 4 + ["none"] * (n - 8)
    controls = (
        ["non_price"] * n if strong else ["undercut"] * 2 + ["non_price"] * 3 + ["none"] * (n - 5)
    )
    events = pd.DataFrame(
        {
            "brand": 1,
            "week": range(n),
            "reaction_type": responses,
            "washout_clear": True,
            "background_week": range(n),
            "background_reaction_type": controls,
            "reaction_depth": 25,
            "lag_weeks": 2,
            "duration_weeks": 1,
        }
    )
    prior = base_rate(events)
    lifts = {"match": 1.0 if strong else 4 / n, "undercut": 0 if strong else 2 / n, "non_price": 0}
    assert prior["lift"] == pytest.approx(lifts)
    adjusted = {"none": 1 - sum(lifts.values()), **lifts}
    assert prior["n_events"] == n
    for label, value in adjusted.items():
        expected = (value + DEFAULT_PROBABILITY[label]) / 2 if blend else value
        assert prior[label] == pytest.approx(expected)
    assert prior["source"] == ("data+default" if blend else "data")
    assert prior["none"] >= 0.2 and prior["delay_median"] >= 1
    empty = base_rate(pd.DataFrame(columns=EVENT_COLUMNS))
    assert empty["none"] == pytest.approx((1 + 0.45) / 2)
    assert empty["delay_median"] == 1


def test_neighbour_limit_and_stable_ties():
    evidence = pd.DataFrame(
        {
            "campaign_id": [str(i) for i in range(12)],
            "fixed_score": np.full(12, 0.5),
            "trigger_depth": np.full(12, 20),
            "reaction_type": ["match"] * 12,
        }
    )
    _, neighbours, ids = rival._posterior(evidence, 20, {"p": DEFAULT_PROBABILITY})
    assert len(neighbours) == 10 and ids == list(evidence["campaign_id"].iloc[:10])


def test_risk_floor_retrieves_current_neighbours_for_response_terms():
    evidence = pd.DataFrame(
        {
            "campaign_id": [f"c{i:02}" for i in range(20)],
            "fixed_score": [0.5] * 20,
            "trigger_depth": [20] * 10 + [80] * 10,
            "reaction_type": ["undercut"] * 10 + ["match"] * 10,
        }
    )
    prior = {"p": DEFAULT_PROBABILITY}
    shallow, _, _ = rival._posterior(evidence, 20, prior)
    deep, neighbours, ids = rival._posterior(evidence, 80, prior)
    assert deep[2] / deep.sum() == pytest.approx(shallow[2] / shallow.sum())
    assert deep.sum() == pytest.approx(5 + 10 * 0.6)
    assert set(neighbours["campaign_id"]) == set(evidence["campaign_id"].iloc[10:])
    assert set(ids) == set(evidence["campaign_id"])


@pytest.mark.parametrize("label", ["match", "undercut"])
def test_brand_specific_evidence_can_make_price_response_modal(spec, monkeypatch, label):
    template = rival.load_library().iloc[0].to_dict()
    rows = pd.DataFrame(
        [
            {
                **template,
                "campaign_id": f"obs{i}",
                "brand": "764",
                "category": "toilet_tissue",
                "season": "11.11",
                "mechanic": "voucher",
                "discount_depth": 0.42,
                "start_date": flat(spec).start,
                "end_date": flat(spec).end,
                "description": flat(spec).text,
                "reaction_type": label,
                "reaction_to_campaign_id": "",
                "reaction_delay_days": 8,
            }
            for i in range(10)
        ]
    )
    monkeypatch.setattr(rival, "load_library", lambda: rows)
    result = [r for r in rival.get_rival_scenarios(spec) if r.rival == "764"]
    assert result[1].reaction == label and result[1].depth == 42
    assert result[1].delay_weeks == 2 and result[1].likelihood > result[0].likelihood
    assert result[2].depth <= 30


@pytest.mark.parametrize(
    "discount,response,expected", [(95, 0.99, 90), (50, 0.24, 24), (5, 0.80, 15)]
)
def test_s3_caps_take_smallest_percent_bound(spec, monkeypatch, discount, response, expected):
    library = rival.load_library().assign(reaction_type="undercut", discount_depth=response)
    monkeypatch.setattr(rival, "load_library", lambda: library)
    results = rival.get_rival_scenarios(make_spec(discount_pct=discount))
    assert all(row.depth == expected for row in results if row.scenario == "S3")


def test_tiers_use_area_and_targets_use_relative_buyer_share():
    brands = [751, 764, 595, 69, 1011, 1315]
    samples = {brand: list(range(10)) for brand in brands}
    samples[751] = [0, 1, 2, 3, 8, 9]  # four loyal, one waiter and one explorer
    samples[764] += [0, 0, 0]
    rows = [
        (brand, brand, buyer, 1, 10, 10, 20)
        for brand, buyers in samples.items()
        for buyer in buyers
    ]
    lines = pd.DataFrame(
        rows,
        columns=[
            "manufacturer_id",
            "product_id",
            "household_id",
            "quantity",
            "paid_total",
            "sales_value",
            "discount_pct",
        ],
    )
    households = pd.DataFrame({"archetype": ["loyal"] * 8 + ["waiter", "explorer"]})
    sizes = pd.Series(
        ["50 SQ FT", "20 S FT", "200 SQ FT", "1000 SQ FT", "4 ROLL", None], index=brands
    )
    tiers = _price_tiers(lines, sizes)
    catalog = brand_catalog(lines, households, tiers).set_index("brand")
    assert tiers.at[764, "paid_thb_per_sq_ft"] == 0.5
    assert tiers.at[69, "paid_thb_per_sq_ft"] == 0.01
    assert tiers.at[69, "price_tier"] == "low" and tiers.at[764, "price_tier"] == "high"
    assert set(catalog.at[751, "target_segments"].split("|")) == {"waiter", "explorer"}
    fallbacks = _price_tiers(lines, sizes.map(lambda s: "6 ROLL"))
    assert fallbacks["source"].eq("default").all()
    assert fallbacks.loc[[764, 751, 595, 69], "price_tier"].tolist() == [
        "high",
        "mid",
        "mid",
        "low",
    ]


def test_spec_inputs_reads_nested_campaign_spec():
    nested = make_spec(discount_pct=35, mechanic="percent_off", target_segments=["deal_hunter"])
    view = rival.spec_inputs(nested)
    assert view.discount_pct == 35 and view.mechanic == "percent_off"
    assert (view.start, view.end) == ("2026-11-09", "2026-11-15")
    assert view.target_segments == ["deal_hunter"] and view.channels == ["shopee", "line"]
    assert "11.11 tissue voucher" in view.text


def test_fixed_baht_mechanic_counts_as_no_price_discount():
    nested = make_spec(discount_pct=50, mechanic="fixed_off")
    nested.offer.mechanics[0].value_unit = "thb"
    assert rival.spec_inputs(nested).discount_pct is None
    assert all(row.reaction == "none" for row in rival.get_rival_scenarios(nested))


def test_timing_event_sets_season():
    nested = make_spec()
    nested.timing.event = "12.12"
    assert rival.spec_inputs(nested).event == "12.12"
