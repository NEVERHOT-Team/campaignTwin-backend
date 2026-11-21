from itertools import product
from statistics import mean

import pandas as pd
import pytest

from data import loaders
from data.backtest import evaluate_backtest, naive_benchmark
from data.etl import config
from data.etl.appeal import build_appeal_fallback
from data.etl.backtest import CONTROL_PRODUCT_TYPES, _historical_benchmark
from data.etl.build_engine_inputs import build_assumptions


def test_published_naive_benchmark_uses_only_comparable_campaigns():
    note = loaders.load_backtest()["naive_benchmark"]
    assert note["source"] == "data"
    assert note["excluded"] == [{"campaign_id": "27", "reason": "incomplete baseline"}]
    campaigns = {row["campaign_id"]: row for row in note["campaigns"]}
    assert set(campaigns) == {"8", "13"}
    assert set(campaigns["8"]["control_changes_pct"]) == set(CONTROL_PRODUCT_TYPES) - {
        "PAPER NAPKINS"
    }
    assert set(campaigns["13"]["control_changes_pct"]) == {"BASIC PAPER PLATES"}
    for row in campaigns.values():
        assert row["baseline_weeks"][1] < row["campaign_weeks"][0] <= row["campaign_weeks"][1] < 45
        assert row["baseline_weeks"][1] - row["baseline_weeks"][0] == 7
        assert row["lift_pct"] == pytest.approx(
            row["tissue_change_pct"] - mean(row["control_changes_pct"].values())
        )
    assert naive_benchmark() == round(mean(row["lift_pct"] for row in campaigns.values()), 2)
    assert naive_benchmark() == 19.02


def test_historical_did_ignores_partial_weeks_contaminated_controls_and_holdout():
    week_one = pd.Timestamp("2017-01-02")

    def day(week, offset=0):
        return week_one + pd.Timedelta(weeks=week - 1, days=offset)

    schedule = pd.DataFrame(
        [
            # Two complete active weeks, after an incomplete launch week.
            (8, day(10, 1), day(12, 6)),
            (27, day(2), day(3, 6)),
            (18, day(45), day(52, 6)),
            (13, day(43), day(45, 1)),
        ],
        columns=["campaign_id", "start_date", "end_date"],
    )
    tissue = config.PRODUCT_PATH[-1]
    weekly = pd.DataFrame(100.0, index=range(1, 53), columns=[tissue, *CONTROL_PRODUCT_TYPES])
    weekly.loc[10, tissue] = 50_000
    weekly.loc[11:12, tissue] = 130.0
    weekly.loc[11:12, list(CONTROL_PRODUCT_TYPES)] = 110.0
    weekly.loc[11:12, CONTROL_PRODUCT_TYPES[0]] = 9000.0
    weekly.loc[45:] = 1_000_000
    offered = pd.DataFrame(
        {
            "campaign_id": [8, 27, 18, 13, 8],
            "product_type": [tissue] * 4 + [CONTROL_PRODUCT_TYPES[0]],
        }
    )
    score, info = _historical_benchmark(weekly, schedule, offered, week_one, 1)
    assert score == 20
    assert len(info["campaigns"]) == 1
    assert info["campaigns"][0]["baseline_weeks"] == [2, 9]
    assert info["campaigns"][0]["campaign_weeks"] == [11, 12]

    score, info = _historical_benchmark(
        weekly, schedule[schedule.campaign_id != 8], offered, week_one, 1
    )
    assert score == 0 and info["source"] == "default" and info["campaigns"] == []


@pytest.mark.parametrize(
    ("simulated", "label", "inside"),
    [
        (1.71, "Calibrated", True),
        (6.24, "Calibrated", True),
        (-5.085, "Partially calibrated", False),
        (13.035, "Partially calibrated", False),
        (-5.086, "Uncalibrated", False),
        (13.036, "Uncalibrated", False),
    ],
)
def test_evaluate_backtest_boundaries(simulated, label, inside):
    result = evaluate_backtest(simulated)
    assert result == {
        "simulated": simulated,
        "observed_range": [1.71, 6.24],
        "naive": 19.02,
        "inside_range": inside,
        "abs_error_vs_mid": pytest.approx(abs(simulated - 3.975)),
        "beats_naive": True,
        "label": label,
    }


def test_evaluate_backtest_strict_improvement_and_finite_input():
    assert not evaluate_backtest(naive_benchmark())["beats_naive"]
    assert not evaluate_backtest(-11.07)["beats_naive"]  # Same error on the opposite side.
    assert not evaluate_backtest(100)["beats_naive"]
    for value in (float("nan"), float("inf"), -float("inf")):
        with pytest.raises(ValueError, match="finite"):
            evaluate_backtest(value)


def test_appeal_grid_rebuild_and_design_preferences():
    frame = loaders.load_appeal_fallback()
    pd.testing.assert_frame_equal(frame, build_appeal_fallback(loaders.load_personas()))
    values = frame.set_index(["archetype", "mechanic", "message_style"]).appeal
    archetypes = ["loyal", "deal_hunter", "waiter", "explorer", "indifferent"]
    mechanics = [
        "percent_off",
        "voucher",
        "bundle",
        "free_shipping",
        "flash_sale",
        "coins_cashback",
    ]
    styles = ["price", "value", "urgency", "family", "premium"]
    assert values.index.is_unique and set(values.index) == set(
        product(archetypes, mechanics, styles)
    )
    assert values.between(-1, 1).all() and set(frame.label) == {"default"}
    for name, top_styles in [("deal_hunter", {"price", "urgency"}), ("loyal", {"value", "family"})]:
        assert set(values[name].groupby("message_style").mean().nlargest(2).index) == top_styles
    assert values["waiter"].idxmax() == ("flash_sale", "urgency")


def test_appeal_changes_with_weighted_traits_and_promotion_share():
    personas = loaders.load_personas()
    changed = [
        {**p, "promo_share": 0, "traits": dict.fromkeys(p["traits"], 0)}
        if p["archetype"] == "deal_hunter"
        else p
        for p in personas
    ]
    current = loaders.load_appeal_fallback().groupby("archetype").appeal.mean()
    updated = build_appeal_fallback(changed).groupby("archetype").appeal.mean()
    assert updated["deal_hunter"] < current["deal_hunter"]
    assert updated.drop("deal_hunter").equals(current.drop("deal_hunter"))


def test_assumptions_cover_defaults_and_match_published_inputs():
    rows = loaders.load_assumptions()
    assert rows == build_assumptions()
    by_name = {r["name"]: r for r in rows}
    assert len(by_name) == len(rows)
    for row in rows:
        assert set(row) == {"name", "value", "unit", "label", "source", "used_by"}
        assert row["label"] in {"data", "default", "llm_estimate"}
        assert row["unit"] and row["source"] and isinstance(row["used_by"], list) and row["used_by"]
    for name, value in config.DEFAULTS.items():
        assert by_name[name]["value"] == value and by_name[name]["label"] == "default"

    market, econ = loaders.load_market(), loaders.load_unit_econ()
    for prefix, source_map in [
        ("market", market["source_notes"]["fields"]),
        ("unit_economics", econ["source"]),
    ]:
        uses = {usage for row in rows if row["label"] == "default" for usage in row["used_by"]}
        assert {
            f"{prefix}.{key}" for key, label in source_map.items() if label == "default"
        } <= uses
    assert by_name["unit_cost"]["value"] == market["unit_cost"] == econ["unit_cost"]
    assert by_name["collected_at"]["value"] is None
    assert by_name["price_elasticity_by_archetype"]["value"] == {
        s["name"]: s["price_elasticity"] for s in market["segments"]
    }
    assert by_name["pooled_price_elasticity"]["value"] == -1.8
    factors = by_name["income_elasticity_factors"]
    assert factors["value"] == market["source_notes"]["income_elasticity_factors"]["value"]
    assert factors["value"] == {"low": 1.2, "middle": 1.0, "high": 0.85}
    assert factors["label"] == "default" and "Maier et al. (2025)" in factors["source"]
    assert by_name["rival_default_probabilities"]["value"] == {
        "none": 0.45,
        "match": 0.30,
        "undercut": 0.15,
        "non_price": 0.10,
    }
    prior = loaders.load_rival_prior()
    weight = by_name["rival_blend_applied_weight"]["value"]
    assert weight == (0.5 if prior["source"] == "data+default" else 0)
    for kind, default in by_name["rival_default_probabilities"]["value"].items():
        empirical = prior["lift"].get(kind, 1 - sum(prior["lift"].values()))
        assert prior["p"][kind] == pytest.approx((1 - weight) * empirical + weight * default)
    for persona in loaders.load_personas():
        if persona["source"]["traits"] == "default":
            assert (
                persona["traits"] == by_name["twin_trait_defaults"]["value"][persona["archetype"]]
            )
    assert "simulated" in by_name["simulated_data_notice"]["value"]
