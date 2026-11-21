import json
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from data import loaders, quality
from data.etl.build_tissue import _calibration
from data.etl.selection_bias import ks_similarity
from engine import shopper


@pytest.fixture
def panel():
    # Household A buys two brands in the first week; the weekly leader alone loses one unit.
    households = pd.DataFrame(
        {"archetype": "loyal", "segment": "loyal", "brand_loyalty": [0.75, 1.0]},
        index=pd.Index([1, 2], name="household_id"),
    )
    weekly = pd.DataFrame(
        [
            (1, 1, 3, 30, False, False, 751),
            (1, 3, 1, 20, True, True, 751),
            (2, 2, 1, 12, True, False, 595),
        ],
        columns=["household_id", "week", "units", "paid_total", "on_promo", "coupon_used", "brand"],
    )
    weekly = weekly.assign(shelf_total=weekly.paid_total, in_mailer=False, on_display=False)
    return _calibration(
        households, weekly.set_index(["household_id", "week"]), pd.Index([1, 2, 3, 4], name="week")
    )


def test_panel_denominators_singletons_and_exact_mixed_brand_loyalty(panel):
    values = quality.household_attributes(panel.sample(frac=1, random_state=7))
    assert values.loc["tt-hh-0001"].to_dict() == {
        "weekly_buy_rate": 0.5,
        "purchase_cycle_weeks": 2,
        "avg_units": 2,
        "avg_price_paid": 12.5,
        "promo_share": 0.5,
        "coupon_rate": 0.5,
        "brand_loyalty": 0.75,
    }
    assert values.loc["tt-hh-0002"].to_dict() == {
        "weekly_buy_rate": 0.25,
        "purchase_cycle_weeks": 4,
        "avg_units": 1,
        "avg_price_paid": 12,
        "promo_share": 1,
        "coupon_rate": 0,
        "brand_loyalty": 1,
    }
    with pytest.raises(ValueError, match="missing household weeks"):
        quality.household_attributes(panel.iloc[1:])


def test_weighted_profile_ks_and_engine_sd_and_correlation():
    fields = [
        "weekly_buy_rate",
        "purchase_cycle_weeks",
        "avg_units",
        "promo_share",
        "brand_loyalty",
    ]
    rows = np.array([[0, 3, 2, 1, 0], [0.5, 1, 3, 0, 1], [1, 4, 1, 1, 1]])
    households = pd.DataFrame(rows, columns=fields).assign(coupon_rate=0.25, avg_price_paid=10)
    personas = households.assign(weight=[3.0, 1.0, 1.0], price_elasticity=[-0.8, -1.8, -2.8])
    agents = personas.iloc[[0, 1, 1, 1, 2, 2]].reset_index(drop=True)
    marginal, variance, corr = quality._reference_checks(personas, households, agents)
    score = marginal["weekly_buy_rate"]
    assert score["profile_value"] == pytest.approx(1 - (3 / 5 - 1 / 3))
    assert score["value"] == pytest.approx(1 - (1 / 3 - 1 / 6))
    assert score["status"] == "fail"
    assert variance["value"] == pytest.approx(agents.weekly_buy_rate.std(ddof=0) / rows[:, 0].std())
    assert variance["persona_sd"] == pytest.approx(np.repeat(rows[:, 0], [3, 1, 1]).std())
    for data, key in [(agents, "engine_mean_abs_corr"), (households, "household_mean_abs_corr")]:
        r = np.corrcoef(data[fields], rowvar=False)
        assert corr[key] == pytest.approx(np.mean(np.abs(r[np.triu_indices(5, k=1)])))
    assert corr["gap"] == pytest.approx(
        abs(corr["engine_mean_abs_corr"] - corr["household_mean_abs_corr"])
    )
    constant = households.assign(weekly_buy_rate=0.5)
    _, variance, corr = quality._reference_checks(personas, constant, agents)
    assert variance["status"] == corr["status"] == "skipped"
    assert variance["value"] is corr["gap"] is None


@pytest.fixture
def audited_population(monkeypatch, panel):
    profiles = [
        {
            "persona_id": name,
            "archetype": name,
            "segment": name,
            "income_band": income,
            "weight": weight,
            "weekly_buy_rate": rate,
            "purchase_cycle_weeks": cycle,
            "avg_units": units,
            "promo_share": promo,
            "brand_loyalty": loyalty,
            "coupon_rate": 0.25,
            "avg_price_paid": 10,
            "price_elasticity": prior,
            "price_elasticity_observed": observed,
        }
        for name, income, weight, rate, cycle, units, promo, loyalty, prior, observed in [
            ("deal_hunter", "25-34K", 0.3, 0.31, 1.15, 1, 0.69, 0.23, -3.0, -0.15),
            ("loyal", "75-99K", 0.6, 0.47, 2.1, 3, 0.11, 0.84, -1.0, -0.25),
            ("waiter", "Under 15K", 0.1, 0.09, 6, 2, 0.56, 0.47, -2.5, -0.05),
        ]
    ]
    monkeypatch.setattr(loaders, "load_personas", lambda: profiles)
    # The engine's order differs deliberately from the order used by the auditor.
    monkeypatch.setattr(shopper, "load_personas", lambda: profiles[::-1])
    monkeypatch.setattr(shopper, "load_market", lambda: {"ref_price": 10})
    monkeypatch.setattr(shopper, "market_shares", lambda: (0.4, 0.3, 0.3))
    monkeypatch.setattr(loaders, "load_household_behavior", lambda: panel)
    original_builder = shopper.build_population
    expected_population = original_builder(n_agents=5_000, seed=42)
    builder = Mock(wraps=original_builder)
    ks = Mock(wraps=quality.ks_similarity)
    monkeypatch.setattr(shopper, "build_population", builder)
    monkeypatch.setattr(quality, "ks_similarity", ks)
    report = quality.build_eval_report()
    builder.assert_called_once_with(n_agents=5_000, seed=42)
    return report, expected_population, pd.DataFrame(profiles), ks


def test_marginals_receive_exact_engine_arrays(audited_population, panel):
    report, population, _, ks = audited_population
    reference = quality.household_attributes(panel)
    scores = report["marginal_fit"]
    attrs = {
        "weekly_buy_rate": population.weekly_buy_rate,
        "purchase_cycle_weeks": population.cycle_weeks,
        "avg_units": population.avg_units,
        "promo_share": population.promo_share,
        "brand_loyalty": population.loyalty,
    }
    for name, values in attrs.items():
        observed = reference[name].to_numpy()
        assert any(
            np.array_equal(call.args[0], values) and np.array_equal(call.args[2], observed)
            for call in ks.call_args_list
        ), name
        expected = ks_similarity(values, np.ones(5_000), observed)
        assert scores[name]["value"] == pytest.approx(expected)


def test_profile_only_diagnostics_and_configured_elasticity_reference(audited_population, panel):
    report, population, profiles, _ = audited_population
    scores = report["marginal_fit"]
    reference = quality.household_attributes(panel)
    for field in ("coupon_rate", "avg_price_paid"):
        row = scores[field]
        assert row["status"] == "profile_only" and row["value"] is row["target"] is None
        expected = ks_similarity(
            profiles[field].to_numpy(),
            profiles["weight"].to_numpy(),
            reference[field].to_numpy(),
        )
        assert row["profile_value"] == pytest.approx(expected)
    assert "not used by engine" in scores["coupon_rate"]["note"].lower()
    prior = scores["price_elasticity"]
    assert prior["status"] == "diagnostic" and prior["target"] is prior["profile_value"] is None
    expected = ks_similarity(
        profiles["price_elasticity"].to_numpy(),
        profiles["weight"].to_numpy(),
        population.elasticity,
    )
    assert prior["value"] == pytest.approx(expected)
    assert prior["engine_mean"] == pytest.approx(population.elasticity.mean())
    assert prior["engine_median"] == pytest.approx(np.median(population.elasticity))
    assert "prior" in prior["note"].lower() and "household" in prior["note"].lower()


def test_engine_directions_join_sampled_persona_ids_and_observed_slopes_remain_weighted(
    audited_population,
):
    report, population, _, _ = audited_population
    checks = report["direction_checks"]
    assert checks["income_observed"]["population"] == "weighted_profiles"
    assert checks["income_observed"]["source"] == "data"
    for name in ("income_prior", "deal_hunter_vs_loyal"):
        assert checks[name]["population"] == "engine_agents" and checks[name]["source"] == "default"
    ids = np.asarray(population.persona_ids)[population.persona_idx]
    for role in ("deal_hunter", "loyal"):
        assert checks["deal_hunter_vs_loyal"][role] == pytest.approx(
            population.elasticity[ids == role].mean()
        )
    assert checks["income_prior"]["low"] == pytest.approx(
        population.elasticity[ids != "loyal"].mean()
    )
    assert checks["income_prior"]["high"] == pytest.approx(
        population.elasticity[ids == "loyal"].mean()
    )
    assert checks["income_observed"]["low"] == pytest.approx((-0.15 * 0.3 - 0.05 * 0.1) / 0.4)
    assert checks["income_observed"]["status"] == "fail"
    assert checks["income_prior"]["status"] == checks["deal_hunter_vs_loyal"]["status"] == "pass"


def test_missing_panel_preserves_badge_inputs_and_reports_skips(monkeypatch, tmp_path, capsys):
    # Resolve persona data before redirecting the published report to the test directory.
    personas = loaders.load_personas()
    monkeypatch.setattr(loaders, "load_personas", lambda: personas)
    monkeypatch.setattr(shopper, "load_personas", lambda: personas)
    monkeypatch.setattr(shopper, "load_market", lambda: {"ref_price": 50})
    monkeypatch.setattr(shopper, "market_shares", lambda: (0.3, 0.4, 0.3))
    monkeypatch.setattr(loaders, "CALIB", tmp_path)
    monkeypatch.setattr(loaders, "PROCESSED", tmp_path)
    monkeypatch.setattr(
        loaders,
        "load_library",
        lambda: pd.DataFrame(
            {
                "source_url": ["https://example.org/a", " ", "https://example.org/b", None],
                "reaction_type": ["none", "match", "", ""],
            }
        ),
    )
    monkeypatch.setattr(
        loaders,
        "load_assumptions",
        lambda: [{"label": label} for label in ("data", "default", "default", "llm_estimate")],
    )
    quality.main()
    assert "household_behavior.parquet is missing" in capsys.readouterr().out
    report = loaders.load_eval_report()
    assert report["reference"]["status"] == "skipped"
    assert report["selection_bias"]["status"] == "skipped"
    marginal = report["marginal_fit"]
    engine_fields = (
        "avg_units",
        "promo_share",
        "brand_loyalty",
        "purchase_cycle_weeks",
        "weekly_buy_rate",
    )
    for field in engine_fields:
        assert marginal[field]["status"] == "skipped"
        assert marginal[field]["value"] is marginal[field]["profile_value"] is None
    for field in ("coupon_rate", "avg_price_paid"):
        assert marginal[field]["profile_value"] is marginal[field]["value"] is None
    assert marginal["price_elasticity"]["status"] == "diagnostic"
    assert marginal["price_elasticity"]["value"] is not None
    assert report["variance_ratio"]["status"] == report["over_correlation"]["status"] == "skipped"
    assert all(d["status"] != "skipped" for d in report["direction_checks"].values())
    assert report["rival_evidence"] == {
        "total_rows": 4,
        "with_source_url": 2,
        "with_known_reaction_type": 2,
        "with_both": 1,
    }
    assert report["data_coverage"]["shares_pct"] == {"data": 25, "default": 50, "llm_estimate": 25}
    assert report["confidence_inputs"] == {
        "coverage_pct": 25,
        "rival_evidence_n": 1,
        "backtest_label": None,
    }


def test_legacy_panel_marks_unrecoverable_loyalty_and_correlation(monkeypatch, panel):
    monkeypatch.setattr(
        loaders, "load_household_behavior", lambda: panel.drop(columns="brand_loyalty")
    )
    report = quality.build_eval_report()
    assert report["reference"]["status"] == "partial"
    assert report["marginal_fit"]["brand_loyalty"]["value"] is None
    assert report["over_correlation"]["status"] == "skipped"
    assert report["marginal_fit"]["weekly_buy_rate"]["value"] is not None
    assert "tt-hh-" not in json.dumps(report, allow_nan=False)


def test_published_report_thresholds_and_source_counts():
    report = loaders.load_eval_report()
    assert report["reference"]["status"] in {"available", "partial", "skipped"}
    assert report["persona_count"] == 150
    audit = report["selection_bias"]
    # Campaign 18 is held out. Using all 52 weeks would include 646 known households.
    assert audit["counts"] == {"with_demographics": 621, "without_demographics": 796}
    assert audit["threshold"] == 0.9
    assert set(audit["metrics"]) == set(report["marginal_fit"]) - {"price_elasticity"}
    triggered = {
        name for name, row in audit["metrics"].items() if round(row["ks_similarity"], 12) < 0.9
    }
    assert set(audit["triggered_attributes"]) == triggered
    assert audit["needs_reweighting"] == audit["reweighting"]["applied"] == bool(triggered)
    for row in audit["metrics"].values():
        assert row["mean_difference"] == pytest.approx(
            row["mean_with_demographics"] - row["mean_without_demographics"]
        )
    # Keep the historical comparisons visible beside the engine measurements.
    historical = {
        "weekly_buy_rate": (0.6995, 0.8384),
        "purchase_cycle_weeks": (0.6977, 0.8372),
        "avg_units": (0.4323, 0.6317),
        "avg_price_paid": (0.7719, 0.9552),
        "promo_share": (0.6390, 0.7786),
        "coupon_rate": (0.6107, 0.9779),
        "brand_loyalty": (0.5358, 0.7472),
    }
    assert set(report["marginal_fit"]) == {*historical, "price_elasticity"}
    for field, (averaged, own_jitter) in historical.items():
        row = report["marginal_fit"][field]
        assert round(row["averaged_60"], 4) == averaged
        assert round(row["own_jitter_150"], 4) == own_jitter
        if field in {"avg_price_paid", "coupon_rate"}:
            assert row["target"] is row["value"] is None and row["status"] == "profile_only"
            continue
        threshold = 0.75 if field == "brand_loyalty" else 0.85
        assert row["target"] == threshold
        if row["value"] is None:
            assert row["status"] == "skipped" and report["reference"]["status"] != "available"
        else:
            assert 0 <= row["value"] <= 1 and 0 <= row["profile_value"] <= 1
            assert row["status"] == ("pass" if row["value"] >= threshold else "fail")
    diagnostic = report["marginal_fit"]["price_elasticity"]
    assert diagnostic["averaged_60"] is diagnostic["own_jitter_150"] is diagnostic["target"] is None
    assert (
        diagnostic["reference"] == "weighted_persona_prior" and diagnostic["status"] == "diagnostic"
    )
    ratio, gap = report["variance_ratio"], report["over_correlation"]
    if ratio["value"] is not None:
        assert ratio["status"] == ("pass" if ratio["value"] >= 0.8 else "fail")
    if gap["gap"] is not None:
        assert gap["status"] == ("pass" if gap["gap"] <= 0.1 else "fail")
    counts = report["check_counts"]
    assert sum(counts.values()) == 10
    assert report["status"] == (
        "fail" if counts["fail"] else "skipped" if counts["skipped"] else "pass"
    )
    fresh = quality.build_eval_report()
    for key in ("direction_checks", "rival_evidence", "data_coverage", "confidence_inputs"):
        assert fresh[key] == report[key]
    if report["reference"]["status"] == fresh["reference"]["status"] == "available":
        for attr in report["marginal_fit"]:
            for key in ("value", "profile_value"):
                assert fresh["marginal_fit"][attr][key] == pytest.approx(
                    report["marginal_fit"][attr][key]
                )
        assert fresh["over_correlation"]["gap"] == pytest.approx(gap["gap"])
        assert fresh["variance_ratio"]["value"] == pytest.approx(ratio["value"])
    assert report["confidence_inputs"]["backtest_label"] is None
