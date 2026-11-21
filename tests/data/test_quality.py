import json

import numpy as np
import pandas as pd
import pytest

from data import loaders, quality
from data.etl.build_tissue import _calibration


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
        "avg_price_paid": 12.5,
        "promo_share": 0.5,
        "coupon_rate": 0.5,
        "brand_loyalty": 0.75,
    }
    assert values.loc["tt-hh-0002"].to_dict() == {
        "weekly_buy_rate": 0.25,
        "purchase_cycle_weeks": 4,
        "avg_price_paid": 12,
        "promo_share": 1,
        "coupon_rate": 0,
        "brand_loyalty": 1,
    }
    with pytest.raises(ValueError, match="missing household weeks"):
        quality.household_attributes(panel.iloc[1:])


def test_weighted_ks_sd_and_correlation_match_replicated_observations():
    # Repeating the first persona three times gives its 3:1:1 population mass.
    rows = np.array([[0, 3, 0, 1, 0, 0], [0.5, 1, 3, 0, 0, 1], [1, 4, 1, 1, 1, 0]])
    personas = pd.DataFrame(rows, columns=quality.ATTRIBUTES).assign(weight=[3.0, 1.0, 1.0])
    expanded = np.repeat(rows, [3, 1, 1], axis=0)
    reference = pd.DataFrame(rows, columns=quality.ATTRIBUTES)
    marginal, variance, corr = quality._reference_checks(personas, reference)
    assert marginal["weekly_buy_rate"]["profile_value"] == pytest.approx(1 - (3 / 5 - 1 / 3))
    assert marginal["weekly_buy_rate"]["status"] == "fail"
    assert variance["value"] == pytest.approx(expanded[:, 0].std() / rows[:, 0].std())
    for data, key in [(expanded, "persona_mean_abs_corr"), (rows, "household_mean_abs_corr")]:
        rs = np.corrcoef(data, rowvar=False)
        assert corr[key] == pytest.approx(np.mean(np.abs(rs[np.triu_indices(6, k=1)])))
    assert corr["gap"] == pytest.approx(
        abs(corr["persona_mean_abs_corr"] - corr["household_mean_abs_corr"])
    )
    assert (
        quality._ks_similarity(
            np.array([0.1, 0.1, 0.9]), np.array([2, 1, 1]), np.array([0.1, 0.1, 0.1, 0.9])
        )
        == 1
    )
    constant = reference.assign(weekly_buy_rate=0.5)
    _, variance, corr = quality._reference_checks(personas, constant)
    assert variance["status"] == corr["status"] == "skipped"
    assert variance["value"] is corr["gap"] is None


def test_directions_use_weighted_lower_slopes_and_numeric_income_bands():
    personas = pd.DataFrame(
        {
            "archetype": ["deal_hunter", "loyal", "loyal", "loyal", "loyal"],
            "income_band": ["Under 15K", "25-34K", "35-49K", "75-99K", "250K+"],
            "weight": [4, 1, 100, 1, 1],
            "price_elasticity": [-3.0, -0.5, -1, -1, -1],
            "price_elasticity_observed": [-0.1, -0.4, -5, -0.5, -0.5],
        }
    )
    checks = quality._direction_checks(personas)
    assert checks["deal_hunter_vs_loyal"]["status"] == "pass"
    assert checks["income_observed"]["status"] == "fail"
    assert checks["income_observed"]["low"] == pytest.approx(-0.16)
    assert checks["income_prior"]["status"] == "pass"
    assert checks["income_prior"]["low"] == pytest.approx(-2.5)
    assert checks["income_prior"]["high"] == -1


def test_population_ks_uses_seeded_weighted_draws_and_bounded_independent_jitter():
    profiles = pd.DataFrame(
        [[0.4, 1, 10, 0, 0.6, 1], [0.8, 9, 100, 0.3, 0.6, 0.8]],
        columns=quality.ATTRIBUTES,
    ).assign(weight=[8, 2])
    original = profiles.copy(deep=True)
    agents = quality._expand_population(profiles)
    assert len(agents) == 5_000
    pd.testing.assert_frame_equal(agents, quality._expand_population(profiles))
    pd.testing.assert_frame_equal(profiles, original)
    draws = np.random.default_rng(42).choice(2, size=5_000, p=[0.8, 0.2])
    sampled = profiles.loc[draws, list(quality.ATTRIBUTES)].reset_index(drop=True)
    assert (agents["avg_price_paid"].gt(50).to_numpy() == (draws == 1)).all()
    assert (agents >= 0.9 * sampled).all().all() and (agents <= 1.1 * sampled).all().all()
    assert agents["purchase_cycle_weeks"].min() == 1
    assert agents["brand_loyalty"].max() == 1
    assert (agents.loc[draws == 0, "promo_share"] == 0).all()
    factors = agents[["weekly_buy_rate", "avg_price_paid"]] / sampled[
        ["weekly_buy_rate", "avg_price_paid"]
    ]
    assert abs(factors.corr().iloc[0, 1]) < 0.05
    # The six-decimal profile distribution matches these households exactly.
    households = profiles.iloc[[0, 0, 0, 0, 1]]
    marginal, _, _ = quality._reference_checks(profiles, households)
    assert marginal["coupon_rate"]["profile_value"] == pytest.approx(1)
    assert marginal["coupon_rate"]["status"] == "fail"
    # All reference coupon rates are 0.6; the larger CDF gap occurs on one side of this atom.
    rates = agents["coupon_rate"].round(6)
    expected = 1 - max((rates < 0.6).mean(), (rates > 0.6).mean())
    assert marginal["coupon_rate"]["value"] == pytest.approx(expected)


def test_adjusted_priors_pass_both_directions_on_published_personas():
    checks = quality._direction_checks(pd.DataFrame(loaders.load_personas()))
    assert checks["income_prior"]["low"] < checks["income_prior"]["high"]
    assert checks["income_prior"]["status"] == "pass"
    assert checks["deal_hunter_vs_loyal"]["deal_hunter"] < checks["deal_hunter_vs_loyal"]["loyal"]
    assert checks["deal_hunter_vs_loyal"]["status"] == "pass"


def test_missing_panel_preserves_badge_inputs_and_reports_skips(monkeypatch, tmp_path, capsys):
    # Resolve persona data before redirecting the published report to the test directory.
    personas = loaders.load_personas()
    monkeypatch.setattr(loaders, "load_personas", lambda: personas)
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
    assert all(
        m == {"value": None, "profile_value": None, "status": "skipped"}
        for m in report["marginal_fit"].values()
    )
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
    assert set(report["marginal_fit"]) == set(quality.ATTRIBUTES)
    for metric in report["marginal_fit"].values():
        if metric["value"] is None:
            assert metric["status"] == "skipped" and report["reference"]["status"] != "available"
            continue
        assert 0 <= metric["value"] <= 1 and 0 <= metric["profile_value"] <= 1
        assert metric["status"] == ("pass" if metric["value"] >= 0.9 else "fail")
    ratio, gap = report["variance_ratio"], report["over_correlation"]
    if ratio["value"] is not None:
        assert ratio["status"] == ("pass" if ratio["value"] >= 0.8 else "fail")
    if gap["gap"] is not None:
        assert gap["status"] == ("pass" if gap["gap"] <= 0.1 else "fail")
    counts = report["check_counts"]
    assert sum(counts.values()) == 11
    assert report["status"] == (
        "fail" if counts["fail"] else "skipped" if counts["skipped"] else "pass"
    )
    fresh = quality.build_eval_report()
    for key in ("direction_checks", "rival_evidence", "data_coverage", "confidence_inputs"):
        assert fresh[key] == report[key]
    if report["reference"]["status"] == fresh["reference"]["status"] == "available":
        for attr in quality.ATTRIBUTES:
            for key in ("value", "profile_value"):
                assert fresh["marginal_fit"][attr][key] == pytest.approx(
                    report["marginal_fit"][attr][key]
                )
        assert fresh["over_correlation"]["gap"] == pytest.approx(gap["gap"])
        assert fresh["variance_ratio"]["value"] == pytest.approx(ratio["value"])
    assert report["confidence_inputs"]["backtest_label"] is None
