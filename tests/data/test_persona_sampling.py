import numpy as np
import pandas as pd
import pytest

from data.etl import personas
from data.etl.selection_bias import calibrate_selection, compare_demographic_groups

BEHAVIOR = (
    "weekly_buy_rate",
    "purchase_cycle_weeks",
    "avg_units",
    "avg_price_paid",
    "promo_share",
    "coupon_rate",
    "brand_loyalty",
)


@pytest.fixture
def buyers():
    """Fabricated complete and incomplete records with identifiable behaviour tuples."""
    rows = []
    clusters = list(personas.DEFAULT_TRAITS)
    archetypes = ["deal_hunter", "loyal", "waiter", "explorer", "indifferent"]
    for group, n in enumerate([40, 48, 64, 80, 96]):
        for j in range(n):
            known = j % 3 != 0
            size = j % 4 + 1
            rows.append(
                {
                    "household_id": 1000 + len(rows),
                    "cluster_label": clusters[group],
                    "archetype": archetypes[group],
                    "segment": archetypes[group],
                    "distance_to_center": j / n,
                    "price_elasticity": -(1 + group / 5),
                    "price_elasticity_observed": -0.2 * group,
                    "weekly_buy_rate": (1 + j % 10) / 44,
                    "purchase_cycle_weeks": 1 + j % 12,
                    "avg_units": 1 + j % 3,
                    "avg_price_paid": (5_000 + len(rows)) / 100,
                    "promo_share": (j % 11) / 10,
                    "coupon_rate": float(known == (j % 5 != 0)),
                    "brand_loyalty": 1 / (1 + j % 4),
                    "age": ["25-34", "35-44", "45-54"][j % 5 % 3] if known else None,
                    "income": ["25-34K", "35-49K", "75-99K"][j % 7 % 3] if known else None,
                    "household_size": str(size) if known else None,
                    "kids_count": str(size - 1) if known else None,
                    "home_ownership": (
                        ("Homeowner" if j % 2 else "Renter") if known and j % 7 else None
                    ),
                }
            )
    return pd.DataFrame(rows).set_index("household_id")


def test_household_sample_preserves_records_coverage_and_edge_definition(buyers, monkeypatch):
    monkeypatch.setattr(personas, "_survey_traits", lambda: None)
    selected = personas.sample_households(buyers)
    pd.testing.assert_frame_equal(
        selected, personas.sample_households(buyers.sample(frac=1, random_state=10))
    )
    assert len(selected) == selected.index.nunique() == 150
    assert selected["archetype"].value_counts().min() >= 20
    assert selected["_edge"].sum() == 15
    for cluster, edge in selected.loc[selected["_edge"]].groupby("cluster_label"):
        distances = buyers.loc[buyers["cluster_label"].eq(cluster), "distance_to_center"]
        assert edge["distance_to_center"].min() >= distances.quantile(0.90, interpolation="lower")
    cells = buyers[["age", "income"]].dropna().value_counts(normalize=True)
    assert set(cells[cells >= 0.02].index) <= set(selected[["age", "income"]].value_counts().index)

    profiles = personas.build_personas(buyers)
    weighted, audit = calibrate_selection(buyers, selected)
    assert profiles == personas.build_personas(buyers, weighted)
    assert audit["reweighting"]["applied"]
    sources = buyers.set_index("avg_price_paid", drop=False)
    assert len({p["avg_price_paid"] for p in profiles}) == len(profiles)
    for profile in profiles:
        row = sources.loc[profile["avg_price_paid"]]
        assert tuple(profile[k] for k in BEHAVIOR) == pytest.approx(
            tuple(row[k] for k in BEHAVIOR), abs=5.1e-7, rel=0
        )
        assert profile["archetype"] == row["archetype"]
        demo_fields = ("age_band", "income_band", "household_size", "kids", "homeowner")
        assert profile["price_elasticity_observed"] == pytest.approx(
            row["price_elasticity_observed"]
        )
        if pd.notna(row["age"]):
            assert (profile["age_band"], profile["income_band"]) == (row["age"], row["income"])
            assert profile["household_size"] == int(row["household_size"])
            assert profile["kids"] == int(row["kids_count"])
            assert all(profile["source"][key] == "data" for key in demo_fields[:-1])
            if pd.notna(row["home_ownership"]):
                assert profile["homeowner"] == (row["home_ownership"] == "Homeowner")
                assert profile["source"]["homeowner"] == "data"
            else:
                assert profile["source"]["homeowner"] == "default"
                assert profile["homeowner"] == (profile["household_size"] % 2 == 0)
        else:
            assert profile["age_band"] == profile["income_band"] == "Unknown"
            assert profile["price_elasticity"] == pytest.approx(row["price_elasticity"])
            assert all(profile["source"][key] == "default" for key in demo_fields)
            assert isinstance(profile["household_size"], int) and isinstance(profile["kids"], int)
            assert isinstance(profile["homeowner"], bool)
            assert profile["kids"] == profile["household_size"] - 1
            assert profile["homeowner"] == (profile["household_size"] % 2 == 0)
        assert "household_id" not in profile and "household_key" not in profile
    masses = pd.DataFrame(profiles).groupby("archetype")["weight"].sum().to_dict()
    assert masses == pytest.approx(buyers["archetype"].value_counts(normalize=True).to_dict())


def test_selection_bias_scores_and_trigger_at_exact_cutoff():
    # A difference at a single atom produces D=0.1 even though the supports overlap.
    known = [0.0] * 5 + [1.0] * 5
    missing = [0.0] * 6 + [1.0] * 4
    frame = pd.DataFrame(dict.fromkeys(BEHAVIOR, known + missing))
    frame["age"] = ["25-34"] * 10 + [None] * 10
    frame["income"] = ["25-34K"] * 10 + [None] * 10
    report = compare_demographic_groups(frame)
    assert report["counts"] == {"with_demographics": 10, "without_demographics": 10}
    assert set(report["metrics"]) == set(BEHAVIOR)
    assert not report["needs_reweighting"]
    frame.loc[19, "coupon_rate"] = 0.0
    report = compare_demographic_groups(frame)
    assert report["triggered_attributes"] == ["coupon_rate"]
    assert report["needs_reweighting"]
    values = report["metrics"]["coupon_rate"]
    assert values["ks_similarity"] == pytest.approx(0.8)
    assert values["mean_with_demographics"] == pytest.approx(0.5)
    assert values["mean_without_demographics"] == pytest.approx(0.3)
    assert values["mean_difference"] == pytest.approx(0.2)


def test_reweighting_only_when_needed(buyers):
    selected = personas.sample_households(buyers)
    constant = buyers.copy()
    constant[list(BEHAVIOR)] = 1.0
    sample = selected.copy()
    sample[list(BEHAVIOR)] = 1.0
    adjusted, report = calibrate_selection(constant, sample)
    assert not report["needs_reweighting"] and not report["reweighting"]["applied"]
    np.testing.assert_allclose(adjusted["weight"], sample["weight"])

    # Distort the coupon mix within demographic strata, preserving their total masses.
    strata = selected["_weight_stratum"]
    masses = selected.groupby(strata)["weight"].sum()
    distorted = selected["weight"] * np.where(selected["coupon_rate"] == 1, 1, 15)
    selected["weight"] = distorted / distorted.groupby(strata).transform("sum") * strata.map(masses)
    adjusted, report = calibrate_selection(buyers, selected)
    assert report["needs_reweighting"] and report["reweighting"]["applied"]
    assert (adjusted["weight"] > 0).all() and adjusted["weight"].sum() == pytest.approx(1)
    target = buyers["coupon_rate"].mean()
    before = np.average(selected["coupon_rate"], weights=selected["weight"])
    after = np.average(adjusted["coupon_rate"], weights=adjusted["weight"])
    assert abs(after - target) < abs(before - target)
    np.testing.assert_allclose(adjusted.groupby(strata)["weight"].sum(), masses)
    rebuilt = selected["weight"].copy()
    for field, factors in report["reweighting"]["factors"].items():
        bins = np.searchsorted(factors["upper_cuts_inclusive"], selected[field].round(6))
        rebuilt *= np.asarray(factors["multiplier"])[bins]
    rebuilt *= strata.map(masses) / rebuilt.groupby(strata).transform("sum")
    np.testing.assert_allclose(rebuilt, adjusted["weight"])
