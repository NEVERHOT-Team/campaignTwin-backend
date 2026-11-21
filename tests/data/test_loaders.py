import json
from collections import Counter, defaultdict

import pandas as pd
import pytest

from data import loaders
from data.etl.personas import BEHAVIOR, TRAITS, _survey_traits


def test_segments_and_personas():
    segments = loaders.load_segments()
    assert segments["households"].sum() == 1514
    assert segments["households"].min() >= 50
    assert segments["share"].sum() == pytest.approx(1)
    assert (segments["price_elasticity"] < 0).all()
    assert set(segments["source"]) == {"data"}

    personas = loaders.load_personas()
    assert len(personas) == 60
    assert len({persona["persona_id"] for persona in personas}) == 60
    assert sum(persona["persona_id"].startswith("tt-edge-") for persona in personas) == 6
    assert sum(persona["weight"] for persona in personas) == pytest.approx(1)
    assert {persona["segment"] for persona in personas} == set(segments["segment_id"])
    archetype_counts = Counter(persona["archetype"] for persona in personas)
    assert len(archetype_counts) == 5
    assert min(archetype_counts.values()) >= 10
    assert len({tuple(persona[field] for field in BEHAVIOR) for persona in personas}) == 60
    expected_weights = segments.set_index("segment_id")["share"].to_dict()
    observed_weights = defaultdict(float)
    for persona in personas:
        observed_weights[persona["segment"]] += persona["weight"]
        assert persona["age_band"] and persona["income_band"] and persona["weight"] > 0
        assert 0 <= persona["kids"] < persona["household_size"]
        assert isinstance(persona["homeowner"], bool)
        assert persona["price_elasticity"] < 0
        assert 0 < persona["weekly_buy_rate"] <= 1
        assert persona["purchase_cycle_weeks"] >= 1
        assert persona["avg_units"] >= 1
        assert persona["avg_price_paid"] >= 0
        assert all(
            0 <= persona[field] <= 1 for field in ("promo_share", "coupon_rate", "brand_loyalty")
        )
        assert set(persona["traits"]) == set(TRAITS)
        assert all(0 <= value <= 1 for value in persona["traits"].values())
        assert persona["source"]["behavior"] == "data"
        assert persona["source"]["traits"] in {"data", "default"}
        assert persona["source"]["backstory"] == "llm_estimate"
    assert dict(observed_weights) == pytest.approx(expected_weights)


def test_market_and_knobs_agree_with_the_legacy_loaders():
    market = loaders.load_market()
    economics = loaders.load_unit_econ()
    assert market["category"] == economics["category"] == "toilet_tissue"
    assert market["currency"] == economics["currency"] == "THB"
    assert market["ref_price"] == economics["price_thb"] > 0
    assert market["unit_cost"] == economics["unit_cost"]
    assert market["shipping_cost_per_order"] == economics["shipping_cost_per_order"]
    assert market["margin_floor_pct"] == economics["margin_floor_pct"]
    assert market["platform_fee_pct"] == economics["shopee_commission_pct"]
    assert market["payment_fee_pct"] == economics["shopee_txn_fee_pct"]
    assert economics["collected_at"] is None
    assert economics["source"]["collected_at"] == "default"
    assert market["weeks"] == {"campaign": 4, "post": 2}
    manufacturers = market["source_notes"]["manufacturers"]
    assert {key: value["role"] for key, value in manufacturers.items()} == {
        "751": "our_brand",
        "764": "leader",
        "595": "peer",
        "69": "private_label",
    }
    assert market["source_notes"]["usd_thb"] == {"value": economics["usd_thb"], "source": "default"}
    fields = market["source_notes"]["fields"]
    for field in ("platform_fee_pct", "payment_fee_pct", "shipping_cost_per_order"):
        assert fields[field] == "default"

    expected_segments = loaders.load_segments()[
        [
            "name",
            "archetype",
            "cluster_label",
            "share",
            "weekly_purchase_prob",
            "households",
            "price_elasticity",
            "price_elasticity_observed",
        ]
    ].rename(columns={"weekly_purchase_prob": "weekly_buy_rate"})
    assert market["segments"] == expected_segments.to_dict("records")
    knobs = loaders.load_knobs()
    levels = [knobs["discount_pct"][key] for key in ("min", "p50", "p90", "max")]
    assert levels == sorted(levels)
    assert 0 < levels[0] <= levels[-1] == 50
    assert knobs["discount_pct"]["raw_max"] == 100
    assert knobs["source"]["discount_pct"] == "data"
    assert knobs["source"]["discount_pct.max"] == "default"
    assert knobs["source"]["discount_pct.raw_max"] == "data"
    assert {"12.12", "payday"} <= set(knobs["calendar"])
    assert knobs["duration_days"]["min"] <= knobs["duration_days"]["max"]
    assert knobs["cap_per_user"]["min"] <= knobs["cap_per_user"]["max"]


def test_promotional_elasticity_prior_and_observed_slopes():
    segments = loaders.load_segments().set_index("archetype")
    assert set(segments.index) == {"deal_hunter", "waiter", "loyal", "explorer", "indifferent"}
    assert segments["cluster_label"].nunique() == 5
    prior = segments["price_elasticity"]
    assert prior["deal_hunter"] < prior["loyal"]
    promo_rank = (segments["promo_response"] + segments["coupon_rate"]).sort_values().index
    assert prior.loc[promo_rank].tolist() == pytest.approx([-0.8, -1.3, -1.8, -2.3, -2.8])
    assert set(segments["price_elasticity_source"]) == {"default"}
    assert set(segments["price_elasticity_observed_source"]) == {"data"}
    notes = loaders.load_market()["source_notes"]
    assert notes["pooled_price_elasticity"]["value"] == pytest.approx(-1.8)
    assert notes["pooled_price_elasticity"]["source"] == "default"
    assert notes["pooled_price_elasticity_observed"]["source"] == "data"
    for persona in loaders.load_personas():
        segment = segments.loc[persona["archetype"]]
        assert persona["cluster_label"] == segment["cluster_label"]
        assert persona["price_elasticity"] == pytest.approx(segment["price_elasticity"])
        assert persona["price_elasticity_observed"] == pytest.approx(
            segment["price_elasticity_observed"], abs=1e-6
        )
        assert persona["source"]["price_elasticity"] == "default"
        assert persona["source"]["price_elasticity_observed"] == "data"


def test_economics_anchors_and_backtest():
    economics = loaders.load_unit_econ()
    assert economics["currency"] == "THB"
    assert economics["price_thb"] > 0
    assert economics["source"]["price_thb"] == "data"
    assert economics["source"]["usd_thb"] == "default"

    anchors = loaders.load_anchors()
    assert set(anchors) == {"en", "th"}
    assert all(len(anchors[language][0]) == 5 for language in anchors)

    backtest = loaders.load_backtest()
    assert set(backtest["actual"]) == {"lift_pct", "redemption_rate", "incremental_sales"}
    assert backtest["source"] == "default"
    assert max(backtest["calibration_weeks"]) < backtest["start_week"]


def test_rival_data_loaders():
    library = loaders.load_library()
    assert {"campaign_id", "reaction_to_campaign_id", "channel_mix", "source_url"} <= set(library)
    assert set(library["reaction_type"]) <= {"", "none", "match", "undercut", "non_price"}
    brands = loaders.load_brands()
    assert set(brands["brand"]) == {"751", "764", "595", "69", "1011", "1315"}
    assert brands.loc[brands["is_ours"], "brand"].tolist() == ["751"]
    assert brands["market_share"].sum() == pytest.approx(1, abs=0.001)
    assert set(brands["price_tier"]) == {"low", "mid", "high"}
    prior = loaders.load_rival_prior()
    assert set(prior["p"]) == {"none", "match", "undercut", "non_price"}
    assert sum(prior["p"].values()) == pytest.approx(1)
    assert prior["source"] == ("data" if prior["n_events"] >= 20 else "default")


def test_loader_names_missing_keys(monkeypatch, tmp_path):
    path = tmp_path / "reaction_base_rate.json"
    path.write_text(json.dumps({"toilet_tissue": {"none": 1}}))
    monkeypatch.setattr(loaders, "PROCESSED", tmp_path)
    loaders.load_rival_prior.cache_clear()
    with pytest.raises(ValueError, match=r"reaction_base_rate.json missing required.*n_events"):
        loaders.load_rival_prior()


def test_local_calibration_loader(monkeypatch, tmp_path):
    monkeypatch.setattr(loaders, "CALIB", tmp_path)
    with pytest.raises(FileNotFoundError, match=r"household_behavior.parquet.*build_tissue"):
        loaders.load_household_behavior()
    record = {
        "household_key": "local-demo",
        "archetype": "deal_hunter",
        "segment": "deal_hunter",
        "week": 1,
        "bought": True,
        "units": 2,
        "price_paid": 100.0,
        "shelf_price": 125.0,
        "on_promo": True,
        "discount_pct": 20.0,
        "coupon_used": False,
        "in_mailer": True,
        "on_display": False,
        "brand": 751,
    }
    path = tmp_path / "household_behavior.parquet"
    pd.DataFrame([record]).to_parquet(path, index=False)
    assert loaders.load_household_behavior().to_dict("records") == [record]
    pd.DataFrame([record]).drop(columns="brand").to_parquet(path, index=False)
    with pytest.raises(ValueError, match=r"household_behavior.parquet.*brand"):
        loaders.load_household_behavior()


def test_optional_twin_csv_uses_numeric_public_export(monkeypatch, tmp_path):
    monkeypatch.setattr(loaders, "RAW", tmp_path)
    assert _survey_traits() is None
    response = {"QID32": 3}
    response.update({f"QID84_{row}": 2 for row in range(4, 13)})
    response.update({f"QID250_{row}": 1 for row in range(1, 15)})
    pd.DataFrame([response] * 12).to_csv(tmp_path / "wave1_3_response.csv", index=False)
    traits = _survey_traits()
    assert traits.iloc[0].to_dict() == {
        "impulsiveness": 0.5,
        "risk_aversion": 0.0,
        "time_preference": 1.0,
    }
