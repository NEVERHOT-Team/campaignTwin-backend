import json
from collections import Counter, defaultdict
from datetime import date

import pandas as pd
import pytest

from contracts.campaign_spec import CampaignSpec
from data import loaders
from data.etl.personas import TRAITS, _survey_traits
from data.rival import get_rival_scenarios


def test_segments_and_personas():
    segments = loaders.load_segments()
    assert segments["households"].sum() == 1417
    assert segments["households"].min() >= 50
    assert segments["share"].sum() == pytest.approx(1)
    assert (segments["price_elasticity"] < 0).all()
    assert set(segments["source"]) == {"data"}

    personas = loaders.load_personas()
    assert len(personas) == 150
    assert len({persona["persona_id"] for persona in personas}) == 150
    assert sum(persona["persona_id"].startswith("tt-edge-") for persona in personas) == 15
    assert sum(persona["weight"] for persona in personas) == pytest.approx(1)
    assert {persona["segment"] for persona in personas} == set(segments["segment_id"])
    archetype_counts = Counter(persona["archetype"] for persona in personas)
    assert len(archetype_counts) == 5
    assert min(archetype_counts.values()) >= 20
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
    base = dict(zip(promo_rank, [-0.8, -1.3, -1.8, -2.3, -2.8], strict=True))
    assert set(segments["price_elasticity_source"]) == {"default"}
    assert set(segments["price_elasticity_observed_source"]) == {"data"}
    notes = loaders.load_market()["source_notes"]
    assert notes["pooled_price_elasticity"]["value"] == pytest.approx(-1.8)
    assert notes["pooled_price_elasticity"]["source"] == "default"
    assert notes["pooled_price_elasticity_observed"]["source"] == "data"
    personas = loaders.load_personas()
    for persona in personas:
        segment = segments.loc[persona["archetype"]]
        assert persona["cluster_label"] == segment["cluster_label"]
        band = persona["income_band"]
        if band in {"Under 15K", "15-24K", "25-34K"}:
            factor = 1.2
        elif band in {"35-49K", "50-74K", "Unknown"}:
            factor = 1.0
        else:
            factor = 0.85
        assert persona["price_elasticity"] == pytest.approx(base[persona["archetype"]] * factor)
        assert persona["price_elasticity_observed"] == pytest.approx(
            segment["price_elasticity_observed"], abs=1e-6
        )
        assert persona["source"]["price_elasticity"] == "default"
        assert persona["source"]["price_elasticity_observed"] == "data"
    for name, segment_prior in prior.items():
        group = [p for p in personas if p["segment"] == name]
        weighted = sum(p["price_elasticity"] * p["weight"] for p in group)
        assert segment_prior == pytest.approx(weighted / sum(p["weight"] for p in group), abs=1e-6)


def test_economics_anchors_and_backtest():
    economics = loaders.load_unit_econ()
    assert economics["currency"] == "THB"
    assert economics["price_thb"] > 0
    assert economics["source"]["price_thb"] == "data"
    assert economics["source"]["usd_thb"] == "default"

    anchors = loaders.load_anchors()
    assert set(anchors) == {"sets", "test_responses"}
    assert Counter(group["lang"] for group in anchors["sets"]) == {"th": 2, "en": 1}
    texts = [anchor for group in anchors["sets"] for anchor in group["anchors"]]
    assert all(len(group["anchors"]) == 5 for group in anchors["sets"])
    assert len(set(texts)) == 15
    responses = anchors["test_responses"]
    assert len({row["text"] for row in responses}) == len(responses) == 10
    assert Counter(row["expected"] for row in responses) == dict.fromkeys(range(1, 6), 2)
    assert set(texts).isdisjoint(row["text"] for row in responses)

    backtest = loaders.load_backtest()
    assert {"lift_pct", "redemption_rate", "incremental_sales"} <= set(backtest["actual"])
    assert backtest["source"] == "data" and backtest["campaign_id"] == "18"
    assert max(backtest["calibration_weeks"]) < backtest["start_week"]


def test_demo_brief_validates_and_retrieves_all_reviewed_scenarios():
    demo = loaders.load_demo_brief()
    spec = CampaignSpec.model_validate(demo["user_plan_spec"])
    assert spec.spec_version == "0.1" and spec.status == "ready"
    assert spec.brand.category == "toilet_tissue" and spec.brand.price_tier == "mid"
    assert spec.objective.primary == "incremental_profit"
    assert spec.budget.total_thb == 500_000
    assert spec.channels == ["shopee", "line"]
    assert spec.timing.event == "11.11"
    assert (spec.timing.start_date, spec.timing.end_date) == (date(2026, 11, 7), date(2026, 12, 4))
    mechanic, = spec.offer.mechanics
    assert (mechanic.type, mechanic.percentage, mechanic.eligibility) == ("percent_off", 40, "all")
    assert spec.audience.include_hints == []
    for path in (
        "objective.primary", "offer.mechanics.0.value", "budget.total_thb", "channels",
        "audience.description", "timing.event", "timing.start_date", "timing.end_date",
    ):
        meta = spec.field_meta[path]
        assert meta.source == "extracted" and meta.evidence in demo["brief_text"]

    plan = demo["approved_plan"]
    assert set(plan["segments_allowed"]) == set(loaders.load_segments()["archetype"])
    assert plan["weeks"] == {"campaign": 4, "post": 2}
    assert plan["seeds"] == 10 and plan["scenarios"] == ["S1", "S2", "S3"]
    rows = get_rival_scenarios(spec)
    assert len(rows) == 9
    assert {(row.rival, row.scenario) for row in rows} == {
        (rival, scenario) for rival in ("764", "595", "69") for scenario in plan["scenarios"]
    }


def test_rival_data_loaders():
    library = loaders.load_library()
    assert {"campaign_id", "reaction_to_campaign_id", "channel_mix", "source_url"} <= set(library)
    assert set(library["campaign_id"]) == {f"th{i:02d}" for i in range(1, 10)}
    brand_ids = {"Cellox": "751", "Scott": "764", "Zilk": "595"}
    assert set(library["brand_name"]) == set(brand_ids)
    assert library["brand"].eq(library["brand_name"].map(brand_ids)).all()
    assert library["category"].eq("toilet_tissue").all()
    assert library["discount_depth"].between(0, 1).all()
    assert library["reaction_type"].eq("").all() and library["source_url"].str.strip().ne("").all()
    brands = loaders.load_brands()
    assert set(brands["brand"]) == {"751", "764", "595", "69", "1011", "1315"}
    assert brands.loc[brands["is_ours"], "brand"].tolist() == ["751"]
    assert brands["market_share"].sum() == pytest.approx(1, abs=0.001)
    assert set(brands["price_tier"]) == {"low", "mid", "high"}
    prior = loaders.load_rival_prior()
    assert set(prior["p"]) == {"none", "match", "undercut", "non_price"}
    assert sum(prior["p"].values()) == pytest.approx(1)
    assert prior["p"]["none"] >= 0.2
    assert prior["source"] in {"data", "data+default"}
    if prior["n_events"] < 20:
        assert prior["source"] == "data+default"
    assert 0 <= prior["depth_p50"] <= prior["depth_p90"] <= 100
    assert prior["delay_weeks_p50"] >= 1


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
    pd.DataFrame([{**record, "week": 45}]).to_parquet(path, index=False)
    with pytest.raises(ValueError, match="build_tissue"):
        loaders.load_household_behavior()
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


def test_twin_response_loader_supports_wave_and_optional_columns(monkeypatch, tmp_path):
    monkeypatch.setattr(loaders, "RAW", tmp_path)
    assert loaders.load_twin_responses(columns=["QID32"]) is None
    for wave in ("1_3", "4"):
        path = loaders.source_asset_path(f"twin2k/wave{wave}_response.csv")
        path.parent.mkdir(exist_ok=True, parents=True)
        path.write_text(f"QID32,wave\n3,{wave}\n")
        assert list(loaders.load_twin_responses(wave=wave)) == ["QID32", "wave"]
    assert loaders.load_twin_responses(["QID32"]).to_dict("list") == {"QID32": [3]}
