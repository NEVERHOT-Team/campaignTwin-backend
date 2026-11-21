import json
from dataclasses import asdict

import pytest

from data import loaders
from data.etl.calibrate_shopper import AGENTS, baseline_volume
from engine import shopper
from engine.shopper import ShopperParams


def test_shopper_parameter_release_matches_engine_fields_and_preserves_response_defaults():
    defaults = asdict(ShopperParams())
    defaults.pop("label")
    records = loaders.load_shopper_params()
    assert records.keys() == defaults.keys()
    for name, record in records.items():
        if record["label"] == "default":
            assert record == {
                "value": defaults[name], "label": "default", "target": None, "achieved": None
            }
        else:
            assert isinstance(record["target"], str)
            assert isinstance(record["achieved"], float)
    for name in ("beta_price", "stockpile", "wait_per_exposure", "households_per_agent"):
        assert records[name]["label"] == "default"


def test_missing_shopper_parameters_use_defaults(tmp_path, monkeypatch):
    monkeypatch.setattr(loaders, "PROCESSED", tmp_path)
    assert loaders.load_shopper_params() == {}


def test_released_parameters_reproduce_own_brand_weekly_volume():
    records = loaders.load_shopper_params()
    params = ShopperParams(**{name: row["value"] for name, row in records.items()})
    population = shopper.build_population(AGENTS, seed=0, params=params)
    volumes = baseline_volume(population, params, panel_weeks=44, demo_weeks=6)
    target = 793 / (1417 * 44)  # Manufacturer 751 units per prelaunch buyer household-week.
    for observed in volumes.values():
        assert observed == pytest.approx(target, rel=0.10)
    for name in ("asc_rival", "asc_other"):
        assert records[name]["achieved"] == pytest.approx(volumes["panel"])


@pytest.mark.parametrize(
    "change",
    [{"value": "2"}, {"label": "estimate"}, {"label": "calibrated", "target": None},
     {"achieved": True}, {"target": {"weeks": 44}}],
)
def test_present_invalid_shopper_parameters_raise(tmp_path, monkeypatch, change):
    record = loaders.load_shopper_params()
    record["beta_price"].update(change)
    (tmp_path / "shopper_params.json").write_text(json.dumps(record))
    monkeypatch.setattr(loaders, "PROCESSED", tmp_path)
    with pytest.raises(ValueError, match="beta_price"):
        loaders.load_shopper_params()
