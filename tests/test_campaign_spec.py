from copy import deepcopy

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from contracts.models import CampaignSpec, SimulationPlan
from main import app
from mocks import load_mock


@pytest.fixture
def spec():
    return deepcopy(load_mock("campaign_spec")["spec"])


def test_nested_spec_roundtrips_and_publishes_in_openapi(spec):
    parsed = CampaignSpec.model_validate(spec)
    assert parsed.budget.total_thb == 80000
    assert parsed.offer.mechanics[0].percentage == 40
    assert parsed.channels == ["shopee", "line"]
    assert parsed.field_meta["budget"].source == "default"
    assert CampaignSpec.model_validate_json(parsed.model_dump_json()) == parsed
    schema = app.openapi()["components"]["schemas"]["CampaignSpec"]
    assert {"brand", "objective", "offer", "budget", "field_meta"} <= schema["properties"].keys()
    assert "budget_thb" not in schema["properties"]


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        ("budget.total_thb", -1, "greater than or equal to 0"),
        ("budget.total_thb", float("inf"), "finite"),
        ("budget.max_discount_pct", 101, "less than or equal to 100"),
        ("budget.max_discount_pct", 20, "exceeds budget.max_discount_pct"),
        ("budget.split", {"discount": 80000, "media": 1}, "must not exceed"),
        ("timing.end_date", "2027-10-24", "end must follow start"),
        ("timing.end_date", "2028-01-01", "at most 8 weeks"),
        ("constraints.max_loss_prob", 20, "less than or equal to 1"),
        ("constraints.margin_floor_pct", 101, "less than or equal to 100"),
        ("channels", ["Shopee"], "Input should be"),
        ("objective.primary", "sales_magic", "Input should be"),
        ("strategist_bounds", {"discount_pct": [40, 10]}, "minimum must not exceed"),
        ("strategist_bounds", {"discount_pct": [10, 30]}, "within strategist_bounds"),
    ],
)
def test_invalid_campaign_values_are_rejected(spec, path, value, message):
    parent = spec
    *parts, key = path.split(".")
    for part in parts:
        parent = parent[part]
    parent[key] = value
    with pytest.raises(ValidationError, match=message):
        CampaignSpec.model_validate(spec)


def test_mixed_percent_and_currency_mechanics(spec):
    spec["budget"]["max_discount_pct"] = 25
    spec["offer"]["mechanics"] = [
        {"type": "percent_off", "value": 25},
        {"type": "voucher", "value": 200, "value_unit": "thb"},
        {"type": "free_shipping", "value": 50},
    ]
    parsed = CampaignSpec.model_validate(spec)
    assert [mechanic.percentage for mechanic in parsed.offer.mechanics] == [25, None, None]
    spec["offer"]["mechanics"][1].pop("value_unit")
    with pytest.raises(ValidationError, match="requires value_unit"):
        CampaignSpec.model_validate(spec)


def test_sources_require_coverage_and_extraction_evidence(spec):
    spec["field_meta"].pop("budget")
    with pytest.raises(ValidationError, match="Missing field_meta source for: budget.total_thb"):
        CampaignSpec.model_validate(spec)
    spec["field_meta"]["budget"] = {"source": "extracted", "confidence": 0.9}
    with pytest.raises(ValidationError, match="evidence quote"):
        CampaignSpec.model_validate(spec)
    spec["field_meta"]["budget"]["evidence"] = "budget 80,000 THB"
    assert CampaignSpec.model_validate(spec).field_meta["budget"].evidence
    spec["field_meta"]["typo"] = {"source": "user", "confidence": 1}
    with pytest.raises(ValidationError, match="path does not exist: typo"):
        CampaignSpec.model_validate(spec)


def test_missing_fields_stay_null_and_block_simulation(spec):
    spec["status"] = "draft"
    spec["budget"]["total_thb"] = None
    spec["timing"]["start_date"] = None
    spec["offer"]["mechanics"] = []
    spec["channels"] = []
    spec["open_questions"] = [
        {"field": "budget.total_thb", "question": "Budget?", "blocking": False}
    ]
    parsed = CampaignSpec.model_validate(spec)
    assert parsed.status == "needs_info"
    assert parsed.budget.total_thb is None
    assert parsed.timing.start_date is None
    blocking = {question.field for question in parsed.open_questions if question.blocking}
    assert blocking == {"budget.total_thb", "timing.start_date", "offer.mechanics", "channels"}
    with pytest.raises(ValidationError, match="before running"):
        SimulationPlan(spec=parsed, context=load_mock("context_pack"), strategies=[])
    spec["status"] = "ready"
    with pytest.raises(ValidationError, match="blocking questions remain"):
        CampaignSpec.model_validate(spec)


def test_defaults_have_provenance_and_do_not_invent_economics_or_bounds(spec):
    spec.pop("constraints")
    spec["field_meta"].pop("constraints")
    spec.pop("objective")
    spec["field_meta"].pop("objective")
    parsed = CampaignSpec.model_validate(spec)
    assert parsed.objective.primary == "incremental_profit"
    assert parsed.field_meta["objective"].source == "default"
    assert parsed.constraints.max_loss_prob == 0.2
    assert parsed.field_meta["constraints"].source == "default"
    assert parsed.unit_economics is None
    assert parsed.strategist_bounds is None
    assert CampaignSpec.model_validate_json(parsed.model_dump_json()) == parsed


def test_context_endpoint_rejects_invalid_spec_and_runs_reject_unready(spec):
    with TestClient(app) as client:
        spec["budget"]["split"] = {"discount": 80001}
        assert client.post("/api/plan/context", json=spec).status_code == 422
        spec["budget"]["split"] = None
        spec["status"] = "draft"
        plan = {"spec": spec, "context": load_mock("context_pack"), "strategies": []}
        response = client.post("/api/runs", json=plan)
        assert response.status_code == 422
        assert "before running" in response.text


def test_legacy_fields_are_not_silently_ignored(spec):
    spec["budget_thb"] = 1
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        CampaignSpec.model_validate(spec)
