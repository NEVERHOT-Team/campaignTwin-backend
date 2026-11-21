"""Resolve a campaign against an authorized CT-CAMPAIGN-EVIDENCE v1 snapshot."""

from collections.abc import Iterator, Mapping
from copy import deepcopy
from datetime import date
from math import isfinite
from typing import Any

Record = Mapping[str, Any]

_CAPABILITY_FIELDS = (
    "capability_id", "market_id", "channel_id", "currency", "timezone", "valid_from",
    "valid_to_exclusive", "evidence_ref",
)
_PRODUCT_FIELDS = (
    "product_id", "name", "brand", "category", "selling_unit_id", "pack_quantity", "pack_unit",
    "units_per_order",
)
_VOUCHER_FIELDS = (
    "mechanic_id", "discount", "discount_unit", "minimum_spend", "minimum_spend_unit",
    "minimum_spend_basis", "eligibility", "max_redemptions_per_order", "stacking", "funder",
)
_ECONOMICS_FIELDS = ("economics_id", "capability_id", "effective_from", "effective_to_exclusive")
_SOURCE_FIELDS = (
    "evidence_ref", "source_id", "source_version", "record_key", "transformation_id",
    "transformation_version", "as_of", "window_start", "window_end_exclusive",
    "scope", "grain", "coverage", "origin", "evidence_class", "label",
)
_SEGMENT_FIELDS = (
    "segment_id",
    "profile_version",
    "capability_id",
    "population_basis",
    "population_count",
    "window_start",
    "window_end_exclusive",
    "evidence_ref",
)
_CONFIGURATION_FIELDS = (
    "product_id",
    "selling_unit_id",
    "market_id",
    "channel_id",
    "mechanic_id",
    "currency",
)
_MATERIAL_COSTS = (
    "regular_price",
    "cogs",
    "channel_commission",
    "voucher_service_fee",
    "fulfillment",
    "paid_media",
    "platform_reimbursement",
)


def resolve_campaign_evidence(selection: Record, source_snapshot: Record) -> dict[str, Any]:
    """Return the supported handoff or typed blockers for one selected campaign."""
    unsupported = _blocked("unsupported_configuration", "selection")
    economics_unresolved = _blocked("economics_unresolved", "economics")
    try:
        configuration = tuple(selection[field] for field in _CONFIGURATION_FIELDS)
        start = date.fromisoformat(selection["start_date"])
        end = date.fromisoformat(selection["end_date_exclusive"])
    except (KeyError, TypeError, ValueError):
        return unsupported

    if start >= end:
        return unsupported

    capabilities = [
        capability
        for capability in source_snapshot["capabilities"]
        if _configuration(capability) == configuration
        and _covers(capability, "valid_from", "valid_to_exclusive", start, end)
    ]
    if len(capabilities) != 1:
        return unsupported
    (capability,) = capabilities
    if set(capability) - {*_CAPABILITY_FIELDS, "product", "voucher"} or (
        set(capability["voucher"]) - set(_VOUCHER_FIELDS)
    ):
        return unsupported
    capability = {
        **_scalar_fields(capability, _CAPABILITY_FIELDS),
        "product": _scalar_fields(capability["product"], _PRODUCT_FIELDS),
        "voucher": _scalar_fields(capability["voucher"], _VOUCHER_FIELDS),
    }

    economics_matches = [
        economics
        for economics in source_snapshot["economics"]
        if economics["capability_id"] == capability["capability_id"]
        and _covers(economics, "effective_from", "effective_to_exclusive", start, end)
    ]
    if len(economics_matches) != 1:
        return economics_unresolved
    (economics,) = economics_matches

    blockers = _cost_blockers(economics)
    if blockers is None:
        return economics_unresolved

    if set(economics) - {*_ECONOMICS_FIELDS, *_MATERIAL_COSTS, "tax_and_returns"}:
        return economics_unresolved
    published_costs = _scalar_fields(economics, _ECONOMICS_FIELDS)
    for key in _MATERIAL_COSTS:
        term = economics[key]
        fields = ("status", "value", "unit", "basis", "payer", "evidence_ref", "reason")
        published_costs[key] = _scalar_fields(term, (name for name in fields if name in term))
    published_costs["tax_and_returns"] = _scalar_fields(
        economics["tax_and_returns"], ("tax_basis", "returns_basis", "evidence_ref")
    )
    segments = [
        _published_segment(segment)
        for segment in source_snapshot["segment_candidates"]
        if segment["capability_id"] == capability["capability_id"] and _can_release(segment)
    ]
    references = {
        reference
        for record in (capability, published_costs, *segments)
        for reference in _evidence_refs(record)
    }
    sources = [
        source for source in source_snapshot["sources"] if source["evidence_ref"] in references
    ]
    published_refs = {source["evidence_ref"] for source in sources}
    if published_refs != references or len(published_refs) != len(sources):
        raise ValueError("Evidence references must resolve uniquely in the source snapshot")

    bundle = {
        "reference": _scalar_fields(
            source_snapshot["reference"], ("contract_version", "snapshot_id", "snapshot_version")
        ),
        "selection": _scalar_fields(
            selection, (*_CONFIGURATION_FIELDS, "start_date", "end_date_exclusive")
        ),
        "capability": capability,
        "economics": published_costs,
        "segments": segments,
        "sources": [_scalar_fields(source, _SOURCE_FIELDS) for source in sources],
    }
    return {
        "status": "blocked" if blockers else "supported",
        "blockers": blockers,
        "bundle": deepcopy(bundle),
    }


def _blocked(code: str, field: str) -> dict[str, Any]:
    return {"status": "blocked", "blockers": [{"code": code, "field": field}], "bundle": None}


def _configuration(capability: Record) -> tuple[Any, ...]:
    return (
        capability["product"]["product_id"],
        capability["product"]["selling_unit_id"],
        capability["market_id"],
        capability["channel_id"],
        capability["voucher"]["mechanic_id"],
        capability["currency"],
    )


def _covers(record: Record, start_key: str, end_key: str, start: date, end: date) -> bool:
    return date.fromisoformat(record[start_key]) <= start and end <= date.fromisoformat(
        record[end_key]
    )


def _cost_blockers(economics: Record) -> list[dict[str, str]] | None:
    blockers = []
    for field in _MATERIAL_COSTS:
        cost = economics.get(field)
        if not isinstance(cost, Mapping):
            return None
        status = cost.get("status")
        value = cost.get("value")
        if not cost.get("evidence_ref"):
            return None
        if status == "unavailable" and "value" in cost and value is None and cost.get("reason"):
            blockers.append({"code": "material_cost_unavailable", "field": f"economics.{field}"})
            continue
        if status != "confirmed" or type(value) not in (int, float):
            return None
        if not all(cost.get(key) for key in ("unit", "basis", "payer")):
            return None
        if isinstance(value, float) and not isfinite(value):
            return None
        if value < 0 or (field == "regular_price" and value == 0):
            return None
    return blockers


def _can_release(segment: Record) -> bool:
    population = segment.get("population_count")
    if segment["population_basis"] == "synthetic":
        return (isinstance(population, Mapping) and population.get("unit") == "synthetic_households"
                and type(population.get("value")) is int and population["value"] >= 50)
    if segment["population_basis"] != "household_derived" or not isinstance(population, Mapping):
        return False
    count = population.get("value")
    return population.get("unit") == "distinct_households" and type(count) is int and count >= 50


def _scalar_fields(record, fields):
    published = {key: record[key] for key in fields}
    if any(value is not None and type(value) not in (str, int, float, bool)
           for value in published.values()):
        raise ValueError("Expected scalar aggregate metadata.")
    return published


def _published_segment(segment: Record) -> dict[str, Any]:
    published = _scalar_fields(segment, (k for k in _SEGMENT_FIELDS if k != "population_count"))
    published["population_count"] = _scalar_fields(segment["population_count"], ("value", "unit"))
    published["metrics"] = {}
    if "weekly_purchase_probability" in segment["metrics"]:
        rate = segment["metrics"]["weekly_purchase_probability"]
        public_rate = _scalar_fields(rate, ("value", "unit", "evidence_ref"))
        for key in ("numerator", "denominator"):
            public_rate[key] = _scalar_fields(rate[key], ("value", "unit"))
        published["metrics"]["weekly_purchase_probability"] = public_rate
    return published


def _evidence_refs(record: Record) -> Iterator[str]:
    for key, value in record.items():
        if key == "evidence_ref":
            yield value
        elif isinstance(value, Mapping):
            yield from _evidence_refs(value)
