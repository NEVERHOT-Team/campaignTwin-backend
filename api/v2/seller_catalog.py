"""Project supported evidence bundles onto the published single-pack review contract."""

from datetime import date, timedelta
from math import isclose, isfinite

from api.v2.contract import validator
from data.campaign_evidence import resolve_campaign_evidence
from data.loaders import load_campaign_evidence_snapshot


def catalog_entries():
    snapshot = load_campaign_evidence_snapshot()
    choices = {}
    for capability in snapshot["capabilities"]:
        product = capability["product"]
        voucher = capability["voucher"]
        if (
            capability["currency"] != "THB"
            or capability["market_id"] != "TH"
            or capability["channel_id"] != "shopee"
            or type(product["pack_quantity"]) is not int
            or product["pack_quantity"] <= 0
            or type(product["units_per_order"]) is not int
            or product["units_per_order"] != 1
            or set(voucher) - {
                "mechanic_id", "discount", "discount_unit", "minimum_spend", "minimum_spend_unit",
                "minimum_spend_basis", "eligibility", "max_redemptions_per_order", "stacking",
                "funder",
            }
            or type(voucher["discount"]) not in (int, float)
            or not 0 < voucher["discount"] <= 1e9
            or voucher["mechanic_id"] != "fixed_order_voucher"
            or voucher["discount_unit"] != "thb_per_redeemed_order"
            or voucher["minimum_spend_unit"] != "thb_per_order"
            or voucher["eligibility"] != "all_focal_product_orders_meeting_minimum_spend"
            or voucher["minimum_spend_basis"] != "eligible_regular_merchandise_before_discount"
            or type(voucher["max_redemptions_per_order"]) is not int
            or voucher["max_redemptions_per_order"] != 1
            or voucher["stacking"] is not False
            or voucher["funder"] != "seller"
        ):
            continue
        choice_id = capability["capability_id"]
        if choice_id in choices:
            raise ValueError("Ambiguous catalog choice")
        choices[choice_id] = (
            {
                "choice_id": choice_id,
                "product_id": product["product_id"],
                "pack_id": product["selling_unit_id"],
                "label": product["name"],
                "pack_label": f"{product['pack_quantity']} {product['pack_unit']}",
                "channel": capability["channel_id"],
                "timezone": capability["timezone"],
                "offer_kinds": ["voucher"],
            },
            capability,
        )
    for choice, capability in choices.values():
        if (
            not validator("ProductChoice").is_valid(choice)
            or capability["timezone"] != "Asia/Bangkok"
            or type(capability["voucher"]["minimum_spend"]) not in (int, float)
            or not 0 <= capability["voucher"]["minimum_spend"] <= 1e9
        ):
            raise ValueError("Unusable catalog product or order threshold.")
    return snapshot, choices


# The evaluator settles one kept order per pack and commissions on net receipts.
COST_MAP = {
    "regular_price": ("regular_price", "thb_per_selling_unit", "THB_per_pack", "buyer"),
    "cogs": ("cogs", "thb_per_selling_unit", "THB_per_pack", "seller"),
    "channel_commission": ("channel_commission", "pct", "pct", "seller"),
    "voucher_service_fee": (
        "voucher_service_fee",
        "thb_per_redeemed_order",
        "THB_per_order",
        "seller",
    ),
    "fulfillment": ("seller_fulfillment", "thb_per_order", "THB_per_order", "seller"),
    "paid_media": ("paid_media", "thb_per_campaign", "THB_per_campaign", "seller"),
}


def seller_context(facts, snapshot, capability):
    product = capability["product"]
    selection = {
        "product_id": product["product_id"],
        "selling_unit_id": product["selling_unit_id"],
        "mechanic_id": capability["voucher"]["mechanic_id"],
        **{k: capability[k] for k in ("currency", "channel_id", "market_id")},
        "start_date": facts.get("start_date", ""),
        "end_date_exclusive": facts.get("end_date", ""),
    }
    match = resolve_campaign_evidence(selection, snapshot)
    if match["bundle"] is None:
        raise ValueError("Catalog dates, costs or selection lack matching evidence.")
    bundle = match["bundle"]
    if bundle["capability"]["capability_id"] != capability["capability_id"]:
        raise ValueError("The selected catalog voucher is unavailable for these selling dates.")
    economics = bundle["economics"]
    known = set(COST_MAP) | {
        "capability_id",
        "economics_id",
        "effective_from",
        "effective_to_exclusive",
        "tax_and_returns",
        "platform_reimbursement",
    }
    if set(economics) - known:
        raise ValueError("The catalog contains a cost outside this settlement policy.")
    if (
        economics["platform_reimbursement"]["value"] != 0
        or economics["channel_commission"]["basis"] != "net_merchandise_after_voucher"
        or economics["tax_and_returns"]["tax_basis"] != "exclusive_of_tax"
        or economics["tax_and_returns"]["returns_basis"] != "completed_kept_orders_only"
    ):
        raise ValueError("Unsupported funding, commission, tax or returns basis.")
    sources = {s["evidence_ref"]: s for s in bundle["sources"]}
    start = date.fromisoformat(selection["start_date"])
    end = date.fromisoformat(selection["end_date_exclusive"])
    if any(date.fromisoformat(s["as_of"]) > start for s in sources.values()):
        raise ValueError("Source evidence became available after the campaign start.")
    measurement_end = end + timedelta(days=facts.get("post_days", 14))
    if date.fromisoformat(economics["effective_to_exclusive"]) < measurement_end:
        raise ValueError("Ordinary unit economics expire before the post period ends.")
    for key, term in (("capability", capability), *economics.items()):
        if not isinstance(term, dict):
            continue
        source = sources[term["evidence_ref"]]
        until = (
            end
            if key in {"capability", "voucher_service_fee", "paid_media", "platform_reimbursement"}
            else measurement_end
        )
        if date.fromisoformat(source["window_start"]) > start or (
            date.fromisoformat(source["window_end_exclusive"]) < until
        ):
            raise ValueError("Source evidence expires before the relevant measurement period.")
    costs = {}
    for key, (name, source_unit, unit, payer) in COST_MAP.items():
        term = economics[key]
        if term["unit"] != source_unit or term["payer"] != payer:
            raise ValueError("Incompatible unit or payer in catalog costs.")
        source = sources[term["evidence_ref"]]
        costs[name] = {
            "status": "confirmed" if term["status"] == "confirmed" else "unknown",
            "value": term["value"],
            "unit": unit,
            "basis": term["basis"],
            "source": source["label"],
            "evidence_ref": term["evidence_ref"],
            "reason": term.get("reason"),
        }
    if not validator("Costs").is_valid(costs) or any(
        not isfinite(c["value"]) or c["value"] > 1e12
        for c in costs.values()
        if c["value"] is not None
    ):
        raise ValueError("Unsupported catalog costs or sources.")
    segments = []
    total = sum(s["population_count"]["value"] for s in bundle["segments"])
    for seg in bundle["segments"]:
        count = seg["population_count"]["value"]
        rate = seg["metrics"]["weekly_purchase_probability"]
        first = date.fromisoformat(seg["window_start"])
        last = date.fromisoformat(seg["window_end_exclusive"])
        if not first < last <= start:
            raise ValueError("Segment observations must come from a completed pre-campaign period.")
        for ref in {seg["evidence_ref"], rate["evidence_ref"]}:
            source = sources[ref]
            if (date.fromisoformat(source["window_start"]) > first
                    or date.fromisoformat(source["window_end_exclusive"]) < last):
                raise ValueError("Source evidence does not cover the segment observation period.")
        numerator, denominator = rate["numerator"], rate["denominator"]
        if (
            type(numerator["value"]) is not int
            or type(denominator["value"]) is not int
            or numerator["unit"] != "buying_household_weeks"
            or denominator["unit"] != "eligible_household_weeks"
            or not 0 <= numerator["value"] <= denominator["value"]
            or denominator["value"] <= 0
            or not isclose(rate["value"], numerator["value"] / denominator["value"])
        ):
            raise ValueError("Invalid purchase probability denominator or basis.")
        if (
            type(count) is not int
            or count < 50
            or total > 100000
            or rate["unit"] != "fraction"
            or not 0 <= rate["value"] <= 1
        ):
            raise ValueError("Unsupported catalog segment population or purchase rate.")
        segments.append(
            {
                "segment_id": seg["segment_id"],
                "name": seg["segment_id"],
                "households": count,
                "share": count / total,
                "weekly_purchase_probability": rate["value"],
                "source": sources[rate["evidence_ref"]]["label"],
            }
        )
    if any(not validator("Segment").is_valid(s) for s in segments):
        raise ValueError("Unsupported segment evidence.")
    if not segments or len({s["segment_id"] for s in segments}) != len(segments):
        raise ValueError("No uniquely identified usable segment population.")
    return bundle, costs, segments
