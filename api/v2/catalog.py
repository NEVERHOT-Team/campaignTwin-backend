"""Resolve the explicitly selected synthetic rehearsal and its approved assumptions."""

import hashlib
import json
from datetime import date, timedelta
from decimal import Decimal

from api.v2.seller_catalog import catalog_entries, seller_context
from data.etl.config import CATEGORY, MIN_SEGMENT_HOUSEHOLDS
from data.loaders import load_market
from engine.seller_ledger import Settlement, subsidy_per_pack, valid_discount_amount
from engine.seller_simulation import POLICY

PRODUCT = {
    "choice_id": "demo-tissue-12-shopee",
    "product_id": "synthetic-tissue",
    "pack_id": "synthetic-12-roll",
    "label": "Synthetic demo toilet tissue",
    "pack_label": "12 rolls",
    "channel": "shopee",
    "timezone": "Asia/Bangkok",
    "offer_kinds": ["percent_off", "voucher"],
}
# Each term has an explicit basis in the review before approval.
COST_FIELDS = {
    "regular_price_thb": ("regular_price", 100, "THB_per_pack", "Regular price per pack"),
    "cogs_thb": ("cogs", 60, "THB_per_pack", "Cost of goods per pack"),
    "commission_pct": ("channel_commission", 0, "pct", "Percent of receipts after voucher"),
    "order_fee_thb": ("payment_fee", 0, "THB_per_order", "Fixed fee on every kept order"),
    "fulfillment_thb": ("seller_fulfillment", 0, "THB_per_order", "Fulfillment per kept order"),
    "voucher_fee_thb": ("voucher_service_fee", 0, "THB_per_order", "Fee per redeemed order"),
    "paid_media_thb": ("paid_media", 0, "THB_per_campaign", "Fixed campaign media outlay"),
}


def fingerprint(value):
    payload = json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def released_segments():
    """Expose only aggregate rows the current evaluator can consume."""
    segments = []
    seen = set()
    try:
        for row in load_market()["segments"]:
            count = row["households"]
            if type(count) is not int or count < 0:
                return []
            if count < MIN_SEGMENT_HOUSEHOLDS:
                continue
            name = row["name"]
            share = row["share"]
            rate = row["weekly_buy_rate"]
            if (
                not isinstance(name, str)
                or not name.strip()
                or name in seen
                or type(share) not in (int, float)
                or not 0 <= share <= 1
                or type(rate) not in (int, float)
                or not 0 <= rate <= 1
            ):
                return []
            seen.add(name)
            segments.append(
                {
                    "segment_id": name,
                    "name": name,
                    "households": count,
                    "share": float(share),
                    "weekly_purchase_probability": float(rate),
                    "source": "data",
                }
            )
    except (OSError, ValueError, KeyError, TypeError):
        return []
    if sum(segment["households"] for segment in segments) > 100_000:
        return []
    return segments


def resolve(facts):
    blockers = []

    def block(code, field, message):
        blockers.append({"code": code, "field": field, "message": message})

    try:
        snapshot, entries = catalog_entries()
    except (OSError, ValueError, KeyError, TypeError):
        snapshot, entries = {}, {}
    # Built-in demo IDs and operator listing IDs share the selection namespace.
    entries = {key: entry for key, entry in entries.items() if key != PRODUCT["choice_id"]}
    selected = facts.get("product_choice_id")
    demo = selected == PRODUCT["choice_id"]
    product, capability = entries.get(selected, (PRODUCT if demo else None, None))
    choices = [PRODUCT, *(item[0] for item in entries.values())]
    category = CATEGORY if demo else capability["product"]["category"] if capability else None
    if facts.get("product_scope") == "multiple_products":
        block(
            "unsupported_choice",
            "/facts/product_scope",
            "One focal product per run. Select single_product to rehearse only the chosen pack.",
        )
    if not product:
        block(
            "exact_product_required",
            "/facts/product_choice_id",
            "Select one of the supported product and pack configurations.",
        )
    if "channel" in facts and facts["channel"] != "shopee":
        block("unsupported_choice", "/facts/channel", "This evaluator supports Shopee.")
    if product:
        for field, supported in {
            "objective": {"incremental_profit"},
            "audience_hint": {"all", "all shoppers", "all buyers", "all eligible buyers"},
            "pack_hint": (
                {"12 rolls", "12-roll", "12-roll pack"} if demo else {product["pack_label"].lower()}
            ),
            "product_hint": (
                {product["label"].lower(), "toilet tissue"} if demo else {product["label"].lower()}
            ),
        }.items():
            if field in facts and facts[field].strip().lower() not in supported:
                choices_text = ", ".join(sorted(supported))
                block(
                    "unsupported_choice", "/facts/" + field, f"Supported values are {choices_text}."
                )

    bundle, costs, segments = None, {}, released_segments() if demo else []
    if capability and "start_date" in facts and "end_date" in facts:
        try:
            bundle, costs, segments = seller_context(facts, snapshot, capability)
        except ValueError as exc:
            block("missing_cost_basis", "/costs", str(exc))
        except (TypeError, KeyError, OverflowError):
            block("missing_cost_basis", "/costs", "Catalog evidence is incomplete or incompatible.")
    if product and not segments:
        block("evaluation_unsupported", "/segments", "Matching aggregate population unavailable.")
    if demo or bundle:
        for field, (key, default, unit, basis) in COST_FIELDS.items():
            if field in facts or key not in costs:
                provided = field in facts
                costs[key] = {
                    "value": facts.get(field, default),
                    "unit": unit,
                    "basis": basis,
                    "source": "data" if provided else "default",
                    "status": "confirmed",
                    "reason": None,
                    "evidence_ref": "/commercial_inputs/" + field if provided else POLICY,
                }
            if costs[key]["status"] == "unknown":
                block(
                    "missing_cost_basis",
                    "/facts/" + field,
                    f"Provide {field} for this listing. {costs[key]['reason']}",
                )
    price = Decimal(str(costs["regular_price"]["value"] or 0)) if costs else Decimal(0)
    offer = facts.get("offer", {})
    resolved_offer = None
    if product and offer:
        kind = offer.get("kind")
        amount = Decimal(str(offer.get("value", 0)))
        threshold = offer.get("min_spend_thb")
        terms = capability["voucher"] if capability else None
        supported = (
            (kind, offer.get("unit")) in {("voucher", "THB"), ("percent_off", "pct")}
            and kind in product["offer_kinds"]
            and offer.get("eligibility") == "all"
            and "cap_per_user" not in offer
            and (threshold is None if demo else threshold == terms["minimum_spend"])
            and (not terms or amount == Decimal(str(terms["discount"])))
            and (threshold is None or threshold <= price)
        )
        if supported and valid_discount_amount(offer, price):
            resolved_offer = {
                "offer_id": f"{product['choice_id']}-{kind}-{amount}",
                "discount": offer,
                "funder": "seller",
                "stackable": False,
                "min_spend_basis": terms["minimum_spend_basis"] if terms else None,
                "max_redemptions_per_order": 1,
                "eligible_order_scope": "One pack per kept order",
            }
        else:
            detail = (
                f"Catalog voucher {terms['discount']} THB, minimum {terms['minimum_spend']} "
                "THB on one pack."
                if terms
                else "Choose a voucher or percentage up to 30% "
                "of pack price for all buyers, without a threshold or shopper cap."
            )
            block("offer_terms_unverified", "/facts/offer", detail)
    default_policy = "fixed_offer" if capability else "half_discount"
    alternative_policy = facts.get("alternative_policy", default_policy)
    if capability and alternative_policy != "fixed_offer":
        block(
            "unsupported_choice",
            "/facts/alternative_policy",
            "This catalog voucher has a fixed value. Choose fixed_offer.",
        )
    floor = facts.get("min_unit_contribution_thb")
    if (
        resolved_offer
        and alternative_policy == "fixed_offer"
        and floor is not None
        and all(c["status"] == "confirmed" for c in costs.values())
    ):
        settlement = Settlement.from_plan({"costs": costs, "role": "original"})
        subsidy = subsidy_per_pack(offer, price)
        contribution = settlement.ledger(1, 1, subsidy, Decimal(0))["contribution_profit"]
        if contribution < Decimal(str(floor)):
            block(
                "unsupported_choice",
                "/facts/min_unit_contribution_thb",
                f"The committed offer contributes {contribution} THB per pack before media.",
            )
    media = costs.get("paid_media", {}).get("value") or 0
    if media > facts.get("budget_thb", media):
        block("unsupported_choice", "/facts/budget_thb", "The budget must cover committed media.")
    evaluation = None
    if product and "start_date" in facts and "end_date" in facts:
        start, end = (date.fromisoformat(facts[field]) for field in ("start_date", "end_date"))
        post = facts.get("post_days", 14)
        if 1 <= (end - start).days <= 42 and end <= date.max - timedelta(days=post):
            behavior = {
                "discount_response": facts.get("discount_response", 1),
                "stockup_days": facts.get("stockup_days", 0),
                "rival_demand_reduction_pct": facts.get("rival_demand_reduction_pct", 0),
            }
            evaluation = {
                "comparator": "paired_no_campaign",
                "scenarios": ["S1", "S2"] if behavior["rival_demand_reduction_pct"] else ["S1"],
                "measurement_window": {
                    "start_date": start.isoformat(),
                    "end_date": (end + timedelta(days=post)).isoformat(),
                },
                "max_loss_probability": facts.get("max_loss_probability", 0.25),
                "goal_target": {
                    "metric": "incremental_profit",
                    "minimum": facts.get("profit_target_thb", 0),
                    "unit": "THB",
                },
                "min_unit_contribution_thb": facts.get("min_unit_contribution_thb"),
                "behavior": behavior,
                "contingency_rules": (
                    [
                        {
                            "rule": "stop_vouchers_at_utilization",
                            "threshold_pct": facts["stop_vouchers_at_utilization_pct"],
                            "tick": "campaign_week_start",
                            "max_firings": 1,
                        }
                    ]
                    if "stop_vouchers_at_utilization_pct" in facts
                    else []
                ),
                "alternative_policy": alternative_policy,
                "assumptions": [
                    "One kept pack per order, exclusive of tax. Seller funds the voucher. "
                    "Percentage discounts round half up to a satang per pack. "
                    "Budget covers vouchers and fixed media. Media is charged at campaign start "
                    "and has no separately estimated exposure effect.",
                    {
                        "half_discount": "Compare a half-discount alternative.",
                        "fixed_offer": "Compare the committed discount against no campaign.",
                        "agent_bounded": "Request a smaller positive discount from the strategist. "
                        "If unavailable or invalid, compare a half discount.",
                    }[alternative_policy],
                    "All plans share eight paired seeds and all other commercial terms. "
                    "S1 assumes no rival action.",
                    f"Daily probability multiplier is 1 + {behavior['discount_response']} × "
                    "discount fraction, capped at certainty. Message appeal is neutral.",
                    f"One induced purchase can displace one ordinary purchase within "
                    f"{behavior['stockup_days']} subsequent days. Zero disables carryover.",
                    "Rank eligible plans on their lowest scenario P10. Every scenario must meet "
                    "the median profit target and risk ceiling with defined leakage in every seed.",
                    "Source segment counts define the simulated market size. "
                    "Outcomes describe this cohort over the campaign and post period.",
                ],
            }
            if evaluation["contingency_rules"]:
                evaluation["assumptions"].append(
                    f"At the start of each subsequent campaign week stop further vouchers if "
                    f"subsidy spend has reached {facts['stop_vouchers_at_utilization_pct']}% "
                    "of the subsidy budget."
                )
            if "S2" in evaluation["scenarios"]:
                evaluation["assumptions"].append(
                    f"S2 reduces active-period purchase probability by "
                    f"{behavior['rival_demand_reduction_pct']}% in all paired streams. "
                    "This planner-specified stress has unknown likelihood."
                )
        else:
            block(
                "evaluation_unsupported",
                "/facts/end_date",
                "Select 1–42 campaign days with room for the approved post period.",
            )
    return {
        "category": category,
        "supported_product_choices": choices,
        "resolved_product": product,
        "resolved_offer": resolved_offer,
        "data_versions": {
            "catalog": fingerprint(bundle) if bundle else POLICY if demo else None,
            "market": fingerprint(costs) if costs else None,
            "segments": fingerprint(segments) if segments else None,
            "evaluation_policy": POLICY if product else None,
        },
        "costs": costs,
        "segments": segments,
        "target_segment_ids": [] if product else None,
        "evaluation": evaluation,
        "blockers": blockers,
        **({"catalog_evidence": bundle} if bundle else {}),
    }
