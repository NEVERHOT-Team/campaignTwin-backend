"""Resolve the stage catalog, offer terms and market evidence for the Cellox rehearsal."""

import hashlib
import json
from datetime import date, timedelta

from data.etl.config import CATEGORY, MIN_SEGMENT_HOUSEHOLDS
from data.loaders import load_market

ARENA_POLICY = "arena-tissue-v1"
ARENA_PRODUCT = {
    "choice_id": "cellox-tissue-shopee", "product_id": "751",
    "pack_id": "cellox-reference-pack", "label": "Cellox toilet tissue",
    "pack_label": "Reference pack", "channel": "shopee", "timezone": "Asia/Bangkok",
    "offer_kinds": ["percent_off", "voucher"],
}
ARENA_MAX_DISCOUNT_PCT = 50
ARENA_SCENARIOS = ["S1", "S2", "S3", "S4"]
ARENA_ASSUMPTIONS = [
    "AI-vs-AI rehearsal for the Cellox reference pack on Shopee, priced from market.json.",
    "The strategist proposes alternatives within the approved discount and budget bounds.",
    "Rivals react per scenario: S1 passive, S2 most likely, S3 aggressive, S4 red team.",
    "Shopper agents come from household personas; message appeal uses SSR or the fallback.",
    "Plans are compared with a paired no-campaign baseline on the same seeds.",
    "Proceed requires median S1 incremental profit of at least zero and loss risk at most 25%.",
    "Includes a two-week post-campaign window to capture pull-forward.",
]


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
            if (not isinstance(name, str) or not name.strip() or name in seen
                    or type(share) not in (int, float) or not 0 <= share <= 1
                    or type(rate) not in (int, float) or not 0 <= rate <= 1):
                return []
            seen.add(name)
            segments.append({
                "segment_id": name, "name": name, "households": count,
                "share": float(share), "weekly_purchase_probability": float(rate), "source": "data",
            })
    except (OSError, ValueError, KeyError, TypeError):
        return []
    if sum(segment["households"] for segment in segments) > 100_000:
        return []
    return segments


def arena_costs(market=None):
    market = market if market is not None else load_market()
    fields = market.get("source_notes", {}).get("fields", {})

    def cost(key, unit, basis):
        return {
            "status": "confirmed", "value": float(market[key]), "unit": unit,
            "source": fields.get(key, "default"), "basis": basis,
            "evidence_ref": f"{ARENA_POLICY}:market.json", "reason": None,
        }

    return {
        "regular_price": cost("ref_price", "THB_per_pack", "Reference shelf price per pack"),
        "cogs": cost("unit_cost", "THB_per_pack", "Unit cost per pack"),
        "channel_commission": cost("platform_fee_pct", "pct", "Percent of discounted order value"),
        "payment_fee": cost("payment_fee_pct", "pct", "Percent of discounted order value"),
        "seller_fulfillment": cost("shipping_cost_per_order", "THB_per_order",
                                   "Shipping cost per order"),
        "margin_floor_pct": cost("margin_floor_pct", "pct", "Minimum allowed margin percentage"),
    }


def stage_product(facts):
    """Default to Cellox unless the planner has asked for a different product or pack."""
    choice = facts.get("product_choice_id")
    if choice:
        return ARENA_PRODUCT if choice == ARENA_PRODUCT["choice_id"] else None
    product = facts.get("product_hint", "").strip().casefold()
    pack = facts.get("pack_hint", "").strip().casefold()
    if product in {"", "cellox", "cellox toilet tissue", "toilet tissue", "toilet paper"} and (
        pack in {"", ARENA_PRODUCT["pack_label"].casefold(), ARENA_PRODUCT["pack_id"]}
    ):
        return ARENA_PRODUCT
    return None


def resolve(facts):
    blockers = []

    def block(code, field, message):
        blockers.append({"code": code, "field": field, "message": message})

    if stage_product(facts) is None:
        block("exact_product_required", "/facts/product_choice_id",
              "Select the listed Cellox reference pack on Shopee for this rehearsal.")
        return {
            "category": None, "supported_product_choices": [ARENA_PRODUCT],
            "resolved_product": None, "resolved_offer": None,
            "data_versions": dict.fromkeys(("catalog", "market", "segments", "evaluation_policy")),
            "costs": {}, "segments": [], "target_segment_ids": None,
            "evaluation": None, "blockers": blockers,
        }
    if "channel" in facts and facts["channel"] != ARENA_PRODUCT["channel"]:
        block("unsupported_choice", "/facts/channel",
              "The Cellox rehearsal is listed on Shopee only.")
    objective = facts.get("objective")
    if objective and objective not in {"incremental_profit", "revenue", "new_buyers"}:
        block("unsupported_choice", "/facts/objective",
              "The arena measures incremental profit, revenue and new buyers.")

    segments = released_segments()
    if not segments:
        block("evaluation_unsupported", "/segments", "Released aggregate segments unavailable.")
    fingerprint = hashlib.sha256(json.dumps(segments, sort_keys=True).encode()).hexdigest()

    resolved_offer = None
    offer = facts.get("offer", {})
    if offer:
        try:
            market = load_market()
            price = float(market["ref_price"])
        except (OSError, ValueError, KeyError, TypeError):
            price = None
        kind, unit, value = offer.get("kind"), offer.get("unit"), offer.get("value", 0)
        pct = value if unit == "pct" else (100 * value / price if price else None)
        supported = (kind, unit) in {("percent_off", "pct"), ("voucher", "pct"), ("voucher", "THB")}
        eligibility = offer.get("eligibility", "all")
        if (supported and pct is not None and 0 < pct <= ARENA_MAX_DISCOUNT_PCT
                and eligibility in {"all", "new_only", "members"}
                and "min_spend_thb" not in offer):
            resolved_offer = {
                "offer_id": f"arena-{kind}-{value}{unit}",
                "discount": {**offer, "eligibility": eligibility},
                "funder": "seller", "stackable": False, "min_spend_basis": None,
                "max_redemptions_per_order": 1, "eligible_order_scope": "One pack per purchase",
            }
        else:
            block("offer_terms_unverified", "/facts/offer",
                  f"Select up to {ARENA_MAX_DISCOUNT_PCT}% off or a THB voucher for all, new or "
                  "member buyers, without a minimum spend.")

    evaluation = None
    if "start_date" in facts and "end_date" in facts:
        start, end = (date.fromisoformat(facts[field]) for field in ("start_date", "end_date"))
        if 1 <= (end - start).days <= 56 and end <= date.max - timedelta(days=14):
            evaluation = {
                "comparator": "paired_no_campaign", "scenarios": list(ARENA_SCENARIOS),
                "measurement_window": {"start_date": start.isoformat(),
                                       "end_date": (end + timedelta(days=14)).isoformat()},
                "max_loss_probability": 0.25,
                "goal_target": {"metric": "incremental_profit", "minimum": 0, "unit": "THB"},
                "assumptions": ARENA_ASSUMPTIONS,
            }
        else:
            block("evaluation_unsupported", "/facts/end_date",
                  "Select 1 to 56 campaign days with room for 14 post-campaign days.")

    hint = str(facts.get("audience_hint", "")).strip().casefold()
    targets = [s["segment_id"] for s in segments if s["segment_id"].casefold() == hint]
    try:
        costs = arena_costs()
    except (OSError, ValueError, KeyError, TypeError):
        costs = {}
        block("evaluation_unsupported", "/costs", "Market economics unavailable.")
    return {
        "category": CATEGORY, "supported_product_choices": [ARENA_PRODUCT],
        "resolved_product": ARENA_PRODUCT, "resolved_offer": resolved_offer,
        "data_versions": {"catalog": ARENA_POLICY, "market": ARENA_POLICY,
                          "segments": fingerprint if segments else None,
                          "evaluation_policy": ARENA_POLICY},
        "costs": costs, "segments": segments, "target_segment_ids": targets,
        "evaluation": evaluation, "blockers": blockers,
    }
