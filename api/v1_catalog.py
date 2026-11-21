"""Resolve the explicitly selected synthetic rehearsal and its approved assumptions."""

import hashlib
import json
from datetime import date, timedelta
from decimal import Decimal

from data.etl.config import CATEGORY, MIN_SEGMENT_HOUSEHOLDS
from data.loaders import load_market
from engine.synthetic import POLICY

PRODUCT = {
    "choice_id": "demo-tissue-12-shopee", "product_id": "synthetic-tissue",
    "pack_id": "synthetic-12-roll", "label": "Synthetic demo toilet tissue",
    "pack_label": "12 rolls", "channel": "shopee", "timezone": "Asia/Bangkok",
    "offer_kinds": ["percent_off", "voucher"],
}
ASSUMPTIONS = [
    "Synthetic rehearsal for one 12-roll pack. Price 100 THB, cost 60 THB per pack.",
    "The seller funds percentage discounts or fixed THB vouchers up to 30% of pack price.",
    "One pack per order, automatic redemption while budget remains, no fees or paid media.",
    "Compare the submitted discount and a half-discount alternative with no campaign.",
    "Round the alternative voucher down to a whole satang.",
    "All eligible shoppers have neutral message appeal. S1 has no rival action.",
    "Daily purchase probability increases proportionally with the discount fraction.",
    "All eight seeds must include subsidized purchases for a decision.",
    "Proceed requires median incremental profit of at least zero and loss risk at most 25%.",
    "Use paired seeds 0 through 7, a two-week post period and a hard THB subsidy cap.",
]


def assumed_cost(value, unit):
    return {
        "status": "confirmed", "value": value, "unit": unit, "source": "default",
        "basis": "Synthetic rehearsal assumption for the selected pack",
        "evidence_ref": POLICY, "reason": None,
    }


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


def resolve(facts):
    blockers = []

    def block(code, field, message):
        blockers.append({"code": code, "field": field, "message": message})

    demo = facts.get("product_choice_id") == PRODUCT["choice_id"]
    if not demo:
        block("exact_product_required", "/facts/product_choice_id",
              "Seller catalog unavailable. To rehearse the synthetic example, select its demo ID.")
    if "channel" in facts and facts["channel"] != PRODUCT["channel"]:
        block("unsupported_choice", "/facts/channel",
              "This platform has no evaluator. The synthetic rehearsal supports Shopee only.")
    if demo:
        for field, supported in {
            "objective": {"incremental_profit"},
            "audience_hint": {"all", "all shoppers", "all buyers", "all eligible buyers"},
            "pack_hint": {"12 rolls", "12-roll", "12-roll pack"},
            "product_hint": {"toilet tissue", "synthetic demo toilet tissue"},
        }.items():
            if field in facts and facts[field].strip().lower() not in supported:
                block("unsupported_choice", "/facts/" + field,
                      f"The synthetic example supports {', '.join(sorted(supported))}.")

    segments = released_segments()
    if not segments:
        block("evaluation_unsupported", "/segments", "Released aggregate segments unavailable.")
    fingerprint = hashlib.sha256(json.dumps(segments, sort_keys=True).encode()).hexdigest()
    offer = facts.get("offer", {})
    resolved_offer = None
    if demo and offer:
        supported = (offer.get("kind"), offer.get("unit")) in {
            ("percent_off", "pct"), ("voucher", "THB"),
        }
        amount = Decimal(str(offer.get("value", 0)))
        voucher_precision = offer.get("kind") != "voucher" or (
            amount >= Decimal("0.02") and amount.normalize().as_tuple().exponent >= -2
        )
        if (supported and voucher_precision
                and 0.01 <= offer.get("value", 0) <= 30 and offer.get("eligibility") == "all"
                and not {"min_spend_thb", "cap_per_user"}.intersection(offer)):
            resolved_offer = {
                "offer_id": f"synthetic-{offer['kind']}-{offer['value']}", "discount": offer,
                "funder": "seller", "stackable": False, "min_spend_basis": None,
                "max_redemptions_per_order": 1, "eligible_order_scope": "One pack per purchase",
            }
        else:
            block("offer_terms_unverified", "/facts/offer",
                  "Select 0.01–30% off or a 0.02–30 THB whole-satang voucher for all, "
                  "without a threshold or cap.")

    evaluation = None
    if demo and "start_date" in facts and "end_date" in facts:
        start, end = (date.fromisoformat(facts[field]) for field in ("start_date", "end_date"))
        if 1 <= (end - start).days <= 42 and end <= date.max - timedelta(days=14):
            evaluation = {
                "comparator": "paired_no_campaign", "scenarios": ["S1"],
                "measurement_window": {"start_date": start.isoformat(),
                                       "end_date": (end + timedelta(days=14)).isoformat()},
                "max_loss_probability": 0.25,
                "goal_target": {"metric": "incremental_profit", "minimum": 0, "unit": "THB"},
                "assumptions": ASSUMPTIONS,
            }
        else:
            block("evaluation_unsupported", "/facts/end_date",
                  "Select 1 to 42 campaign days with room for 14 post-campaign days.")
    return {
        "category": CATEGORY if demo else None, "supported_product_choices": [PRODUCT],
        "resolved_product": PRODUCT if demo else None, "resolved_offer": resolved_offer,
        "data_versions": {"catalog": POLICY if demo else None, "market": POLICY if demo else None,
                          "segments": fingerprint if segments else None,
                          "evaluation_policy": POLICY if demo else None},
        "costs": {"regular_price": assumed_cost(100, "THB_per_pack"),
                  "cogs": assumed_cost(60, "THB_per_pack")} if demo else {},
        "segments": segments, "target_segment_ids": [] if demo else None,
        "evaluation": evaluation, "blockers": blockers,
    }
