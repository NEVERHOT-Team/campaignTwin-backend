"""Planning arithmetic on the current review using the live evaluator's settlement basis."""

from decimal import Decimal
from fractions import Fraction
from math import ceil

from api.v2.contract import V2Error
from engine.seller_ledger import Settlement, subsidy_per_pack


def unit_economics(review):
    costs = review["costs"]
    offer = review["resolved_offer"]
    budget = review["facts"].get("budget_thb")
    if (
        not offer
        or budget is None
        or not costs
        or any(c["status"] != "confirmed" or c["value"] is None for c in costs.values())
    ):
        return None
    ledger = Settlement.from_plan({"costs": costs, "role": "original"})
    discount = subsidy_per_pack(offer["discount"], ledger.price)
    remaining = Decimal(str(budget)) - ledger.media
    if discount <= 0 or remaining < 0:
        return None
    zero = Decimal(0)
    return {
        "ordinary_contribution_thb": ledger.ledger(1, 0, zero, zero)["contribution_profit"],
        "redeemed_contribution_thb": ledger.ledger(1, 1, discount, zero)["contribution_profit"],
        "subsidy_per_order_thb": discount,
        "committed_media_thb": ledger.media,
        "voucher_budget_thb": remaining,
        "max_funded_redemptions": Fraction(remaining) // Fraction(discount),
    }


def what_if(review, redemptions, additional):
    terms = unit_economics(review)
    if terms is None or review["blockers"]:
        raise V2Error(409, "what_if_unavailable", "Resolve the draft before trying order volumes.")
    if additional > redemptions or redemptions > terms["max_funded_redemptions"]:
        raise V2Error(
            422,
            "infeasible_volume",
            "Additional orders must fit within redeemed "
            "orders, and all redemptions must fit within the voucher budget.",
        )
    ordinary = terms["ordinary_contribution_thb"]
    redeemed = terms["redeemed_contribution_thb"]
    media = terms["committed_media_thb"]
    target = Decimal(str(review["evaluation"]["goal_target"]["minimum"]))
    # Both ledgers omit the identical, unaffected ordinary orders.
    plan = redeemed * redemptions - media
    baseline = ordinary * (redemptions - additional)
    hurdle = target + media + redemptions * (ordinary - redeemed)
    minimum = 0 if hurdle <= 0 else None
    if minimum is None and ordinary > 0:
        minimum = max(0, ceil(Fraction(hurdle) / Fraction(ordinary)))
    feasible = minimum is not None and minimum <= redemptions
    return {
        "review_id": review["review_id"],
        "revision": review["revision"],
        "basis": "All additional orders redeem one voucher. Other orders and later purchases "
        "are unchanged. Fixed media is charged once. Volumes are planner assumptions.",
        "redeemed_orders": redemptions,
        "additional_orders": additional,
        "ordinary_buyers_redeeming_orders": redemptions - additional,
        "campaign_spend_thb": redemptions * terms["subsidy_per_order_thb"] + media,
        "incremental_profit_thb": plan - baseline,
        "profit_target_thb": target,
        "meets_profit_target": plan - baseline >= target,
        "minimum_additional_orders_for_target": minimum if feasible else None,
        "target_reachable_at_this_volume": feasible,
    }
