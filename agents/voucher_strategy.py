"""Propose the one permitted voucher lever using aggregate approved financial terms."""

import json
from decimal import Decimal

from pydantic import BaseModel, Field

from agents import llm
from engine.seller_ledger import subsidy_per_pack, valid_discount_amount


class DiscountProposal(BaseModel):
    value: float = Field(gt=0, allow_inf_nan=False)


def propose_discount(review):
    offer = review["resolved_offer"]["discount"]
    messages = [
        {
            "role": "system",
            "content": "Propose one smaller positive discount for comparison. "
            "Preserve the input discount unit, product, eligibility, dates, funding and budget. "
            "Use whole satang for a THB voucher. Return only the proposed discount value. "
            "Consider seller margins and preserve the contribution floor where feasible.",
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "unit": offer["unit"],
                    "original_discount": offer["value"],
                    "budget_thb": review["facts"]["budget_thb"],
                    "costs": {
                        key: {"value": row["value"], "unit": row["unit"]}
                        for key, row in review["costs"].items()
                    },
                    "floor_thb_per_pack": review["evaluation"]["min_unit_contribution_thb"],
                    "goal_thb": review["evaluation"]["goal_target"]["minimum"],
                }
            ),
        },
    ]
    proposal = llm.call_llm("strong", messages, DiscountProposal)
    if proposal is None:
        return {"source": "default", "reason": "strategy_unavailable", "value": None}
    value = Decimal(str(proposal.value))
    price = Decimal(str(review["costs"]["regular_price"]["value"]))
    candidate = {**offer, "value": proposal.value}
    if (
        not valid_discount_amount(candidate, price)
        or not value < Decimal(str(offer["value"]))
        or subsidy_per_pack(candidate, price) >= subsidy_per_pack(offer, price)
    ):
        return {"source": "default", "reason": "candidate_outside_bounds", "value": None}
    return {
        "source": "llm_estimate",
        "model": llm.resolve_model("strong"),
        "reason": "bounded_candidate",
        "value": float(value),
    }
