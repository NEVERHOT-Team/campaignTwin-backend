"""Seller-facing v2 review, interview, approval, and what-if routes."""

from uuid import UUID, uuid4

from fastapi import APIRouter

import db
from api.v2.business import what_if
from api.v2.contract import V2Error
from api.v2.interview import conversation_turn, next_step, present
from api.v2.models import Change, Choose, Reply, Session, Start, VolumeCase, VolumeResult
from api.v2.reviews import (
    approve_review,
    build_review,
    create_review,
    current_review,
    field_name,
    get_review,
    revise_review,
)

PREFIX = "/api/v2"
REVIEWS = PREFIX + "/campaign-reviews"
router = APIRouter(tags=["Campaign reviews"])


@router.get("/", summary="Discover the seller interview and simulation handoff")
def discovery():
    return {
        "version": "2.0.0",
        "create_review": REVIEWS,
        "docs": PREFIX + "/docs",
        "schema": PREFIX + "/openapi.json",
        "simulation_schema": PREFIX + "/openapi.json",
        "opening_prompt": "What are you promoting, when, and what would make it worthwhile?",
        "example": {
            "brief": "I want to promote toilet tissue.",
            "fields": {"budget_thb": 1500},
            "selling_period": {"first_day": "2026-12-01", "last_day": "2026-12-07"},
        },
        "business_inputs": {
            "seller_decides": [
                "product_choice_id",
                "offer",
                "audience_hint",
                "budget_thb",
                "selling_period",
                "objective",
                "profit_target_thb",
                "min_unit_contribution_thb",
                "alternative_policy",
            ],
            "from_listing_when_available": ["channel", "pack", "timezone", "costs", "segments"],
            "model_assumptions_to_review": [
                "discount_response",
                "stockup_days",
                "post_days",
                "rival_demand_reduction_pct",
                "max_loss_probability",
            ],
            "calculated": ["end_date", "measurement_window", "unit_economics"],
        },
    }


def with_period(fields, period):
    result = dict(fields or {})
    if period:
        if {"start_date", "end_date"}.intersection(result):
            raise V2Error(422, "ambiguous_period", "Use one representation of the selling period.")
        result.update(period.edits())
    return result


@router.post(
    "/campaign-reviews",
    response_model=Session,
    status_code=201,
    summary="Start or describe a campaign",
)
def start(payload: Start):
    fields = with_period(payload.fields, payload.selling_period)
    if payload.brief is not None or fields or payload.fields is not None:
        body = {"brief": payload.brief} if payload.brief is not None else {}
        if fields or payload.fields is not None:
            body["fields"] = fields
        return present(create_review(body))
    identifier = str(uuid4())
    review = build_review(identifier, 1, {}, {}, [], {"status": "not_requested", "reason": None})
    with db.v2_transaction() as state:
        state["reviews"][identifier] = [review]
    return present(review)


@router.get(
    "/campaign-reviews/{review_id}",
    response_model=Session,
    summary="Resume a seller interview",
)
def resume(review_id: UUID):
    return present(get_review(review_id))


@router.post(
    "/campaign-reviews/{review_id}/turns",
    response_model=Session,
    status_code=201,
    summary="Answer, choose a suggestion, or correct business inputs",
)
def turn(review_id: UUID, payload: Reply | Choose | Change):
    with db.v2_transaction() as state:
        review = current_review(state, review_id, payload.base_revision)
    if isinstance(payload, Reply):
        return present(conversation_turn(review_id, review, payload.answer.strip()))
    body = {"base_revision": payload.base_revision}
    if isinstance(payload, Choose):
        step = next_step(review)
        choice = next((c for c in step["choices"] if c["id"] == payload.choice), None)
        if choice is None:
            raise V2Error(422, "unknown_choice", "Choose a listed option for the current question.")
        if step["mode"] == "confirm":
            body["confirm"] = [choice["id"]]
        else:
            body["edits"] = {field_name(step["field"]): choice["value"]}
    else:
        edits = with_period(payload.edits, payload.selling_period)
        if edits or payload.edits is not None:
            body["edits"] = edits
        if payload.confirm is not None:
            body["confirm"] = payload.confirm
    return present(revise_review(review_id, body))


@router.post(
    "/campaign-reviews/{review_id}/what-if",
    response_model=VolumeResult,
    summary="Check the profit hurdle for assumed voucher volumes",
)
def volume_case(review_id: UUID, payload: VolumeCase):
    with db.v2_transaction() as state:
        review = current_review(state, review_id, payload.base_revision)
    return what_if(review, payload.redeemed_orders, payload.additional_orders)


@router.post("/campaign-reviews/{review_id}/revisions", response_model=Session, status_code=201)
def revision(review_id: UUID, payload: dict):
    return present(revise_review(review_id, payload))


@router.post("/campaign-reviews/{review_id}/approvals", status_code=201)
def approve(review_id: UUID, payload: dict):
    return approve_review(review_id, payload)
