"""Planner intake, revision history and explicit approval for the v1 workflow."""

import os
from datetime import date, datetime, timezone
from uuid import UUID, uuid4

from fastapi import APIRouter
from fastapi.responses import JSONResponse

import db
from agents.review import affirmative, extract, order_questions
from api.v1_catalog import CHOICES, PRODUCT, resolve
from api.v1_contract import V1Error, validate_request

router = APIRouter(prefix="/api/v1/campaign-reviews")


def build_review(review_id, revision, facts, origins, candidates, agent):
    # Recompute derived facts whenever the user changes their underlying choice.
    if "channel" in facts:
        channel = facts["channel"].strip().lower()
        facts["channel"] = "tiktok_shop" if channel in {"tiktok shop", "tiktok-shop"} else channel
    channel_key = "/facts/channel"
    if origins.get(channel_key) == "data":
        facts.pop("channel", None)
        origins.pop(channel_key)
    choice = CHOICES.get(facts.get("product_choice_id"))
    if (choice and "channel" not in facts
            and not any(c["field"] == channel_key for c in candidates)):
        facts["channel"] = choice["channel"]
        origins[channel_key] = "data"
    offer = facts.get("offer")
    eligibility_key = "/facts/offer/eligibility"
    if offer and origins.pop(eligibility_key, None) == "default":
        offer = facts["offer"] = {k: v for k, v in offer.items() if k != "eligibility"}
    if (offer and "eligibility" not in offer and facts.get("audience_hint", "").lower()
            in {"all", "all buyers", "all shoppers", "all eligible buyers"}):
        facts["offer"] = {**offer, "eligibility": "all"}
        origins[eligibility_key] = "default"
    context = resolve(facts)
    blockers = list(context["blockers"])

    def block(code, field, message):
        blockers.append({"code": code, "field": "/facts/" + field, "message": message})

    for field in ("channel", "offer", "objective", "audience_hint", "budget_thb",
                  "start_date", "end_date"):
        if field not in facts:
            block("missing_input", field, "Please provide " + field.replace("_", " ") + ".")
    if "start_date" in facts and "end_date" in facts:
        start = date.fromisoformat(facts["start_date"])
        days = (date.fromisoformat(facts["end_date"]) - start).days
        if not 1 <= days <= 56:
            block("unsupported_choice", "end_date", "Select a campaign lasting 1 to 56 days.")
    pending = []
    for candidate in candidates:
        field = candidate["field"].removeprefix("/facts/")
        proposed = candidate["value"]
        entered = facts.get(field)
        if isinstance(proposed, dict) and isinstance(entered, dict):
            if all(entered.get(key) == value for key, value in proposed.items()):
                continue
        elif proposed == entered:
            continue
        pending.append({**candidate, "candidate_id": str(uuid4())})
        block("conflicting_values" if field in facts else "missing_input", field,
              "Confirm the quoted suggestion or enter " + field.replace("_", " ") + ".")
    prompts = {
        "product_choice_id": "Would you like to use the synthetic 12-roll toilet tissue listing "
                             "on Shopee? Reply ‘use the demo’ to select it.",
        "channel": "Which sales channel should this campaign use?",
        "objective": "What outcome do you want to measure? This rehearsal supports "
                     "incremental profit.",
        "audience_hint": "Who should receive the offer? The demo supports all buyers.",
        "offer": "What discount should each pack receive, in percent or as a THB voucher?",
        "budget_thb": "What is the maximum total subsidy budget in THB?",
        "start_date": "When should the campaign start? You can also give the whole period.",
        "end_date": "What is the first day the offer will no longer be active?",
    }
    questions = []
    for item in blockers:
        field = item["field"]
        if any(question["field"] == field for question in questions):
            continue
        if any(candidate["field"] == field for candidate in pending):
            prompt = "Check the proposed " + field.split("/")[-1].replace("_", " ") + (
                " and confirm the intended value or reply with a correction."
            )
        elif item["code"] in {"missing_input", "exact_product_required"}:
            prompt = prompts.get(field.split("/")[-1], item["message"])
        else:
            prompt = item["message"]
        questions.append({"field": field, "prompt": prompt, "blocking": True})
    if agent["status"] != "not_requested":
        questions, interview = order_questions(questions, facts, pending)
        if interview and agent["status"] != "degraded":
            agent = interview
    return {
        **context, "review_id": review_id, "revision": revision, "facts": facts,
        "fact_origins": origins, "agent": agent, "candidates": pending,
        "status": "blocked" if blockers else "ready_for_approval",
        "blockers": blockers, "questions": questions,
    }


@router.post("", status_code=201)
def create_review(payload: dict):
    validate_request("CreateReview", payload)
    facts = payload.get("fields", {})
    candidates, agent = (extract(payload["brief"]) if "brief" in payload else
                         ([], {"status": "not_requested", "reason": None}))
    review_id = str(uuid4())
    review = build_review(review_id, 1, facts, {"/facts/" + key: "planner" for key in facts},
                          candidates, agent)
    with db.v1_transaction() as state:
        state["reviews"][review_id] = [review]
    return review


def current_review(state, review_id, revision=None):
    history = state["reviews"].get(str(review_id))
    if not history:
        raise V1Error(404, "not_found", "Review not found.")
    review = history[-1]
    if revision is not None and review["revision"] != revision:
        raise V1Error(409, "stale_review", "A newer revision exists.",
                      current_revision=review["revision"], current_review=review)
    return review


@router.get("/{review_id}")
def get_review(review_id: UUID):
    with db.v1_transaction() as state:
        return current_review(state, review_id)


@router.post("/{review_id}/revisions", status_code=201)
def revise_review(review_id: UUID, payload: dict):
    validate_request("ReviseReview", payload)
    edits = payload.get("edits", {})
    with db.v1_transaction() as state:
        review = current_review(state, review_id, payload["base_revision"])
    facts = review["facts"].copy()
    origins = review["fact_origins"].copy()
    answered = set(edits)
    candidates = {item["candidate_id"]: item for item in review["candidates"]}
    confirmations = payload.get("confirm", [])
    agent = review["agent"]
    suggestions = []
    if "answer" in payload:
        answer = payload["answer"].strip()
        if answer.lower().rstrip(".!") in {"confirm all", "confirm all suggestions"}:
            if not candidates:
                raise V1Error(422, "invalid_request", "There are no suggestions to confirm.")
            confirmations = list(candidates)
        elif affirmative(answer):
            question = next(iter(review["questions"]), {})
            matches = [key for key, item in candidates.items()
                       if item["field"] == question.get("field")]
            if len(matches) == 1 and "confirm" in question["prompt"].lower():
                confirmations = matches
            elif not matches and question.get("field") == "/facts/product_choice_id" and (
                "would you like to use the synthetic" in question["prompt"].lower()
            ):
                facts["product_choice_id"] = PRODUCT["choice_id"]
                origins["/facts/product_choice_id"] = "planner"
            else:
                raise V1Error(422, "invalid_request", "Please give a value or select a suggestion.")
        else:
            suggestions, agent = extract(answer, review)
            # A conversational amount correction keeps previously agreed redemption terms.
            existing_offer = facts.get("offer", {})
            previous_offers = [c["value"] for c in candidates.values()
                               if c["field"] == "/facts/offer"]
            if len(previous_offers) == 1:
                existing_offer = {**existing_offer, **previous_offers[0]}
            for suggestion in suggestions:
                if suggestion["field"] == "/facts/offer":
                    inherited = {k: v for k, v in existing_offer.items()
                                 if k in {"eligibility", "cap_per_user", "min_spend_thb"}
                                 and origins.get("/facts/offer/" + k) != "default"}
                    suggestion["value"] = {**inherited, **suggestion["value"]}
    for candidate_id in confirmations:
        item = candidates.get(candidate_id)
        if item is None:
            raise V1Error(422, "invalid_request", "Unknown candidate for this revision.")
        field = item["field"].removeprefix("/facts/")
        if field in answered:
            raise V1Error(422, "invalid_request", "Provide one answer per field.")
        answered.add(field)
        facts[field] = item["value"]
        if field == "offer":
            origins.pop("/facts/offer/eligibility", None)
        origins[item["field"]] = "planner_confirmed_extraction"
    for key, value in edits.items():
        pointer = "/facts/" + key
        if key == "offer":
            origins.pop("/facts/offer/eligibility", None)
        if value is None:
            facts.pop(key, None)
            origins.pop(pointer, None)
        else:
            facts[key] = value
            origins[pointer] = "planner"
    pending = [item for item in candidates.values()
               if item["field"].removeprefix("/facts/") not in answered]
    replaced = {item["field"] for item in suggestions}
    pending = [item for item in pending if item["field"] not in replaced] + suggestions
    updated = build_review(str(review_id), review["revision"] + 1, facts, origins,
                           pending, agent)
    confirmed = answered.difference(edits)
    if edits and confirmed:
        for blocker in updated["blockers"]:
            if (blocker["code"] == "unsupported_choice"
                    and blocker["field"].removeprefix("/facts/") in confirmed):
                raise V1Error(422, "invalid_request", "The edits conflict with a confirmed value.")
    with db.v1_transaction() as state:
        current_review(state, review_id, payload["base_revision"])
        state["reviews"][str(review_id)].append(updated)
        return updated


@router.post("/{review_id}/approvals", status_code=201)
def approve_review(review_id: UUID, payload: dict):
    validate_request("ApproveReview", payload)
    with db.v1_transaction() as state:
        review = current_review(state, review_id, payload["revision"])
        if review["blockers"]:
            raise V1Error(409, "incomplete_review", "Resolve all blockers before approval.",
                          current_revision=review["revision"], blockers=review["blockers"])
        for record in state["approvals"].values():
            receipt = record["receipt"]
            if receipt["review_id"] == str(review_id) and receipt["revision"] == review["revision"]:
                return JSONResponse(receipt)
        receipt = {
            "approval_id": str(uuid4()), "review_id": str(review_id),
            "revision": review["revision"], "approved_at": datetime.now(timezone.utc).isoformat(),
            "data_versions": review["data_versions"],
        }
        state["approvals"][receipt["approval_id"]] = {
            "receipt": receipt, "review": review,
            "actor": os.environ.get("CAMPAIGNTWIN_PLANNER_ID", "local-planner"),
        }
        return receipt
