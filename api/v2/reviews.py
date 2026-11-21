"""Planner intake, revision history and explicit approval for the seller workflow."""

import os
from datetime import date, datetime, timezone
from uuid import UUID, uuid4

from fastapi.responses import JSONResponse

import db
from agents.review_v2 import extract, order_questions
from api.v2.catalog import resolve
from api.v2.contract import V2Error, validate_request

# Keep the editable facts surface compatible with the deployed form. It diffs
# every returned key and would otherwise submit null for unrecognized controls.
FORM_FIELDS = {
    "channel",
    "product_hint",
    "pack_hint",
    "product_choice_id",
    "audience_hint",
    "budget_thb",
    "start_date",
    "end_date",
    "objective",
    "offer",
}


def field_name(pointer):
    return pointer.removeprefix("/facts/").removeprefix("/commercial_inputs/")


def field_pointer(key):
    return ("/facts/" if key.split("/")[0] in FORM_FIELDS else "/commercial_inputs/") + key


def build_review(review_id, revision, facts, origins, candidates, agent):
    # Recompute derived facts whenever the user changes their underlying choice.
    if "channel" in facts:
        channel = facts["channel"].strip().lower()
        facts["channel"] = "tiktok_shop" if channel in {"tiktok shop", "tiktok-shop"} else channel
    channel_key = "/facts/channel"
    if origins.get(channel_key) == "data":
        facts.pop("channel", None)
        origins.pop(channel_key)
    offer = facts.get("offer")
    eligibility_key = "/facts/offer/eligibility"
    if offer and origins.pop(eligibility_key, None) == "default":
        offer = facts["offer"] = {k: v for k, v in offer.items() if k != "eligibility"}
    if (
        offer
        and "eligibility" not in offer
        and facts.get("audience_hint", "").lower()
        in {"all", "all buyers", "all shoppers", "all eligible buyers"}
    ):
        facts["offer"] = {**offer, "eligibility": "all"}
        origins[eligibility_key] = "default"
    for candidate in candidates:
        key = field_name(candidate["field"])
        validate_request("PlannerFields", {key: candidate["value"]})
        candidate["field"] = field_pointer(key)
    context = resolve(facts)
    if (
        context["resolved_product"]
        and "channel" not in facts
        and not any(c["field"] == channel_key for c in candidates)
    ):
        facts["channel"] = context["resolved_product"]["channel"]
        origins[channel_key] = "data"
    blockers = [
        {**b, "field": field_pointer(field_name(b["field"]))}
        if b["field"].startswith("/facts/")
        else b
        for b in context["blockers"]
    ]

    def block(code, field, message):
        blockers.append({"code": code, "field": field_pointer(field), "message": message})

    for field in (
        "channel",
        "offer",
        "objective",
        "audience_hint",
        "budget_thb",
        "start_date",
        "end_date",
    ):
        if field not in facts:
            block("missing_input", field, "Please provide " + field.replace("_", " ") + ".")
    if "start_date" in facts and "end_date" in facts:
        start = date.fromisoformat(facts["start_date"])
        days = (date.fromisoformat(facts["end_date"]) - start).days
        if not 1 <= days <= 42:
            block("unsupported_choice", "end_date", "Select a campaign lasting 1 to 42 days.")
    pending = []
    for candidate in candidates:
        field = field_name(candidate["field"])
        proposed = candidate["value"]
        entered = facts.get(field)
        if isinstance(proposed, dict) and isinstance(entered, dict):
            if all(entered.get(key) == value for key, value in proposed.items()):
                continue
        elif proposed == entered:
            continue
        pending.append({**candidate, "candidate_id": str(uuid4())})
        block(
            "conflicting_values" if field in facts else "missing_input",
            field,
            "Confirm the quoted suggestion or enter " + field.replace("_", " ") + ".",
        )
    prompts = {
        "product_choice_id": "Which supported listing and pack will you promote?",
        "channel": "Which sales channel should this campaign use?",
        "objective": "What outcome do you want to measure? This rehearsal supports "
        "incremental profit.",
        "audience_hint": "Who should receive the offer? The demo supports all buyers.",
        "offer": "What discount should each pack receive, in percent or as a THB voucher?",
        "budget_thb": "What is the total budget for vouchers and committed media in THB?",
        "start_date": "When should the campaign start? You can also give the whole period.",
        "end_date": "What is the first day the offer will no longer be active?",
    }
    questions = []
    for item in blockers:
        field = item["field"]
        if any(question["field"] == field for question in questions):
            continue
        if any(candidate["field"] == field for candidate in pending):
            prompt = (
                "Check the proposed "
                + field.split("/")[-1].replace("_", " ")
                + (" and confirm the intended value or reply with a correction.")
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
        **context,
        "review_id": review_id,
        "revision": revision,
        "facts": {k: v for k, v in facts.items() if k in FORM_FIELDS},
        "commercial_inputs": {k: v for k, v in facts.items() if k not in FORM_FIELDS},
        "fact_origins": origins,
        "agent": agent,
        "candidates": pending,
        "status": "blocked" if blockers else "ready_for_approval",
        "blockers": blockers,
        "questions": questions,
    }


def create_review(payload: dict):
    validate_request("CreateReview", payload)
    facts = payload.get("fields", {})
    candidates, agent = (
        extract(payload["brief"])
        if "brief" in payload
        else ([], {"status": "not_requested", "reason": None})
    )
    review_id = str(uuid4())
    review = build_review(
        review_id, 1, facts, {field_pointer(key): "planner" for key in facts}, candidates, agent
    )
    with db.v2_transaction() as state:
        state["reviews"][review_id] = [review]
    return review


def current_review(state, review_id, revision=None):
    history = state["reviews"].get(str(review_id))
    if not history:
        raise V2Error(404, "not_found", "Review not found.")
    review = history[-1]
    if revision is not None and review["revision"] != revision:
        raise V2Error(
            409,
            "stale_review",
            "A newer revision exists.",
            current_revision=review["revision"],
            current_review=review,
        )
    return review


def get_review(review_id: UUID):
    with db.v2_transaction() as state:
        return current_review(state, review_id)


def revise_review(review_id: UUID, payload: dict):
    validate_request("ReviseReview", payload)
    edits = payload.get("edits", {})
    with db.v2_transaction() as state:
        review = current_review(state, review_id, payload["base_revision"])
    facts = {**review["facts"], **review.get("commercial_inputs", {})}
    origins = review["fact_origins"].copy()
    answered = set(edits)
    candidates = {item["candidate_id"]: item for item in review["candidates"]}
    confirmations = payload.get("confirm", [])
    if "answer" in payload:
        answer = payload["answer"].strip()
        if answer.lower().rstrip(".!") in {"confirm all", "confirm all suggestions"}:
            if not candidates:
                raise V2Error(422, "invalid_request", "There are no suggestions to confirm.")
            confirmations = list(candidates)
        else:
            # The revision alias and the guided endpoint must interpret the displayed question.
            from api.v2.interview import conversation_turn

            return conversation_turn(review_id, review, answer)
    for candidate_id in confirmations:
        item = candidates.get(candidate_id)
        if item is None:
            raise V2Error(422, "invalid_request", "Unknown candidate for this revision.")
        field = field_name(item["field"])
        if field in answered:
            raise V2Error(422, "invalid_request", "Provide one answer per field.")
        answered.add(field)
        facts[field] = item["value"]
        if field == "offer":
            origins.pop("/facts/offer/eligibility", None)
        origins[item["field"]] = "planner_confirmed_extraction"
    for key, value in edits.items():
        pointer = field_pointer(key)
        if key == "offer":
            origins.pop("/facts/offer/eligibility", None)
        if value is None:
            facts.pop(key, None)
            origins.pop(pointer, None)
        else:
            facts[key] = value
            origins[pointer] = "planner"
    pending = [item for item in candidates.values() if field_name(item["field"]) not in answered]
    if facts:
        validate_request("PlannerFields", facts)
    updated = build_review(
        str(review_id), review["revision"] + 1, facts, origins, pending, review["agent"]
    )
    quotes = {
        k: v for k, v in review.get("confirmed_quotes", {}).items() if field_name(k) not in answered
    }
    quotes.update({candidates[i]["field"]: candidates[i]["brief_quote"] for i in confirmations})
    if quotes:
        updated["confirmed_quotes"] = quotes
    confirmed = answered.difference(edits)
    if edits and confirmed:
        for blocker in updated["blockers"]:
            if (
                blocker["code"] == "unsupported_choice"
                and field_name(blocker["field"]) in confirmed
            ):
                raise V2Error(422, "invalid_request", "The edits conflict with a confirmed value.")
    with db.v2_transaction() as state:
        current_review(state, review_id, payload["base_revision"])
        state["reviews"][str(review_id)].append(updated)
        return updated


def approve_review(review_id: UUID, payload: dict):
    validate_request("ApproveReview", payload)
    with db.v2_transaction() as state:
        review = current_review(state, review_id, payload["revision"])
        if review["blockers"]:
            raise V2Error(
                409,
                "incomplete_review",
                "Resolve all blockers before approval.",
                current_revision=review["revision"],
                blockers=review["blockers"],
            )
        for record in state["approvals"].values():
            receipt = record["receipt"]
            if receipt["review_id"] == str(review_id) and receipt["revision"] == review["revision"]:
                return JSONResponse(receipt)
        receipt = {
            "approval_id": str(uuid4()),
            "review_id": str(review_id),
            "revision": review["revision"],
            "approved_at": datetime.now(timezone.utc).isoformat(),
            "data_versions": review["data_versions"],
        }
        state["approvals"][receipt["approval_id"]] = {
            "receipt": receipt,
            "review": review,
            "actor": os.environ.get("CAMPAIGNTWIN_PLANNER_ID", "local-planner"),
        }
        return receipt
