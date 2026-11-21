"""Choose one actionable question from the pinned review and preserve the v2 handoff."""

import re
from copy import deepcopy
from datetime import date
from decimal import Decimal

import db
from agents.review_v2 import affirmative, extract
from api.v2.business import unit_economics
from api.v2.catalog import COST_FIELDS
from api.v2.contract import V2Error, validate_request
from api.v2.reviews import build_review, current_review, field_name, field_pointer, revise_review

# Ask for facts that unlock other answers first. For example, dated evidence needs a listing
# and a period before the server can suggest its voucher terms or identify a missing fee.
ORDER = (
    "product_scope",
    "product_choice_id",
    "product_hint",
    "pack_hint",
    "channel",
    "start_date",
    "end_date",
    "costs",
    "audience_hint",
    "objective",
    "offer",
    "budget_thb",
)
PROMPTS = {
    "product_choice_id": (
        "Which listing and pack will you promote?",
        "The listing determines the sale unit, channel and available evidence.",
    ),
    "start_date": (
        "When will customers first be able to use the offer?",
        "The selling period determines which catalog costs apply.",
    ),
    "end_date": (
        "What is the first day customers can no longer use the offer?",
        "You can also enter the first and last selling days together.",
    ),
    "audience_hint": (
        "Who can redeem the offer?",
        "Eligibility determines who receives a subsidy.",
    ),
    "objective": (
        "Which outcome should decide whether you keep the offer?",
        "The evaluator can compare incremental contribution profit after seller costs.",
    ),
    "offer": (
        "What does the buyer get on one eligible order?",
        "The discount and redemption terms determine seller receipts.",
    ),
    "budget_thb": (
        "How much can you spend on vouchers and committed media in total, in THB?",
        "The remaining budget after media sets the maximum number of funded vouchers.",
    ),
}


def next_step(review):
    if not review["blockers"]:
        commercial = review.get("commercial_inputs", {})
        if "profit_target_thb" not in commercial:
            return {
                "mode": "choose",
                "field": "/commercial_inputs/profit_target_thb",
                "prompt": "How much extra profit would make this campaign worth running, in THB?",
                "reason": "The simulator compares median profit with ordinary trading over "
                "the campaign and follow-up. Enter a target or choose break-even.",
                "choices": [
                    {
                        "id": "break-even",
                        "label": "Cover all incremental costs (0 THB)",
                        "value": 0,
                        "source": "supported_choice",
                    }
                ],
            }
        if "alternative_policy" not in commercial and not review.get("catalog_evidence"):
            return {
                "mode": "choose",
                "field": "/commercial_inputs/alternative_policy",
                "prompt": "Is the advertised discount fixed, or may the strategist reduce it?",
                "reason": "A fixed offer is compared only with ordinary trading. An adjustable "
                "offer can also be compared with a smaller discount.",
                "choices": [
                    {
                        "id": "fixed",
                        "label": "Keep the advertised discount fixed",
                        "value": "fixed_offer",
                        "source": "supported_choice",
                    },
                    {
                        "id": "flexible",
                        "label": "Let the strategist propose a smaller discount",
                        "value": "agent_bounded",
                        "source": "supported_choice",
                    },
                    {
                        "id": "half",
                        "label": "Compare with half the original discount",
                        "value": "half_discount",
                        "source": "supported_choice",
                    },
                ],
            }
        return {
            "mode": "review",
            "field": None,
            "prompt": "Check the costs, assumptions and "
            "permitted changes, then approve this revision.",
            "reason": "Approval freezes the inputs used by the simulator.",
            "choices": [],
        }

    def priority(blocker):
        key = field_name(blocker["field"]).strip("/")
        return ORDER.index(key) if key in ORDER else len(ORDER)

    issue = min(review["blockers"], key=priority)
    pointer = issue["field"]
    key = field_name(pointer)
    candidates = [item for item in review["candidates"] if item["field"] == pointer]
    if candidates:
        return {
            "mode": "confirm",
            "field": pointer,
            "prompt": f"Is this the intended {key.replace('_', ' ')}?",
            "reason": "Check the quoted words and normalized value, or enter a correction.",
            "choices": [
                {
                    "id": c["candidate_id"],
                    "label": c["brief_quote"],
                    "value": c["value"],
                    "source": "extracted",
                }
                for c in candidates
            ],
        }

    prompt, reason = PROMPTS.get(key, (issue["message"], "This issue blocks approval."))
    if issue["code"] not in {"missing_input", "exact_product_required"}:
        reason = issue["message"]
        if key not in PROMPTS:
            prompt = "How would you like to resolve this issue?"
    choices = []

    def option(identifier, label, value, source):
        choices.append({"id": identifier, "label": label, "value": value, "source": source})

    if key == "product_choice_id":
        hint = review["facts"].get("product_hint", "").strip().lower()
        products = sorted(
            review["supported_product_choices"],
            key=lambda p: not (hint and hint in p["label"].lower()),
        )
        for p in products:
            option(
                p["choice_id"],
                f"{p['label']} · {p['pack_label']} · {p['channel']}",
                p["choice_id"],
                "catalog",
            )
    elif key == "offer" and (bundle := review.get("catalog_evidence")):
        terms = bundle["capability"]["voucher"]
        offer = {
            "kind": "voucher",
            "value": terms["discount"],
            "unit": "THB",
            "min_spend_thb": terms["minimum_spend"],
            "eligibility": "all",
        }
        option(
            "catalog-voucher",
            f"Use the {terms['discount']} THB catalog voucher "
            f"with {terms['minimum_spend']} THB minimum spend",
            offer,
            "catalog",
        )
    elif key == "objective":
        option(
            "profit",
            "Measure incremental profit after seller costs",
            "incremental_profit",
            "supported_choice",
        )
    elif key == "audience_hint":
        option("all-buyers", "Include all eligible buyers", "all", "supported_choice")
    return {
        "mode": "choose" if choices else "answer",
        "field": pointer,
        "prompt": prompt,
        "reason": reason,
        "choices": choices,
    }


def policy_inputs(review):
    evaluation = review["evaluation"]
    if evaluation is None:
        return []
    behavior = evaluation["behavior"]
    window = evaluation["measurement_window"]
    post_days = (
        date.fromisoformat(window["end_date"]) - date.fromisoformat(review["facts"]["end_date"])
    ).days
    rows = [
        (
            "profit_target_thb",
            evaluation["goal_target"]["minimum"],
            "THB",
            "Minimum median incremental contribution versus the paired baseline.",
        ),
        (
            "max_loss_probability",
            evaluation["max_loss_probability"],
            "fraction",
            "Maximum share of paired seeds losing money in each scenario.",
        ),
        (
            "min_unit_contribution_thb",
            evaluation.get("min_unit_contribution_thb"),
            "THB/pack",
            "Minimum contribution on a redeemed pack before fixed media.",
        ),
        (
            "alternative_policy",
            evaluation["alternative_policy"],
            "policy",
            "Controls whether and how the strategist may change the discount.",
        ),
        ("post_days", post_days, "days", "Additional days included in measurement."),
        (
            "discount_response",
            behavior["discount_response"],
            "multiplier",
            "Scales the assumed purchase response to a discount.",
        ),
        (
            "stockup_days",
            behavior["stockup_days"],
            "days",
            "Time in which a promoted purchase can replace a later purchase.",
        ),
        (
            "rival_demand_reduction_pct",
            behavior["rival_demand_reduction_pct"],
            "percent",
            "Shared demand stress. Zero runs S1 alone. Likelihood is unknown.",
        ),
    ]
    result = []
    bundle = review.get("catalog_evidence")
    for key, value, unit, basis in rows:
        pointer = field_pointer(key)
        if key in review.get("commercial_inputs", {}):
            origin, reference = review["fact_origins"][pointer], pointer
        elif key == "alternative_policy" and bundle:
            reference = bundle["capability"]["evidence_ref"]
            origin = next(s["label"] for s in bundle["sources"] if s["evidence_ref"] == reference)
        else:
            origin, reference = "default", review["data_versions"]["evaluation_policy"]
        result.append({"field": key, "value": value, "unit": unit, "basis": basis,
                       "origin": origin, "evidence_ref": reference})
    return result


def present(review):
    identifier = review["review_id"]
    url = f"/api/v2/campaign-reviews/{identifier}"
    cost_inputs = [
        {
            "field": field,
            **review["costs"][cost],
            "origin": review["fact_origins"].get(
                field_pointer(field), review["costs"][cost]["source"]
            ),
        }
        for field, (cost, *_) in COST_FIELDS.items()
        if cost in review["costs"]
    ]
    return {
        "review": review,
        "next_step": next_step(review),
        "cost_inputs": cost_inputs,
        "policy_inputs": policy_inputs(review),
        "unit_economics": unit_economics(review),
        "links": {
            "self": url,
            "turn": url + "/turns",
            "what_if": url + "/what-if",
            "approve": f"/api/v2/campaign-reviews/{identifier}/approvals",
            "run": "/api/v2/runs",
        },
    }


def focused_candidate(review, step, answer):
    key = field_name(step["field"] or "")
    for option in step["choices"]:
        if step["mode"] == "choose" and answer.casefold() in {
            option["id"].casefold(),
            option["label"].casefold(),
        }:
            return {"field": step["field"], "value": option["value"], "brief_quote": answer}
    if key == "product_choice_id":
        # A short listing reply is safe to quote when it matches catalog words verbatim.
        matches = [
            p
            for p in review["supported_product_choices"]
            if answer.casefold()
            in {p["choice_id"].casefold(), p["label"].casefold(), *p["label"].casefold().split()}
        ]
        if len(matches) == 1:
            return {"field": step["field"], "value": matches[0]["choice_id"], "brief_quote": answer}
    # A bare amount belongs to the displayed question. Bind the unit before invoking
    # generic extraction so a commission percentage cannot become a voucher.
    if key.endswith(("_thb", "_pct")):
        suffix = r"(?:%|percent)" if key.endswith("_pct") else r"(?:THB|baht|฿)"
        match = re.fullmatch(
            r"(-?(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]+)?)" + r"\s*" + suffix + "?",
            answer,
            re.I,
        )
        if match:
            amount = Decimal(match[1].replace(",", ""))
            value = int(amount) if amount == amount.to_integral_value() else float(amount)
            return {"field": step["field"], "value": value, "brief_quote": answer}
    return None


def conversation_turn(review_id, review, answer):
    step = next_step(review)
    revision = review["revision"]
    if affirmative(answer) or answer.lower().rstrip(".!") in {
        "confirm all",
        "confirm all suggestions",
    }:
        if step["mode"] == "confirm" and len(step["choices"]) == 1 and affirmative(answer):
            return revise_review(
                review_id, {"base_revision": revision, "confirm": [step["choices"][0]["id"]]}
            )
        raise V2Error(422, "choose_explicitly", "Select the intended choices or enter a value.")

    # Bind a short reply ("1000", "next week") to the question actually shown by v2.
    # Reuse the sanitized extractor. A turn only saves candidates until the planner chooses.
    focus = {"field": step["field"] or "", "prompt": step["prompt"]}
    candidate = focused_candidate(review, step, answer)
    if candidate:
        proposed, agent = [candidate], {"status": "not_requested", "reason": None}
    else:
        proposed, agent = extract(answer, {**review, "questions": [focus]})
    if not proposed:
        raise V2Error(
            422,
            "answer_unresolved",
            "No campaign value was identified. Use a choice or a structured edit.",
            agent=agent,
        )
    facts = {**review["facts"], **review.get("commercial_inputs", {})}
    for item in proposed:
        key = field_name(item["field"])
        if key == "offer":
            previous = [c["value"] for c in review["candidates"] if c["field"] == "/facts/offer"]
            offer = dict(facts.get("offer", {}))
            if len(previous) == 1:
                offer.update(previous[0])
            # Preserve confirmed restrictions when a seller changes just the amount.
            inherited = {
                k: offer[k]
                for k in ("min_spend_thb", "cap_per_user", "eligibility")
                if k in offer and review["fact_origins"].get("/facts/offer/" + k) != "default"
            }
            item["value"] = {**inherited, **item["value"]}
        validate_request("PlannerFields", {key: item["value"]})
        item["field"] = field_pointer(key)
    changed = {item["field"] for item in proposed}
    pending = [deepcopy(c) for c in review["candidates"] if c["field"] not in changed]
    updated = build_review(
        str(review_id),
        revision + 1,
        facts,
        review["fact_origins"].copy(),
        pending + proposed,
        agent,
    )
    if review.get("confirmed_quotes"):
        updated["confirmed_quotes"] = review["confirmed_quotes"]
    with db.v2_transaction() as state:
        current_review(state, review_id, revision)
        state["reviews"][str(review_id)].append(updated)
    return updated
