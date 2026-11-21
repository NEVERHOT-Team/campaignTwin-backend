"""Admission and retrieval for runs bound to a frozen campaign review."""

import logging
from uuid import UUID, uuid4

from fastapi import APIRouter, BackgroundTasks
from fastapi.responses import JSONResponse

import db
from agents.voucher_strategy import propose_discount
from api.historical_prediction import compare_prediction
from api.historical_reference import pin_reference
from api.v2.catalog import resolve
from api.v2.contract import V2Error, validate_request
from api.v2.evaluation import business_evaluation
from engine.seller_simulation import evaluate

router = APIRouter(prefix="/runs", tags=["Runs"])
logger = logging.getLogger(__name__)


def _accepted(run_id, status):
    url = f"/api/v2/runs/{run_id}"
    return JSONResponse({"run_id": run_id, "status_url": url}, status, headers={"Location": url})


def _existing_run(state, run_id):
    run = state["runs"].get(run_id)
    if run is None:
        raise V2Error(404, "not_found", "Run not found.")
    return run


def _run_worker(run_id):
    with db.v2_run_worker(run_id) as acquired:
        if not acquired:
            return
        with db.v2_transaction() as state:
            run = _existing_run(state, run_id)
            if run["report"] is not None or run["failure"] is not None:
                return
            if run.get("stage") == "running":
                run["failure"] = {
                    "code": "evaluation_failed",
                    "retryable": True,
                    "message": "The previous worker stopped before saving results.",
                }
                return
            run["stage"] = "running"
            review = run["review"]
            approval_id = run["request"]["approval_id"]
        report, failure = None, None
        try:
            if review["evaluation"].get("alternative_policy") == "agent_bounded":
                proposal = propose_discount(review)
                with db.v2_transaction() as state:
                    state["runs"][run_id]["strategy_proposal"] = proposal
                report = evaluate(run_id, approval_id, review, proposal=proposal)
            else:
                report = evaluate(run_id, approval_id, review)
        except (KeyError, TypeError, ValueError):
            logger.exception("Invalid approved simulation input for %s", run_id)
            failure = {
                "code": "invalid_approved_input",
                "retryable": False,
                "message": "Approved inputs are incompatible with the evaluator.",
            }
        except Exception:
            logger.exception("Evaluation failed for %s", run_id)
            failure = {
                "code": "evaluation_failed",
                "retryable": True,
                "message": "The evaluator failed before publishing a report.",
            }
        with db.v2_transaction() as state:
            run = _existing_run(state, run_id)
            run.update(report=report, failure=failure, stage="error" if failure else "done")


@router.post("", operation_id="submitRun", status_code=202)
def submit_run(payload: dict, background_tasks: BackgroundTasks):
    validate_request("SubmitRun", payload)
    inputs = {key: str(UUID(value)) for key, value in payload.items()}
    approval_id = inputs["approval_id"]
    submission_id = inputs["submission_id"]
    parent_id = inputs.get("retry_of_run_id")
    with db.v2_transaction() as state:
        existing_id = state["submissions"].get(submission_id)
        if existing_id:
            if state["runs"][existing_id]["request"] != inputs:
                raise V2Error(409, "submission_id_conflict", "Submission ID has different inputs.")
            existing = state["runs"][existing_id]
            if existing["report"] is None and existing["failure"] is None:
                background_tasks.add_task(_run_worker, existing_id)
            return _accepted(existing_id, 200)
        approval = state["approvals"].get(approval_id)
        if approval is None:
            raise V2Error(404, "not_found", "Approval not found.")
        review = approval["review"]
        latest = state["reviews"][review["review_id"]][-1]
        if latest["revision"] != review["revision"]:
            raise V2Error(409, "approval_stale", "Approve the current review before submitting.")
        available = resolve({**review["facts"], **review.get("commercial_inputs", {})})
        if available["blockers"] or available["data_versions"] != review["data_versions"]:
            raise V2Error(409, "approval_stale", "Approve the currently available inputs.")
        run_id = str(uuid4())
        if parent_id:
            parent = _existing_run(state, parent_id)
            if parent["request"]["approval_id"] != approval_id or not (parent["failure"] or {}).get(
                "retryable"
            ):
                raise V2Error(409, "retry_not_allowed", "The parent is ineligible for this retry.")
            if parent.get("retry_run_id"):
                raise V2Error(
                    409,
                    "retry_already_submitted",
                    "A direct retry already exists.",
                    run_id=parent["retry_run_id"],
                )
            parent["retry_run_id"] = run_id

        run = {
            "request": inputs,
            "review": review,
            "report": None,
            "failure": None,
            "historical_snapshot": pin_reference(review),
        }
        run["stage"] = "queued"
        state["runs"][run_id] = run
        state["submissions"][submission_id] = run_id
    background_tasks.add_task(_run_worker, run_id)
    return _accepted(run_id, 202)


@router.get("/{run_id}", operation_id="getRunStatus")
def run_status(run_id: UUID, background_tasks: BackgroundTasks):
    with db.v2_transaction() as state:
        run = _existing_run(state, str(run_id))
        failed = run["failure"] is not None
        if run["report"] is None and not failed:
            background_tasks.add_task(_run_worker, str(run_id))
            return {
                "run_id": str(run_id),
                "status": run.get("stage", "queued"),
                "progress": 0,
                "poll_after_seconds": 1,
                "report_url": None,
                "failure": None,
            }
        return {
            "run_id": str(run_id),
            "status": "error" if failed else "done",
            "progress": 0 if failed else 1,
            "poll_after_seconds": None,
            "report_url": None if failed else f"/api/v2/runs/{run_id}/report",
            "failure": run["failure"],
        }


@router.get("/{run_id}/report", operation_id="getRunReport")
def run_report(run_id: UUID):
    with db.v2_transaction() as state:
        run = _existing_run(state, str(run_id))
        if run["failure"]:
            raise V2Error(409, "run_failed", "This run ended before publishing a report.")
        if run["report"] is None:
            raise V2Error(409, "report_pending", "The run has no completed report yet.")
        return run["report"]


@router.get("/{run_id}/input", operation_id="getRunInput")
def run_input(run_id: UUID):
    """Return the approved review and identifiers supplied to the evaluator."""
    with db.v2_transaction() as state:
        run = _existing_run(state, str(run_id))
        inputs = {
            "run_id": str(run_id),
            "approval_id": run["request"]["approval_id"],
            "review": run["review"],
        }
        if "strategy_proposal" in run:
            inputs["strategy_proposal"] = run["strategy_proposal"]
        return inputs


@router.get("/{run_id}/evaluation", operation_id="getBusinessEvaluation")
def run_evaluation(run_id: UUID):
    return business_evaluation(run_report(run_id))


@router.get("/{run_id}/historical-comparison", operation_id="getHistoricalComparison")
def historical_comparison(run_id: UUID):
    with db.v2_transaction() as state:
        run = _existing_run(state, str(run_id))
        if run["failure"]:
            raise V2Error(409, "run_failed", "This run ended before publishing a report.")
        if run["report"] is None:
            raise V2Error(409, "report_pending", "The run has no completed report yet.")
        snapshot = run.get("historical_snapshot", {"reference": None, "reason": "snapshot_missing"})
        reference = snapshot["reference"]
        if reference is None:
            result = {"status": "unavailable", "reason": snapshot["reason"], "prediction": None}
        else:
            result = compare_prediction(reference)
        return {
            "run_id": str(run_id),
            "approval_id": run["request"]["approval_id"],
            "revision": run["review"]["revision"],
            "reference": reference,
            **result,
        }
