import json
import logging
from pathlib import Path
from time import sleep
from uuid import UUID, uuid4

import psycopg
from fastapi import APIRouter, BackgroundTasks, HTTPException

import db
import demo
from contracts.models import (
    BriefParseRequest,
    BriefParseResponse,
    CampaignSpec,
    ContextPack,
    DemoReplay,
    HealthResponse,
    Report,
    RunCreated,
    RunStatus,
    SimulationPlan,
)
from mocks import load_mock

router = APIRouter(prefix="/api")
logger = logging.getLogger(__name__)
REPLAY_PATH = Path(demo.__file__).with_name("replay.json")


@router.get("/health", response_model=HealthResponse)
def health():
    try:
        db.check_connection()
    except (psycopg.Error, ValueError) as error:
        raise HTTPException(status_code=503, detail="Run storage unavailable") from error
    return {"ok": True, "db_mode": db.get_mode()}


@router.post("/brief/parse", response_model=BriefParseResponse)
def parse_brief(request: BriefParseRequest):
    return load_mock("campaign_spec")


@router.post("/plan/context", response_model=ContextPack)
def plan_context(spec: CampaignSpec):
    return load_mock("context_pack")


def _run_fixture(run_id: str) -> None:
    progress = 0.0
    try:
        fixture = RunStatus.model_validate(load_mock("run_status"))
        results = fixture.partial_results
        published = 0
        db.update_progress(run_id, progress)
        for step in range(1, 11):
            sleep(1)
            end = step * len(results) // 10
            for result in results[published:end]:
                db.save_result(run_id, result.model_dump(mode="json"))
            published = end
            progress = step / 10
            status = "done" if step == 10 else "running"
            db.update_progress(run_id, progress, status=status, generation=fixture.generation)
    except Exception:
        logger.exception("Mock run failed for %s", run_id)
        db.update_progress(run_id, progress, status="error")


@router.post("/runs", response_model=RunCreated)
def start_run(plan: SimulationPlan, background_tasks: BackgroundTasks):
    run_id = str(uuid4())
    db.save_run(run_id, plan.model_dump(mode="json"))
    background_tasks.add_task(_run_fixture, run_id)
    return {"run_id": run_id}


def _get_run(run_id: UUID):
    run = db.get_run(str(run_id))
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@router.get("/runs/{run_id}", response_model=RunStatus)
def run_status(run_id: UUID):
    run = _get_run(run_id)
    return {**run, "partial_results": db.get_results(str(run_id))}


@router.get("/runs/{run_id}/report", response_model=Report)
def run_report(run_id: UUID):
    run = _get_run(run_id)
    if run["status"] != "done":
        raise HTTPException(status_code=409, detail="Run has no completed report")
    return load_mock("report")


@router.get("/demo/replay", response_model=DemoReplay)
def demo_replay():
    return json.loads(REPLAY_PATH.read_text(encoding="utf-8"))
