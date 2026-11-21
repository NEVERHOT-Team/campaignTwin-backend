from typing import Any, Literal

from pydantic import BaseModel, Field

Label = Literal["data", "llm_estimate", "default"]
Scenario = Literal["S1", "S2", "S3", "S4"]
Status = Literal["queued", "running", "done", "error"]


class Assumption(BaseModel):
    name: str
    value: Any
    label: Label
    source: str | None


class CampaignSpec(BaseModel):
    goal: str
    target_segments: list[str]
    budget_thb: float | None
    discount_pct: float | None
    mechanic: str | None
    channel: list[str]
    start: str | None
    end: str | None
    kpi_target: str | None
    missing_fields: list[str]


class BriefParseRequest(BaseModel):
    text: str = Field(min_length=1)


class BriefParseResponse(BaseModel):
    spec: CampaignSpec
    questions: list[str]


class RivalScenario(BaseModel):
    rival: str
    scenario: Scenario
    reaction: Literal["none", "match", "undercut", "non_price"]
    depth: float
    delay_weeks: int
    duration_weeks: int
    likelihood: float
    ci_low: float
    ci_high: float
    evidence_ids: list[str]
    rival_score: float


class ContextPack(BaseModel):
    category: str
    segments: list[dict[str, Any]]
    rivals: list[RivalScenario]
    assumptions: list[Assumption]
    gaps: list[str]


class Strategy(BaseModel):
    id: str
    name: str
    discount_pct: float
    mechanic: str
    target_segments: list[str]
    message: str
    channel_mix: dict[str, float]
    timing: dict[str, Any]
    cap_per_user: int | None
    budget_split: dict[str, float]
    contingency_rules: list[str]
    rationale: str
    is_user_plan: bool = False
    is_baseline: bool = False


class SimulationPlan(BaseModel):
    spec: CampaignSpec
    context: ContextPack
    strategies: list[Strategy]
    scenarios: list[str] = Field(default_factory=lambda: ["S1", "S2", "S3"])
    seeds: int = 10
    n_agents: int = 5000
    weeks: int = 6


class Pct(BaseModel):
    p10: float
    p50: float
    p90: float


class WeekRow(BaseModel):
    week: int
    sales: float
    profit: float
    buyers: int
    from_rival: int
    waiting: int


class KPI(BaseModel):
    incr_profit: Pct
    iroas: Pct
    leakage: Pct
    new_buyers: Pct
    p_hit_target: float
    p_loss: float


class Breakpoint(BaseModel):
    week: int | None
    cause: str | None
    share: float | None


class SimResult(BaseModel):
    strategy_id: str
    scenario: Scenario
    weekly: list[WeekRow]
    kpi: KPI
    breakpoint: Breakpoint


class RunStatus(BaseModel):
    run_id: str
    status: Status
    progress: float
    generation: int
    partial_results: list[SimResult]


class Report(BaseModel):
    summary: str
    confidence: Literal["High", "Medium", "Low"]
    kpi_tiles: list[dict[str, Any]]
    scorecard: list[dict[str, Any]]
    weekly: list[dict[str, Any]]
    audience: list[dict[str, Any]]
    competitive: dict[str, Any]
    customer_voice: list[dict[str, Any]]
    risks: list[str]
    recommendation: dict[str, Any]


class HealthResponse(BaseModel):
    ok: bool
    db_mode: Literal["json", "postgres"]


class RunCreated(BaseModel):
    run_id: str


class DemoReplay(BaseModel):
    status: RunStatus
    report: Report
