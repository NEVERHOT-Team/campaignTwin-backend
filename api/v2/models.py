"""V2 wire types. Link shared shapes to the canonical OpenAPI served on the same origin."""

import re
from datetime import date, timedelta
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator


class Input(BaseModel):
    model_config = ConfigDict(extra="forbid")


def shared(name):
    return Field(description="See " + name + " in the shared planner contract.")


Fields = Annotated[dict, shared("PlannerFields")]
Edits = Annotated[dict, shared("PlannerEdits")]
Revision = Annotated[int, Field(strict=True, ge=1)]
Message = Annotated[str, Field(min_length=1, max_length=1000, pattern=r"\S")]
Count = Annotated[int, Field(strict=True, ge=0, le=1_000_000_000)]


class SellingPeriod(Input):
    first_day: date = Field(description="First local selling day, inclusive.", strict=False)
    last_day: date = Field(description="Last local selling day, inclusive.", strict=False)

    @field_validator("first_day", "last_day", mode="before")
    @classmethod
    def local_calendar_date(cls, value):
        if not isinstance(value, str) or not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value):
            raise ValueError("Enter a local selling date in YYYY-MM-DD format.")
        return value

    @model_validator(mode="after")
    def supported_window(self):
        if self.last_day == date.max or not 0 <= (self.last_day - self.first_day).days < 42:
            raise ValueError("Choose between 1 and 42 selling days.")
        return self

    def edits(self):
        return {
            "start_date": self.first_day.isoformat(),
            "end_date": (self.last_day + timedelta(days=1)).isoformat(),
        }


class Start(Input):
    brief: Message | None = None
    fields: Fields | None = None
    selling_period: SellingPeriod | None = None


class Reply(Input):
    base_revision: Revision
    answer: Message


class Choose(Input):
    base_revision: Revision
    choice: str = Field(min_length=1, description="An id from next_step.choices for this revision.")


class Change(Input):
    base_revision: Revision
    edits: Edits | None = None
    confirm: list[str] | None = None
    selling_period: SellingPeriod | None = None


class VolumeCase(Input):
    base_revision: Revision
    redeemed_orders: Count
    additional_orders: Count = Field(description="Redeemed orders absent in the ordinary case.")


class Choice(BaseModel):
    id: str
    label: str
    value: JsonValue
    source: Literal["catalog", "supported_choice", "extracted"]


class Step(BaseModel):
    mode: Literal["choose", "confirm", "answer", "review"]
    field: str | None
    prompt: str
    reason: str
    choices: list[Choice]


class PolicyInput(BaseModel):
    field: str
    value: JsonValue
    unit: str
    basis: str
    origin: str
    evidence_ref: str


class CostInput(BaseModel):
    field: str
    value: float | None
    unit: str
    basis: str
    origin: str
    evidence_ref: str
    source: str
    status: str
    reason: str | None


class UnitEconomics(BaseModel):
    ordinary_contribution_thb: float
    redeemed_contribution_thb: float
    subsidy_per_order_thb: float
    committed_media_thb: float
    voucher_budget_thb: float
    max_funded_redemptions: int


class Links(BaseModel):
    self: str
    turn: str
    what_if: str
    approve: str
    run: str


class Session(BaseModel):
    review: Annotated[dict, shared("Review")]
    next_step: Step
    cost_inputs: list[CostInput]
    policy_inputs: list[PolicyInput]
    unit_economics: UnitEconomics | None = Field(description="Preview on resolved draft terms.")
    links: Links


class VolumeResult(BaseModel):
    review_id: UUID
    revision: Revision
    basis: str
    redeemed_orders: int
    additional_orders: int
    ordinary_buyers_redeeming_orders: int
    campaign_spend_thb: float
    incremental_profit_thb: float
    profit_target_thb: float
    meets_profit_target: bool
    minimum_additional_orders_for_target: int | None
    target_reachable_at_this_volume: bool
