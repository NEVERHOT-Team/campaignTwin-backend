"""CampaignSpec v0.1: the user's brief, before data matching and plan review."""

from datetime import date
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Percent = Annotated[float, Field(ge=0, le=100)]
Money = Annotated[float, Field(ge=0)]
Channel = Literal["shopee", "lazada", "tiktok_shop", "line", "meta", "google", "offline"]
Objective = Literal[
    "incremental_profit", "revenue", "new_buyers", "repeat_rate", "awareness", "leads"
]
MechanicType = Literal[
    "percent_off", "fixed_off", "voucher", "free_shipping", "bundle", "gift",
    "coins_cashback", "flash_sale",
]


class SpecModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, allow_inf_nan=False)


class Brand(SpecModel):
    name: str | None = Field(default=None, min_length=1)
    category: str | None = Field(default=None, min_length=1)
    price_tier: Literal["budget", "mid", "premium"] | None = None


class SmartGoal(SpecModel):
    metric: str = Field(min_length=1)
    target: float
    unit: str = Field(min_length=1)
    deadline: date | None = None


class CampaignObjective(SpecModel):
    primary: Objective = "incremental_profit"
    secondary: list[Objective] = Field(default_factory=list)
    smart_goals: list[SmartGoal] = Field(default_factory=list)


class Audience(SpecModel):
    description: str | None = Field(default=None, min_length=1)
    include_hints: list[str] = Field(default_factory=list)
    exclude_hints: list[str] = Field(default_factory=list)
    geo: list[str] = Field(default_factory=list)


class Mechanic(SpecModel):
    type: MechanicType
    value: Money | None = None
    value_unit: Literal["pct", "thb"] | None = None
    min_spend_thb: Money | None = None
    cap_per_user: int | None = Field(default=None, ge=1)
    eligibility: Literal["all", "new_only", "members", "segment_hint"] = "all"
    duration_days: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_value_unit(self):
        expected = {"percent_off": "pct", "fixed_off": "thb", "free_shipping": "thb"}
        unit = expected.get(self.type)
        if unit and self.value_unit not in (None, unit):
            raise ValueError(f"{self.type} value_unit must be {unit}")
        if self.value is not None:
            if not unit and self.value_unit is None:
                raise ValueError(f"{self.type} with a value requires value_unit: pct or thb")
            if (unit or self.value_unit) == "pct" and self.value > 100:
                raise ValueError("Percentage mechanic value must be between 0 and 100")
        return self

    @property
    def percentage(self) -> float | None:
        if self.type == "percent_off" or self.value_unit == "pct":
            return self.value
        return None


class CampaignMessage(SpecModel):
    headline: str | None = None
    key_message: str | None = None
    tone: str | None = None


class Offer(SpecModel):
    mechanics: list[Mechanic] = Field(default_factory=list)
    message: CampaignMessage | None = None


class BudgetSplit(SpecModel):
    """Absolute THB allocations, unlike Strategy.budget_split's engine contract."""

    discount: Money = 0
    media: Money = 0
    fees: Money = 0


class Budget(SpecModel):
    total_thb: Money | None = None
    split: BudgetSplit | None = None
    max_discount_pct: Percent | None = None

    @model_validator(mode="after")
    def validate_split(self):
        if self.split is not None and self.total_thb is not None:
            if sum(self.split.model_dump().values()) > self.total_thb + 1e-8:
                raise ValueError("budget.split allocations must not exceed budget.total_thb")
        return self


class Timing(SpecModel):
    start_date: date | None = None
    end_date: date | None = None
    event: Literal[
        "1.1", "2.2", "3.3", "4.4", "5.5", "6.6", "7.7", "8.8", "9.9",
        "10.10", "11.11", "12.12", "payday", "none",
    ] | None = None
    post_window_weeks: int = Field(default=2, ge=0)

    @model_validator(mode="after")
    def validate_dates(self):
        if self.start_date is not None and self.end_date is not None:
            days = (self.end_date - self.start_date).days
            if not 0 < days <= 56:
                raise ValueError("Campaign end must follow start by at most 8 weeks")
        return self


class UnitEconomics(SpecModel):
    avg_price_thb: float | None = Field(default=None, gt=0)
    margin_pct: Percent | None = None
    fee_pct: Percent | None = None


class Constraints(SpecModel):
    margin_floor_pct: Percent | None = None
    max_loss_prob: float = Field(default=0.2, ge=0, le=1)
    must_avoid: list[str] = Field(default_factory=list)
    must_include: list[str] = Field(default_factory=list)


class StrategistBounds(SpecModel):
    discount_pct: tuple[Percent, Percent] | None = None
    mechanics_allowed: list[MechanicType] = Field(default_factory=list)
    budget_flex_pct: Percent = 0
    timing_shift_days: int = Field(default=0, ge=0)
    segments_locked: bool = False

    @model_validator(mode="after")
    def validate_range(self):
        if self.discount_pct is not None and self.discount_pct[0] > self.discount_pct[1]:
            raise ValueError("strategist_bounds.discount_pct minimum must not exceed maximum")
        return self


class FieldMeta(SpecModel):
    source: Literal["user", "extracted", "default"]
    confidence: float = Field(ge=0, le=1)
    evidence: str | None = None

    @model_validator(mode="after")
    def require_extraction_quote(self):
        if self.source == "extracted" and not self.evidence:
            raise ValueError("Extracted values require an evidence quote from the brief")
        return self


class OpenQuestion(SpecModel):
    field: str = Field(min_length=1)
    question: str = Field(min_length=1)
    blocking: bool = False


def _leaf_paths(value: Any, prefix: str = ""):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _leaf_paths(child, f"{prefix}.{key}" if prefix else key)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _leaf_paths(child, f"{prefix}.{index}")
    elif value is not None:
        yield prefix


def _default_paths(model: BaseModel, prefix: str = ""):
    for name in type(model).model_fields:
        if name in {"field_meta", "open_questions", "spec_version", "status"}:
            continue
        value = getattr(model, name)
        path = f"{prefix}.{name}" if prefix else name
        if name not in model.model_fields_set:
            if value is not None and value != []:
                yield path
        elif isinstance(value, BaseModel):
            yield from _default_paths(value, path)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                if isinstance(child, BaseModel):
                    yield from _default_paths(child, f"{path}.{index}")


class CampaignSpec(SpecModel):
    spec_version: Literal["0.1"] = "0.1"
    campaign_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    status: Literal["draft", "needs_info", "ready"] = "draft"
    brand: Brand
    objective: CampaignObjective = Field(default_factory=CampaignObjective)
    audience: Audience
    offer: Offer
    budget: Budget
    channels: list[Channel]
    timing: Timing
    unit_economics: UnitEconomics | None = None
    constraints: Constraints = Field(default_factory=Constraints)
    strategist_bounds: StrategistBounds | None = None
    competitor_hints: list[str] = Field(default_factory=list)
    field_meta: dict[str, FieldMeta]
    open_questions: list[OpenQuestion] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_brief(self):
        self._validate_offer()
        self._validate_sources()
        self._check_missing_fields()
        return self

    def _validate_offer(self):
        for mechanic in self.offer.mechanics:
            discount = mechanic.percentage
            limit = self.budget.max_discount_pct
            if discount is not None and limit is not None and discount > limit:
                raise ValueError("Offer percentage exceeds budget.max_discount_pct")
        if self.strategist_bounds and self.strategist_bounds.discount_pct:
            minimum, maximum = self.strategist_bounds.discount_pct
            for mechanic in self.offer.mechanics:
                discount = mechanic.percentage
                if discount is not None and not minimum <= discount <= maximum:
                    raise ValueError(
                        "Offer percentage must be within strategist_bounds.discount_pct"
                    )

    def _validate_sources(self):
        payload = self.model_dump(
            mode="json", exclude={"field_meta", "open_questions", "spec_version", "status"}
        )
        for path in _default_paths(self):
            if not any(path == key or path.startswith(f"{key}.") for key in self.field_meta):
                self.field_meta[path] = FieldMeta(source="default", confidence=1)
        for path in self.field_meta:
            value = payload
            try:
                for part in path.split("."):
                    value = value[int(part)] if isinstance(value, list) else value[part]
            except (KeyError, IndexError, ValueError, TypeError) as error:
                raise ValueError(f"field_meta path does not exist: {path}") from error
        missing = [
            path for path in _leaf_paths(payload)
            if not any(path == key or path.startswith(f"{key}.") for key in self.field_meta)
        ]
        if missing:
            raise ValueError(f"Missing field_meta source for: {', '.join(missing)}")

    def _check_missing_fields(self):
        required = {
            "brand.name": self.brand.name,
            "brand.category": self.brand.category,
            "audience.description": self.audience.description,
            "budget.total_thb": self.budget.total_thb,
            "timing.start_date": self.timing.start_date,
            "timing.end_date": self.timing.end_date,
            "offer.mechanics": self.offer.mechanics or None,
            "channels": self.channels or None,
        }
        for index, mechanic in enumerate(self.offer.mechanics):
            if mechanic.type in {"percent_off", "fixed_off", "voucher", "coins_cashback"}:
                required[f"offer.mechanics.{index}.value"] = mechanic.value
        existing = {question.field: question for question in self.open_questions}
        for path, value in required.items():
            if value is None:
                if path in existing:
                    existing[path].blocking = True
                else:
                    self.open_questions.append(OpenQuestion(
                        field=path, question=f"Please provide {path}.", blocking=True
                    ))
        if any(question.blocking for question in self.open_questions):
            if self.status == "ready":
                raise ValueError("CampaignSpec cannot be ready while blocking questions remain")
            self.status = "needs_info"
