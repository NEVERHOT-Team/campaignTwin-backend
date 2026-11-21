"""Contingency rules: our side of the weekly closed loop.

The strategist writes rules as strings (agents/strategist.py RULE_PATTERNS). This module
parses them into actions and applies them at the start of each week, using only what
the plan could observe after the previous week (one-week information lag). No LLM calls.

Supported rules (units are explicit):
- ``if rival_undercuts then shift_budget:<segment>``: when the rival's posted discount in
  the previous week was deeper than ours, concentrate the offer (and its media reach) on
  one segment. Fires once.
- ``if share_drop>X then cap_per_user:N``: when our share of category buyers fell by more
  than X percentage points from campaign week 1, cap redemptions per shopper at N.
  Fires once.
- ``if week>=W and loss then end_early``: from week W, when incremental gross profit to
  date against the paired no-campaign twin (minus media spent so far) is negative, stop
  the offer. Media is treated as already paid. Fires once and needs the paired baseline.

Our discount depth never changes here, so the evaluator's cost accounting stays valid.
"""

import re
from dataclasses import dataclass, field

SHIFT_BUDGET = re.compile(r"^if rival_undercuts then shift_budget:([a-z0-9_]+)$")
CAP_PER_USER = re.compile(r"^if share_drop>(\d+(?:\.\d+)?) then cap_per_user:(\d+)$")
END_EARLY = re.compile(r"^if week>=(\d+) and loss then end_early$")


@dataclass(frozen=True)
class Rule:
    text: str
    kind: str  # shift_budget | cap_per_user | end_early
    segment: str | None = None
    threshold: float = 0.0  # share_drop in percentage points, or the first week
    cap: int | None = None


@dataclass
class PlanState:
    """What our plan looks like this week. Rules change it; the offer is built from it."""

    target_segments: tuple[str, ...]
    cap_per_user: int | None
    reach: float
    stopped: bool = False
    fired: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class Observation:
    """What both sides can see at the start of a week (about the previous week)."""

    week: int  # the week just finished; 0 before the campaign
    our_discount_pct: float = 0.0  # our posted discount in that week (0 if not offered)
    rival_discount_pct: float = 0.0  # rival posted discount in that week
    our_share_pp: float | None = None  # our share of category buyers, percentage points
    first_share_pp: float | None = None  # our share in campaign week 1
    incr_profit_to_date: float | None = None  # vs paired baseline, minus media spent


@dataclass(frozen=True)
class Event:
    week: int
    side: str  # "us" | "rival"
    action: str
    detail: str

    def as_dict(self) -> dict[str, object]:
        return {"week": self.week, "side": self.side, "action": self.action,
                "detail": self.detail}


def parse_rule(text: str) -> Rule | None:
    """A supported rule, or None (unknown rules are ignored, never guessed)."""
    text = text.strip()
    if match := SHIFT_BUDGET.match(text):
        return Rule(text=text, kind="shift_budget", segment=match.group(1))
    if match := CAP_PER_USER.match(text):
        return Rule(text=text, kind="cap_per_user", threshold=float(match.group(1)),
                    cap=int(match.group(2)))
    if match := END_EARLY.match(text):
        return Rule(text=text, kind="end_early", threshold=float(match.group(1)))
    return None


def parse_rules(texts: list[str] | None) -> list[Rule]:
    return [rule for text in texts or [] if (rule := parse_rule(text)) is not None]


def apply_rules(rules: list[Rule], plan: PlanState, obs: Observation, week: int) -> list[Event]:
    """Apply every rule whose trigger holds on `obs`. Mutates `plan`; returns what fired."""
    events = []
    for rule in rules:
        if rule.text in plan.fired or plan.stopped:
            continue
        if rule.kind == "shift_budget":
            if obs.rival_discount_pct > obs.our_discount_pct and rule.segment:
                plan.target_segments = (rule.segment,)
                events.append(Event(week, "us", "shift_budget",
                                    f"rival {obs.rival_discount_pct:g}% > ours "
                                    f"{obs.our_discount_pct:g}%: focus on {rule.segment}"))
                plan.fired.add(rule.text)
        elif rule.kind == "cap_per_user":
            if (obs.our_share_pp is not None and obs.first_share_pp is not None
                    and obs.first_share_pp - obs.our_share_pp > rule.threshold):
                plan.cap_per_user = rule.cap
                events.append(Event(week, "us", "cap_per_user",
                                    f"share {obs.first_share_pp:.1f}pp -> "
                                    f"{obs.our_share_pp:.1f}pp: cap {rule.cap} per shopper"))
                plan.fired.add(rule.text)
        elif rule.kind == "end_early":
            if (week >= rule.threshold and obs.incr_profit_to_date is not None
                    and obs.incr_profit_to_date < 0):
                plan.stopped = True
                events.append(Event(week, "us", "end_early",
                                    f"incremental profit to date "
                                    f"{obs.incr_profit_to_date:,.0f} THB < 0: stop the offer"))
                plan.fired.add(rule.text)
    return events
