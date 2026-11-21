"""Reactive rivals: the other side of the weekly closed loop.

data/rival.py and agents/rival_policy.py produce RivalScenario rows before the run (no LLM
calls inside the weekly loop). Here each row is read as a policy, not a calendar:

- reaction: none | match | undercut | non_price
- depth: the deepest discount (percentage points) the evidence supports, a hard cap
- delay_weeks: weeks between noticing our campaign and reacting
- duration_weeks: how long one reaction lasts

A rival notices our campaign when our posted discount this week is at least
TRIGGER_DISCOUNT_PCT (prices are public the same week), or when our share of category
buyers last week was TRIGGER_SHARE_GAIN_PP above our share in the paired no-campaign run
that week (sales are seen with a one-week lag; without a paired run only price triggers).
Once noticed at week t, it reacts in weeks t + delay .. t + delay + duration - 1:

- match sets its discount to ours; undercut goes UNDERCUT_STEP_PP deeper than ours;
- both are capped at `depth` and never go back down inside the window (a price war
  ratchets), so if we deepen, the rival follows up to its cap;
- S4 (red team) rows keep the designed depth: the counter was chosen before the run;
- non_price runs a non-price promotion in the window; none never reacts.

The baseline gets no rows, so it never faces a reaction (rivals react to our campaign).
The thresholds below are engine defaults (source label: default) until calibrated.
"""

from dataclasses import dataclass

from contracts.models import RivalScenario
from engine.rules import Event

TRIGGER_DISCOUNT_PCT = 5.0
TRIGGER_SHARE_GAIN_PP = 5.0
UNDERCUT_STEP_PP = 5.0
SOURCE_LABEL = "default"


@dataclass
class _RivalState:
    row: RivalScenario
    noticed_week: int | None = None
    depth: float = 0.0
    announced: bool = False

    @property
    def window(self) -> tuple[int, int] | None:
        if self.noticed_week is None:
            return None
        start = self.noticed_week + max(int(self.row.delay_weeks), 0)
        return start, start + max(int(self.row.duration_weeks), 1) - 1


class RivalBook:
    """All rivals in one scenario for one run. Call `react` once per week, in order."""

    def __init__(self, rows: list[RivalScenario] | None):
        self.rivals = [_RivalState(row) for row in rows or [] if row.reaction != "none"]

    @staticmethod
    def _notices(our_posted_pct: float, share_gain_pp: float | None) -> bool:
        if our_posted_pct >= TRIGGER_DISCOUNT_PCT:
            return True
        return share_gain_pp is not None and share_gain_pp >= TRIGGER_SHARE_GAIN_PP

    def react(
        self,
        week: int,
        our_posted_pct: float,
        share_gain_pp: float | None = None,
        our_active: bool = True,
    ) -> tuple[float, bool, list[Event]]:
        """(rival discount %, non-price promo on?, events) for this week.

        share_gain_pp: last week's share of category buyers minus the paired baseline's.
        our_active: whether our offer runs this week. A rival does not open or deepen a
        price war against an offer that has ended (we stopped early or the campaign is
        over); a discount it already launched keeps running to the end of its window.
        Rivals that do the same thing in the same week are reported as one event.
        """
        raw: list[tuple[str, str, str]] = []  # (action, rival, what) before merging
        depth, non_price = 0.0, False
        noticed = our_active and self._notices(our_posted_pct, share_gain_pp)
        for rival in self.rivals:
            row = rival.row
            if rival.noticed_week is None and noticed:
                rival.noticed_week = week
                raw.append(("noticed", row.rival,
                            f"see our {our_posted_pct:g}% offer; {row.reaction} after "
                            f"{int(row.delay_weeks)} week(s)"))
            window = rival.window
            if window is None or not window[0] <= week <= window[1]:
                continue
            if row.reaction == "non_price":
                non_price = True
                if not rival.announced:
                    raw.append(("non_price", row.rival, "run a non-price promotion"))
                    rival.announced = True
                continue
            if not our_active:
                target = 0.0
            elif row.scenario == "S4":
                target = float(row.depth)
            elif row.reaction == "match":
                target = our_posted_pct
            else:  # undercut
                target = our_posted_pct + UNDERCUT_STEP_PP
            new_depth = max(rival.depth, min(max(target, 0.0), float(row.depth)))
            if new_depth > rival.depth:
                raw.append((row.reaction, row.rival,
                            f"{row.reaction} at {new_depth:g}% (cap {float(row.depth):g}%)"))
            rival.depth = new_depth
            depth = max(depth, rival.depth)
        return depth, non_price, _merge(week, raw)


def _merge(week: int, raw: list[tuple[str, str, str]]) -> list[Event]:
    """One event per (action, what), naming every rival that did it, in first-seen order."""
    grouped: dict[tuple[str, str], list[str]] = {}
    for action, name, what in raw:
        grouped.setdefault((action, what), []).append(name)
    return [
        Event(week, "rival", action, f"{', '.join(names)} {what}")
        for (action, what), names in grouped.items()
    ]
