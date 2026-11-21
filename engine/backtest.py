"""Back-test: replay Complete Journey campaign 18 through the shopper model.

The historical campaign (data/processed/backtest_campaign.json) is simulated against a
paired no-campaign baseline on the same seeds, and the simulated lift in *category*
sales (us + rival + other) over the campaign weeks is compared with the observed lift
range by data.backtest.evaluate_backtest().

The 22.2% discount is split into two parts, as agreed with Gan:
- shelf discount (shelf_share of the depth): every shopper sees it, reach 1.0;
- coupon (the rest): only the treated share of households holds it.
The observed redemption rate is an outcome, so it is never used as reach. The simulated
redemption rate is reported next to it as an extra check.

Run `python -m engine.backtest` for a JSON report. `sweep()` evaluates several
ShopperParams overrides for calibration (values go to shopper_params.json).
"""

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np

from data.backtest import evaluate_backtest
from data.loaders import load_backtest
from engine import shopper

DEFAULT_SEEDS = (0, 1, 2, 3, 4)
DEFAULT_AGENTS = 5000
PANEL_HOUSEHOLDS = 2500  # Complete Journey panel size, used only if treated_share is absent


@dataclass(frozen=True)
class BacktestResult:
    campaign_id: str
    weeks: int
    discount_pct: float
    shelf_share: float
    treated_share: float
    simulated_lift_pct: float
    lift_by_seed: list[float]
    simulated_redemption_rate: float | None
    observed_redemption_rate: float | None
    evaluation: dict[str, Any]
    labels: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def campaign_inputs(record: dict[str, Any]) -> tuple[float, float, float, int, dict[str, str]]:
    """(discount %, shelf share, treated share, weeks, labels) for the campaign record."""
    labels: dict[str, str] = {}
    discount = 100 * float(record["discount_depth"])
    weeks = int(record["end_week"]) - int(record["start_week"]) + 1
    if "shelf_share" in record:
        shelf, labels["shelf_share"] = float(record["shelf_share"]), "data"
    else:
        shelf, labels["shelf_share"] = 1.0, "default"
    if "treated_share" in record:
        treated, labels["treated_share"] = float(record["treated_share"]), "data"
    else:
        treated = float(record.get("treated_households", 0)) / PANEL_HOUSEHOLDS
        labels["treated_share"] = "default"
    return discount, float(np.clip(shelf, 0, 1)), float(np.clip(treated, 0, 1)), weeks, labels


def _category_sales(stats: Sequence[shopper.WeekStats]) -> float:
    return float(sum(s.revenue_us + s.revenue_rival + s.revenue_other for s in stats))


def _run(
    population: shopper.Population,
    offer: shopper.WeekOffer,
    weeks: int,
    seed: int,
    params: shopper.ShopperParams,
) -> list[shopper.WeekStats]:
    state = shopper.initial_state(population, seed)
    out = []
    for week in range(1, weeks + 1):
        stats, state = shopper.step(population, state, offer, seed, week, params)
        out.append(stats)
    return out


def run_backtest(
    seeds: Sequence[int] = DEFAULT_SEEDS,
    n_agents: int = DEFAULT_AGENTS,
    params: shopper.ShopperParams | None = None,
    record: dict[str, Any] | None = None,
) -> BacktestResult:
    record = record or load_backtest()
    params = params or shopper.default_params()
    # The whole panel is measured, so no holdout is kept out of the offer here.
    params = replace(params, holdout_share=0.0)
    discount, shelf, treated, weeks, labels = campaign_inputs(record)
    population = shopper.build_population(n_agents, seed=0, params=params)
    targets = tuple(record.get("target_segments") or ())
    offer = shopper.WeekOffer(
        active=True,
        our_discount_pct=discount * shelf,
        target_segments=targets,
        coupon_discount_pct=discount * (1 - shelf),
        coupon_reach=treated,
    )
    lifts = []
    redemptions = holders = 0.0
    for seed in seeds:
        base = _run(population, shopper.WeekOffer(), weeks, seed, params)
        plan = _run(population, offer, weeks, seed, params)
        base_sales = _category_sales(base)
        lifts.append(100 * (_category_sales(plan) - base_sales) / base_sales if base_sales else 0.0)
        redemptions += sum(s.coupon_redemptions for s in plan)
        holders += max((s.coupon_holders for s in plan), default=0)
    lift = float(np.mean(lifts))
    observed = (record.get("observed_redemption_rate")
                or (record.get("actual") or {}).get("redemption_rate"))
    return BacktestResult(
        campaign_id=str(record["campaign_id"]),
        weeks=weeks,
        discount_pct=round(discount, 4),
        shelf_share=shelf,
        treated_share=treated,
        simulated_lift_pct=round(lift, 4),
        lift_by_seed=[round(v, 4) for v in lifts],
        simulated_redemption_rate=round(redemptions / holders, 6) if holders else None,
        observed_redemption_rate=float(observed) if observed is not None else None,
        evaluation=evaluate_backtest(lift),
        labels={**labels, "params": params.label},
    )


def sweep(
    overrides: Sequence[dict[str, float]],
    seeds: Sequence[int] = DEFAULT_SEEDS,
    n_agents: int = DEFAULT_AGENTS,
) -> list[dict[str, Any]]:
    """Back-test each ShopperParams override (e.g. {"beta_price": 3.0}) for calibration."""
    base = shopper.default_params()
    rows = []
    for change in overrides:
        result = run_backtest(seeds, n_agents, replace(base, **change))
        rows.append({
            "params": change,
            "simulated_lift_pct": result.simulated_lift_pct,
            "label": result.evaluation["label"],
            "simulated_redemption_rate": result.simulated_redemption_rate,
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agents", type=int, default=DEFAULT_AGENTS)
    parser.add_argument("--seeds", type=int, default=len(DEFAULT_SEEDS))
    args = parser.parse_args()
    result = run_backtest(range(args.seeds), args.agents)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
