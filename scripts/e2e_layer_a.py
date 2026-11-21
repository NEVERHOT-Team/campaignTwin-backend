"""Run the offline CampaignTwin Layer A acceptance checks from a Git checkout."""

import json
import math
import platform
import subprocess
import sys
import warnings
from collections import Counter
from dataclasses import asdict
from datetime import date, datetime, timezone
from functools import cached_property, wraps
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import numpy as np
import pandas as pd
from pydantic_core import to_json

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.strategist import propose  # noqa: E402
from contracts.campaign_spec import CampaignSpec  # noqa: E402
from contracts.models import ContextPack, SimulationPlan  # noqa: E402
from data import loaders  # noqa: E402
from data.rival import get_rival_scenarios  # noqa: E402
from engine import arena, backtest, evaluator, shopper  # noqa: E402
from mocks import load_mock  # noqa: E402

SCENARIOS = ("S1", "S2", "S3")
AGENTS = 5_000
POPULATION_SEED = 0
SEEDS = range(10)
AA_SEEDS = range(10, 20)


class Audit:
    """Share the exact same inputs, primary result and observed replay across checks."""

    outcome = None
    replay = None

    def __init__(self):
        self.baselines = {}

    @cached_property
    def spec(self):
        return CampaignSpec.model_validate(loaders.load_demo_brief()["user_plan_spec"])

    @cached_property
    def context(self):
        return ContextPack.model_validate(load_mock("context_pack"))

    @cached_property
    def strategies(self):
        return propose(self.spec, self.context)

    @cached_property
    def plan(self):
        return SimulationPlan(
            spec=self.spec, context=self.context,
            strategies=self.strategies, scenarios=list(SCENARIOS), seeds=len(SEEDS),
            n_agents=AGENTS, weeks=6,
        )

    def result(self):
        if self.outcome is None:
            raise RuntimeError("A5 did not produce an arena result")
        return self.outcome

    def scenarios_for(self, strategy_id):
        rows = {r.scenario: r for r in self.result().results if r.strategy_id == strategy_id}
        if set(rows) != set(SCENARIOS):
            raise RuntimeError(f"Missing scenario results for {strategy_id}")
        return rows


def check(check_id, owner):
    """Keep all checks reportable, including those with missing prerequisites."""
    def decorate(fn):
        @wraps(fn)
        def run(audit):
            try:
                passed, evidence = fn(audit)
                status = "PASS" if passed else "FAIL"
            except (ValueError, AssertionError) as exc:
                status, evidence = "FAIL", {"detail": f"{type(exc).__name__}: {exc}"}
            except Exception as exc:
                status, evidence = "BLOCKED", {"detail": f"{type(exc).__name__}: {exc}"}
            return {"id": check_id, "status": status, "evidence": evidence, "owner": owner}
        return run
    return decorate


@check("A1", "Gan")
def check_a1(a):
    s = a.spec
    offers = [(m.type, m.percentage) for m in s.offer.mechanics]
    detail = f"{s.status}, {offers}, {s.timing.event}, {', '.join(s.channels)}"
    return (s.status == "ready" and offers == [("percent_off", 40)]
            and s.timing.event == "11.11" and set(s.channels) == {"shopee", "line"}), {
        "detail": detail, "status": s.status, "offers": offers, "event": s.timing.event,
        "channels": s.channels, "budget_thb": s.budget.total_thb,
    }


@check("A2", "Mic")
def check_a2(a):
    record = loaders.load_shopper_params()
    params = shopper.default_params()
    active = {name: getattr(params, name, None) for name in record}
    labels = {name: entry["label"] for name, entry in record.items()}
    counts = dict(Counter(labels.values()))
    scale = shopper.build_population(AGENTS, seed=POPULATION_SEED).scale_label
    passed = (counts.get("calibrated", 0) > 0 and params.label == scale == "calibrated"
              and all(active[k] == entry["value"] for k, entry in record.items()))
    return passed, {"detail": f"{counts}, population label {scale}", "labels": labels,
                    "active_values": active, "file": record, "engine_label": params.label,
                    "population_label": scale}


def knob_violations(plan, knobs):
    problems = []
    timing = plan.timing
    if timing["event"] not in knobs["calendar"]:
        problems.append(f"calendar {timing['event']}")
    aliases = {"percent_off": {"percentage_discount"}, "voucher": {"seller_voucher",
                                                                   "loyalty_voucher"}}
    mechanics = {plan.mechanic} | aliases.get(plan.mechanic, set())
    if not mechanics.intersection(knobs["mechanics"]):
        problems.append(f"mechanic {plan.mechanic}")
    days = timing.get("flash_days") or (
        date.fromisoformat(timing["end_date"]) - date.fromisoformat(timing["start_date"])
    ).days
    for name, value in (("discount_pct", plan.discount_pct), ("duration_days", days),
                        ("cap_per_user", plan.cap_per_user)):
        if value is not None and not knobs[name]["min"] <= value <= knobs[name]["max"]:
            problems.append(f"{name} {value}")
    if not set(plan.channel_mix) <= {c.casefold() for c in knobs["channels"]}:
        problems.append(f"channels {list(plan.channel_mix)}")
    return problems


@check("A3", "Mic")
def check_a3(a):
    knobs = loaders.load_knobs()
    plans = a.strategies
    ids = [p.id for p in plans]
    violations = {p.id: issues for p in plans if not p.is_baseline
                  if (issues := knob_violations(p, knobs))}
    discounts = {p.id: p.discount_pct for p in plans}
    shape = (len(set(ids)) == len(ids) == 8 and {"baseline", "user-plan"} <= set(ids)
             and sum(i.startswith("g0-") for i in ids) == 6)
    return shape and max(discounts.values()) <= 50 and not violations, {
        "detail": f"{len(ids)} plans, max discount {max(discounts.values()):g}%, "
                  f"{len(violations)} plans outside published knobs",
        "discounts_pct": discounts, "knobs": knobs, "violations": violations,
    }


@check("A4", "Gan")
def check_a4(a):
    expected = {(r, s) for r in ("764", "595", "69") for s in SCENARIOS}
    evidence = {}
    valid = True
    for p in a.strategies:
        rows = get_rival_scenarios(a.spec, p)
        evidence[p.id] = [r.model_dump() for r in rows]
        valid &= len(rows) == 9 and {(r.rival, r.scenario) for r in rows} == expected
        for row in rows:
            valid &= 0 <= row.depth <= 100
            if row.scenario == "S1":
                target = 1.0 if p.is_baseline else 0.72
                valid &= row.depth == 0 and abs(row.likelihood - target) <= 0.01
            if p.is_baseline:
                valid &= row.reaction == "none" and row.depth == 0 and row.likelihood == 1
            if row.scenario == "S3":
                valid &= row.depth <= p.discount_pct + 10
    return valid, {"detail": f"{sum(map(len, evidence.values()))} rival/scenario rows",
                   "by_plan": evidence}


@check("A5", "Mic")
def check_a5(a):
    plan = a.plan
    start = perf_counter()
    out = a.outcome = arena.run(plan, base_seed=POPULATION_SEED)
    runtime = perf_counter() - start

    # The first run supplies the timing. Observe the original simulator on the replay so
    # the same weekly rows can be checked here and the actual baselines reused in A7.
    simulate = shopper.simulate
    units = []
    sizes = set()
    invalid_weeks = 0
    calls = 0

    def observe(population, strategy, weeks, campaign_weeks, seed, *args, **kwargs):
        nonlocal calls, invalid_weeks
        rows = simulate(population, strategy, weeks, campaign_weeks, seed, *args, **kwargs)
        calls += 1
        sizes.add(population.size)
        if strategy.is_baseline:
            a.baselines[seed] = rows
        for row in rows:
            fields = asdict(row)
            extra = fields.pop("extra")
            finite = all(math.isfinite(x) for x in (*fields.values(), *extra.values()))
            overlap = extra.get("discounted_units_would_buy")
            valid = finite and 0 <= row.discounted_units_us <= row.units_us
            if overlap is not None:
                valid &= 0 <= overlap <= row.discounted_units_us
            invalid_weeks += int(not valid)
            units.append(row.units_us)
        return rows

    with patch.object(shopper, "simulate", observe):
        a.replay = arena.run(plan, base_seed=POPULATION_SEED)
    expected = {(p.id, s) for p in plan.strategies if not p.is_baseline for s in SCENARIOS}
    actual = {(r.strategy_id, r.scenario) for r in out.results}
    json.dumps([r.model_dump() for r in out.results] + out.summary, allow_nan=False)
    passed = (runtime < 60 and actual == expected and len(out.results) == len(expected) == 21
              and out.seeds == list(SEEDS) and set(a.baselines) == set(SEEDS) and sizes == {AGENTS}
              and (out.campaign_weeks, out.weeks) == (4, 6)
              and out.appeal_source == "appeal_fallback"
              and calls == 220 and len(units) == 1320 and invalid_weeks == 0)
    return passed, {"detail": f"{len(out.results)} rows, {calls} simulations, {len(units)} weeks",
                    "arena_seconds": runtime, "calls": calls, "weeks_checked": len(units),
                    "invalid_weeks": invalid_weeks, "unit_range": [min(units), max(units)],
                    "population_sizes": sorted(sizes), "seeds": out.seeds,
                    "campaign_weeks": out.campaign_weeks, "total_weeks": out.weeks,
                    "appeal_source": out.appeal_source}


@check("A6", "Mic")
def check_a6(a):
    original = a.result()
    if a.replay is None:
        raise RuntimeError("A5 did not complete the identical-seed replay")
    same = asdict(original) == asdict(a.replay)
    return same, {"detail": f"Complete arena results identical = {same}", "identical": same}


@check("A7", "Mic")
def check_a7(a):
    out = a.result()
    if set(a.baselines) != set(SEEDS):
        raise RuntimeError("A5 did not capture all ten baseline seeds")
    pop = shopper.build_population(AGENTS, seed=POPULATION_SEED)
    baseline = next(p for p in a.strategies if p.is_baseline)
    other = {slot: shopper.simulate(pop, baseline, out.weeks, out.campaign_weeks, seed)
             for slot, seed in zip(SEEDS, AA_SEEDS, strict=True)}
    with warnings.catch_warnings(record=True) as messages:
        warnings.simplefilter("always")
        lift = evaluator.aa_lift(a.baselines, other)
    totals = [sum(w.revenue_us for rows in runs.values() for w in rows)
              for runs in (a.baselines, other)]
    return abs(lift) <= 2, {"detail": f"Baseline sales lift {lift:+.4f}% (limit +/-2%)",
                            "seeds_a": list(SEEDS), "seeds_b": list(AA_SEEDS),
                            "sales_thb": totals, "lift_pct": lift,
                            "warnings": [str(m.message) for m in messages]}


@check("A8", "Gan")
def check_a8(a):
    rows = a.scenarios_for("user-plan")
    profits = {s: rows[s].kpi.incr_profit.p50 for s in ("S2", "S3")}
    depths = {s: {r.rival: r.depth for r in a.result().rivals["user-plan"][s]}
              for s in profits}
    deeper = max(depths["S3"].values()) > max(depths["S2"].values())
    return profits["S2"] != profits["S3"] and deeper, {
        "detail": f"User S2 {profits['S2']:,.2f}, S3 {profits['S3']:,.2f} THB",
        "profit_p50_thb": profits, "depth_pct": depths,
    }


@check("A9", "Mic")
def check_a9(a):
    rows = a.scenarios_for("user-plan")
    user = {s: rows[s].kpi.incr_profit.p50 for s in ("S1", "S3")}
    rank = {row["strategy_id"]: row["rank"] for row in a.result().summary}
    ai = {r.strategy_id: r.kpi.incr_profit.p50 for r in a.result().results
          if r.scenario == "S1" and r.strategy_id.startswith("g0-")}
    above = [p for p, v in ai.items() if v > 0 and rank[p] < rank["user-plan"]]
    discount = next(p.discount_pct for p in a.strategies if p.id == "user-plan")
    return discount == 40 and max(user.values()) < 0 and bool(above), {
        "detail": f"{len(above)}/{len(ai)} AI plans profitable in S1 and above user",
        "user_profit_p50_thb": user, "ai_s1_profit_p50_thb": ai, "rank": rank,
        "profitable_above_user": above, "user_discount_pct": discount,
    }


@check("A10", "Gan")
def check_a10(a):
    rows = a.result().results
    missing = [f"{r.strategy_id}/{r.scenario}" for r in rows
               if r.kpi.leakage_label != "data" or r.kpi.leakage is None
               or not 0 <= r.kpi.leakage.p50 <= 1]
    return len(rows) == 21 and not missing, {
        "detail": f"{len(rows)-len(missing)}/{len(rows)} measured leakage rows",
        "measured": len(rows) - len(missing), "missing_or_invalid": missing,
    }


@check("A11", "Gan")
def check_a11(a):
    user = {s: r.breakpoint.model_dump() for s, r in a.scenarios_for("user-plan").items()}
    winner = next(r["strategy_id"] for r in a.result().summary
                  if r["strategy_id"].startswith("g0-"))
    peaks = {s: r.peak_week for s, r in a.scenarios_for(winner).items()}
    valid = all(b["cause"] == "never_profitable" or (b["week"] and b["cause"])
                for b in user.values())
    return valid and all(w is not None for w in peaks.values()), {
        "detail": f"{winner} peak weeks {list(peaks.values())}", "winner": winner,
        "user_breakpoints": user, "winner_peak_weeks": peaks,
    }


@check("A12", "Mic")
def check_a12(a):
    result = backtest.run_backtest(seeds=range(5), n_agents=AGENTS)
    evaluation = result.evaluation
    passed = (result.campaign_id == "18" and evaluation["observed_range"] == [1.71, 6.24]
              and evaluation["naive"] == 19.02 and math.isfinite(evaluation["simulated"])
              and evaluation["label"] in {"Calibrated", "Partially calibrated", "Uncalibrated"})
    return passed, {"detail": f"Campaign {result.campaign_id}, {result.simulated_lift_pct:.4f}% "
                              f"({evaluation['label']})", **result.to_dict()}


@check("A13", "Gan")
def check_a13(a):
    market = loaders.load_market()
    price, cost = market["ref_price"], market["unit_cost"]
    fees = (market["platform_fee_pct"] + market["payment_fee_pct"]) / 100
    net = price * (1 - 0.40) * (1 - fees) - cost
    return round(net, 2) == -18.93, {"detail": f"{net:.3f} THB per unit before shipping",
                                      "price_thb": price, "unit_cost_thb": cost,
                                      "fee_fraction": fees, "discount_pct": 40,
                                      "contribution_thb": net}


CHECKS = (check_a1, check_a2, check_a3, check_a4, check_a5, check_a6, check_a7,
          check_a8, check_a9, check_a10, check_a11, check_a12, check_a13)


def timestamp():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def git_hash(ref):
    cmd = subprocess.run(["git", "rev-parse", "--verify", ref], cwd=ROOT,
                         capture_output=True, text=True)
    return cmd.stdout.strip() if cmd.returncode == 0 else None


def arena_record(out):
    return {
        "seeds": out.seeds, "campaign_weeks": out.campaign_weeks, "weeks": out.weeks,
        "appeal_source": out.appeal_source, "ranking": out.summary,
        "results": [r.model_dump() for r in out.results],
        "rivals": {pid: {s: [r.model_dump() for r in rows] for s, rows in scenarios.items()}
                   for pid, scenarios in out.rivals.items()},
    }


def plan_table(out):
    yield ("Rank Plan          Sc   Profit P10   Profit P50   Profit P90"
           "     iROI  Leakage  Breakpoint / peak")
    if out is None:
        yield "Arena results unavailable"
        return
    ranks = {r["strategy_id"]: r["rank"] for r in out.summary}
    for row in sorted(out.results, key=lambda r: (ranks.get(r.strategy_id, 999), r.scenario)):
        kpi = row.kpi
        p = kpi.incr_profit
        leak = f"{kpi.leakage.p50:.2%}" if kpi.leakage is not None else "n/a"
        bp = row.breakpoint
        cause = f"wk {bp.week} {bp.cause}" if bp.week else bp.cause or "no crossing"
        yield (f"{ranks.get(row.strategy_id, '-'):>4} {row.strategy_id:13} {row.scenario:2} "
               f"{p.p10:12,.0f} {p.p50:12,.0f} {p.p90:12,.0f} {kpi.iroi.p50:8.2%} "
               f"{leak:>7}  {cause} / {row.peak_week or '-'}")
    yield ("   - baseline      S1-S3          0            0            0"
           "      n/a      n/a  paired comparator")


def main():
    audit = Audit()
    start = perf_counter()
    metadata = {
        "started_utc": timestamp(), "head": git_hash("HEAD"), "origin_dev": git_hash("origin/dev"),
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
        "platform": f"{platform.system()} {platform.machine()}", "population_seed": POPULATION_SEED,
        "agents": AGENTS, "sim_seeds": list(SEEDS), "aa_b_seeds": list(AA_SEEDS),
        "scenarios": SCENARIOS, "weeks": [4, 2], "backtest_seeds": list(range(5)),
    }
    log = [
        f"Layer A | {metadata['started_utc']}",
        f"HEAD {metadata['head']} | origin/dev {metadata['origin_dev']}",
        f"Python {metadata['python']} | NumPy {metadata['numpy']} | pandas {metadata['pandas']} "
        f"| {metadata['platform']}",
        "Demo brief | gen-0 templates | fixture context_pack | offline appeal_fallback",
        "5,000 agents | population seed 0 | S1-S3 | seeds 0-9 | 4 campaign + 2 post weeks",
        "A/A seeds 0-9 vs 10-19 | campaign 18 back-test seeds 0-4",
        "",
        "Started (UTC)                 ID  Status   Owner  Seconds  Evidence",
    ]
    checks, steps = [], {}
    for fn in CHECKS:
        started = timestamp()
        step_start = perf_counter()
        result = fn(audit)
        seconds = perf_counter() - step_start
        checks.append(result)
        key = result["id"]
        steps[key] = {"started_utc": started, "seconds": seconds}
        detail = " ".join(result["evidence"]["detail"].splitlines())
        log.append(f"{started} {key:3} {result['status']:8} {result['owner']:5} "
                   f"{seconds:7.3f}  {detail}")
    counts = Counter(r["status"] for r in checks)
    log.append(f"{counts['PASS']} PASS | {counts['FAIL']} FAIL | {counts['BLOCKED']} BLOCKED")
    log.extend(["", "Incremental profit in THB | iROI and leakage at P50"])
    log.extend(plan_table(audit.outcome))
    metadata.update(finished_utc=timestamp(), elapsed_seconds=perf_counter() - start)
    log.append(f"Finished {metadata['finished_utc']} | {metadata['elapsed_seconds']:.3f} seconds")
    report = {
        "metadata": metadata, "steps": steps, "checks": checks,
        "counts": {s: counts[s] for s in ("PASS", "FAIL", "BLOCKED")},
        "simulation_plan": audit.plan.model_dump(mode="json") if "plan" in vars(audit) else None,
        "arena": arena_record(audit.outcome) if audit.outcome is not None else None,
    }
    dest = ROOT / "runs"
    dest.mkdir(exist_ok=True)
    # Keep a failed finite-value check readable as standard JSON, including its error evidence.
    (dest / "e2e_layer_a.json").write_bytes(to_json(report, indent=2, inf_nan_mode="null") + b"\n")
    rendered = "\n".join(log) + "\n"
    (dest / "e2e_layer_a.log").write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return int(bool(counts["FAIL"] or counts["BLOCKED"]))


if __name__ == "__main__":
    raise SystemExit(main())
