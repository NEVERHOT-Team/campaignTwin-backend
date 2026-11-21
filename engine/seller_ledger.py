"""Paired one-pack orders and seller settlement for approved voucher rehearsals."""

from dataclasses import dataclass
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from fractions import Fraction

import numpy as np


def subsidy_per_pack(offer, price):
    if offer is None:
        return Decimal(0)
    value = Decimal(str(offer["value"]))
    if offer["unit"] == "pct":
        value = value * price / 100
    return value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def valid_discount_amount(offer, price):
    """Amount bounds for a single-pack intervention, including satang settlement."""
    try:
        amount = Decimal(str(offer.get("value")))
        price = Decimal(str(price))
    except InvalidOperation:
        return False
    if not amount.is_finite() or not price.is_finite() or price <= 0:
        return False
    if (offer.get("kind"), offer.get("unit")) == ("percent_off", "pct"):
        return Decimal("0.01") <= amount <= 30 and subsidy_per_pack(offer, price) > 0
    return (
        (offer.get("kind"), offer.get("unit")) == ("voucher", "THB")
        and Decimal("0.02") <= amount <= price * Decimal("0.30")
        and amount.normalize().as_tuple().exponent >= -2
    )


@dataclass(frozen=True)
class Settlement:
    price: Decimal
    cogs: Decimal
    commission: Decimal
    order_fee: Decimal
    fulfillment: Decimal
    voucher_fee: Decimal
    media: Decimal

    @classmethod
    def from_plan(cls, plan):
        costs = plan["costs"]

        def amount(key):
            return Decimal(str(costs[key]["value"]))

        return cls(
            amount("regular_price"),
            amount("cogs"),
            amount("channel_commission") / 100,
            amount("payment_fee"),
            amount("seller_fulfillment"),
            amount("voucher_service_fee"),
            Decimal(0) if plan["role"] == "baseline" else amount("paid_media"),
        )

    def ledger(self, units, redemptions, discount, media):
        gross = units * self.price
        subsidy = redemptions * discount
        receipts = gross - subsidy
        commission = receipts * self.commission
        order_fees = units * self.order_fee
        fulfillment = units * self.fulfillment
        voucher_fees = redemptions * self.voucher_fee
        cogs = units * self.cogs
        contribution = (
            receipts - cogs - commission - order_fees - fulfillment - voucher_fees - media
        )
        return dict(
            orders=units,
            units=units,
            redemptions=redemptions,
            gross_sales=gross,
            seller_subsidy=subsidy,
            net_receipts=receipts,
            cogs=cogs,
            channel_commission=commission,
            order_fees=order_fees,
            fulfillment=fulfillment,
            voucher_fees=voucher_fees,
            media_spend=media,
            campaign_spend=subsidy + media,
            contribution_profit=contribution,
        )


def metric(name, unit, values, reason="undefined_denominator"):
    valid = values is not None and all(v is not None for v in values)
    value = None
    if valid:
        value = dict(zip(("p10", "p50", "p90"), np.percentile(values, [10, 50, 90]).tolist()))
    return {
        "metric": name,
        "unit": unit,
        "status": "available" if valid else "unavailable",
        "value": value,
        "unavailable_reason": None if valid else reason,
    }


def _ratio(numerator, denominator):
    return float(numerator / denominator) if denominator > 0 else None


def simulate(plan, scenario, daily, active_days, days):
    terms = plan["evaluation"]
    costs = Settlement.from_plan(plan)
    offer = plan["offer"]
    discount = subsidy_per_pack(offer["discount"] if offer else None, costs.price)
    budget = Fraction(str(plan["subsidy_budget"]))
    limit = budget // Fraction(discount) if discount else 0
    settings = terms["behavior"]
    ordinary = daily
    if scenario == "S2":
        ordinary = daily * (1 - settings["rival_demand_reduction_pct"] / 100)
    boosted = np.minimum(
        1, ordinary * (1 + float(discount / costs.price) * settings["discount_response"])
    )
    starts = np.cumsum([0] + [s["households"] for s in plan["segments"]][:-1])
    start_date = date.fromisoformat(plan["campaign_window"]["start_date"])
    # Split partial weeks at the campaign boundary. No post days in an active bucket.
    periods = [
        (i, min(i + 7, end), phase)
        for first, end, phase in [(0, active_days, "campaign"), (active_days, days, "post")]
        for i in range(first, end, 7)
    ]
    weekly = [[] for _ in periods]
    phase_rows = {"campaign": [], "post": []}
    segments = [[] for _ in starts]
    rows = []
    rule_events = []
    for seed in plan["seeds"]:
        rng = np.random.default_rng(seed)
        remaining = limit
        baseline_buyer = np.zeros(len(daily), dtype=bool)
        campaign_buyer = baseline_buyer.copy()
        post_buyer = baseline_buyer.copy()
        redemption_counts = np.zeros(len(daily), dtype=int)
        stocked_until = np.full(len(daily), -1)
        totals = np.zeros((days, len(starts), 3), dtype=int)
        for day in range(days):
            draws = rng.random(len(daily))
            active = day < active_days
            base = draws < (ordinary if active else daily)
            purchased = base.copy()
            # One induced purchase can displace at most one later ordinary purchase.
            stocked = stocked_until >= day
            purchased[stocked] = False
            stocked_until[stocked & base] = -1
            redeemed = np.zeros(len(daily), dtype=bool)
            if active and remaining and day and day % 7 == 0:
                for rule in plan["contingency_rules"]:
                    utilized = Fraction(limit - remaining) * Fraction(discount) / budget
                    if utilized * 100 >= Fraction(str(rule["threshold_pct"])):
                        rule_events.append(
                            {
                                "seed": seed,
                                "rule": rule["rule"],
                                "date": (start_date + timedelta(days=day)).isoformat(),
                                "observed_pct": float(utilized * 100),
                            }
                        )
                        remaining = 0
            if active and remaining:
                possible = np.flatnonzero((draws < boosted) & ~stocked)
                priority = np.random.default_rng([seed, day, 1]).permutation(possible)
                redeemed[priority[:remaining]] = True
                remaining -= int(redeemed.sum())
                purchased |= redeemed
                stocked_until[redeemed & ~base] = day + settings["stockup_days"]
            if active:
                baseline_buyer |= base
                campaign_buyer |= purchased
                redemption_counts += redeemed
            else:
                post_buyer |= purchased
            for col, mask in enumerate((purchased, base, redeemed)):
                totals[day, :, col] = np.add.reduceat(mask, starts, dtype=int)

        def settle(counts, media=Decimal(0)):
            units, base_units, redeemed = map(int, counts)
            ledger = costs.ledger(units, redeemed, discount, media)
            baseline = costs.ledger(base_units, 0, discount, Decimal(0))
            return {
                **ledger,
                "baseline_units": base_units,
                "incremental_units": units - base_units,
                "baseline_contribution": baseline["contribution_profit"],
                "incremental_profit": ledger["contribution_profit"]
                - baseline["contribution_profit"],
                "incremental_net_receipts": ledger["net_receipts"] - baseline["net_receipts"],
            }

        row = settle(totals.sum(axis=(0, 1)), costs.media)
        overlap = int(redemption_counts[baseline_buyer].sum())
        post_units, post_base, _ = totals[active_days:].sum(axis=(0, 1))
        row.update(
            seed=seed,
            campaign_buyers=int(campaign_buyer.sum()),
            repeat_buyers=int((campaign_buyer & post_buyer).sum()),
            post_incremental_units=int(post_units - post_base),
            overlapping_subsidy=overlap * discount,
            incremental_campaign_buyers=int(campaign_buyer.sum() - baseline_buyer.sum()),
        )
        rows.append(row)
        phase_rows["campaign"].append(settle(totals[:active_days].sum(axis=(0, 1)), costs.media))
        phase_rows["post"].append(settle(totals[active_days:].sum(axis=(0, 1))))
        cumulative = Decimal(0)
        for i, (first, end, _) in enumerate(periods):
            period = settle(
                totals[first:end].sum(axis=(0, 1)), costs.media if i == 0 else Decimal(0)
            )
            cumulative += period["incremental_profit"]
            weekly[i].append({**period, "cumulative_incremental_profit": cumulative})
        for i, counts in enumerate(totals.sum(axis=0)):
            segment = settle(counts)
            last = starts[i + 1] if i + 1 < len(starts) else len(daily)
            group = slice(starts[i], last)
            segment["subsidy_leakage"] = _ratio(
                int(redemption_counts[group][baseline_buyer[group]].sum()), segment["redemptions"]
            )
            segment["incremental_campaign_buyers"] = int(
                campaign_buyer[group].sum() - baseline_buyer[group].sum()
            )
            segment["incremental_profit_before_media"] = segment["incremental_profit"]
            segments[i].append(segment)

    def distribution(records, name, unit):
        return metric(
            name, unit, [float(r[name]) if r[name] is not None else None for r in records]
        )

    units = {
        "orders": "orders",
        "units": "units",
        "redemptions": "orders",
        "incremental_units": "units",
        "incremental_campaign_buyers": "buyers",
    }
    metrics = [
        distribution(rows, name, units.get(name, "THB"))
        for name in (
            *costs.ledger(0, 0, discount, Decimal(0)),
            "incremental_units",
            "incremental_campaign_buyers",
            "incremental_profit",
            "incremental_net_receipts",
        )
    ]
    for name, num, den, unit in (
        ("campaign_iroi", "incremental_profit", "campaign_spend", "ratio"),
        ("incremental_roas", "incremental_net_receipts", "media_spend", "ratio"),
        ("subsidy_leakage", "overlapping_subsidy", "seller_subsidy", "fraction"),
    ):
        metrics.append(metric(name, unit, [_ratio(r[num], r[den]) for r in rows]))
    profits = [r["incremental_profit"] for r in rows]
    target = Decimal(str(terms["goal_target"]["minimum"]))
    for name, count in (
        ("probability_loss", sum(p < 0 for p in profits)),
        ("probability_hit_target", sum(p >= target for p in profits)),
    ):
        metrics.append(
            {
                "metric": name,
                "unit": "fraction",
                "status": "available",
                "value": count / len(rows),
                "unavailable_reason": None,
            }
        )
    metrics.append(distribution(rows, "post_incremental_units", "units"))
    metrics.append(
        metric(
            "repeat_rate",
            "fraction",
            [_ratio(r["repeat_buyers"], r["campaign_buyers"]) for r in rows],
        )
    )
    metrics.append(
        metric(
            "pull_forward_share",
            "fraction",
            [
                _ratio(max(0, -r["post_incremental_units"]), c["incremental_units"])
                for r, c in zip(rows, phase_rows["campaign"], strict=True)
            ],
        )
    )
    metrics.append(
        metric(
            "budget_utilization",
            "fraction",
            [_ratio(r["campaign_spend"], Decimal(str(plan["total_budget_thb"]))) for r in rows],
        )
    )
    for name, unit in (
        ("platform_roas", "ratio"),
        ("incremental_new_buyers", "buyers"),
        ("impressions", "impressions"),
        ("clicks", "clicks"),
        ("click_through_rate", "fraction"),
        ("conversion_rate", "fraction"),
    ):
        metrics.append(metric(name, unit, None, "measurement_unsupported"))

    return {
        "scenario": scenario,
        "metrics": metrics,
        "paired_seed_count": len(rows),
        "rule_events": rule_events,
        "seed_ledgers": [
            {k: float(v) if isinstance(v, Decimal) else v for k, v in r.items()} for r in rows
        ],
        "period_results": [
            {
                "phase": phase,
                "start_date": plan["campaign_window"]["start_date"]
                if phase == "campaign"
                else plan["campaign_window"]["end_date"],
                "end_date": plan["campaign_window"]["end_date"]
                if phase == "campaign"
                else terms["measurement_window"]["end_date"],
                "metrics": [
                    distribution(records, n, units.get(n, "THB"))
                    for n in (
                        "orders",
                        "units",
                        "redemptions",
                        "incremental_units",
                        "gross_sales",
                        "net_receipts",
                        "seller_subsidy",
                        "campaign_spend",
                        "contribution_profit",
                        "incremental_profit",
                    )
                ],
            }
            for phase, records in phase_rows.items()
        ],
        "weekly": [
            {
                "start_date": (start_date + timedelta(days=first)).isoformat(),
                "end_date": (start_date + timedelta(days=end)).isoformat(),
                "phase": phase,
                "metrics": [
                    distribution(weekly[i], n, units.get(n, "THB"))
                    for n in (
                        "units",
                        "incremental_units",
                        "campaign_spend",
                        "incremental_profit",
                        "cumulative_incremental_profit",
                    )
                ],
            }
            for i, (first, end, phase) in enumerate(periods)
        ],
        "audience": [
            {
                "segment_id": segment["segment_id"],
                "metrics": [
                    distribution(segments[i], n, u)
                    for n, u in (
                        ("incremental_campaign_buyers", "buyers"),
                        ("incremental_units", "units"),
                        ("subsidy_leakage", "fraction"),
                        ("seller_subsidy", "THB"),
                        ("incremental_profit_before_media", "THB"),
                    )
                ],
            }
            for i, segment in enumerate(plan["segments"])
        ],
    }
