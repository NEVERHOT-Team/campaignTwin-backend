"""Calibrate the no-offer shopper baseline against prelaunch manufacturer units.

Run `python -m data.etl.calibrate_shopper` from the project root with the local
calibration panel and raw Complete Journey tables available. WeekStats exposes
own-brand units only, so both sides use the selected manufacturer's units per
panel household-week. The 6-week (demo horizon) and full-panel means must both
land within 10% of that target.
"""

import json
import os
from dataclasses import fields, replace
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from data import loaders
from data.etl.config import MANUFACTURERS
from engine import shopper

AGENTS = 5000
SEEDS = tuple(range(5))
TOLERANCE = 0.10


def panel_target():
    panel = loaders.load_household_behavior()
    households = panel.household_key.nunique()
    weeks = sorted(map(int, panel.week.unique()))
    if not weeks or weeks != list(range(1, weeks[-1] + 1)):
        raise ValueError("Need contiguous prelaunch weeks starting at week 1")
    if (
        households == 0
        or len(panel) != households * len(weeks)
        or panel.duplicated(["household_key", "week"]).any()
    ):
        raise ValueError("Need one panel row per household and prelaunch week")

    (manufacturer,) = [key for key, role in MANUFACTURERS.items() if role == "our_brand"]
    products = loaders.load_category_products()
    product_ids = products.loc[products.manufacturer_id.eq(manufacturer), "product_id"]
    # A household-week may include several brands. The panel's single brand is the
    # weekly winner, so using it would misassign mixed baskets. Recover the exact
    # own-brand numerator from the same positive-quantity lines used to build the panel.
    purchases = loaders.load_raw_table(
        "transactions",
        columns=["quantity"],
        filters=[
            ("week", ">=", weeks[0]),
            ("week", "<=", weeks[-1]),
            ("product_id", "in", product_ids.tolist()),
            ("quantity", ">", 0),
        ],
    )
    units = float(purchases.quantity.sum())
    if not np.isfinite(units) or units <= 0:
        raise ValueError("Missing prelaunch own-brand unit volume")
    return {
        "manufacturer": int(manufacturer),
        "households": int(households),
        "weeks": len(weeks),
        "own_units": units,
        "units_per_household_week": units / len(panel),
    }


def baseline_volume(population, params, panel_weeks, demo_weeks):
    totals = np.zeros(panel_weeks)
    no_offer = shopper.WeekOffer()
    for seed in SEEDS:
        state = shopper.initial_state(population, seed)
        for index in range(panel_weeks):
            stats, state = shopper.step(population, state, no_offer, seed, index + 1, params)
            totals[index] += stats.units_us
    units = totals / (len(SEEDS) * population.households)
    return {"demo": float(units[:demo_weeks].mean()), "panel": float(units.mean())}


def calibrate():
    panel = panel_target()
    target = panel["units_per_household_week"]
    demo_weeks = sum(loaders.load_market()["weeks"].values())
    if not 1 <= demo_weeks <= panel["weeks"]:
        raise ValueError("Demo horizon must fit within the training panel")
    defaults = shopper.ShopperParams()
    population = shopper.build_population(AGENTS, seed=0, params=defaults)

    # Both non-focal alternatives receive the same offset, preserving their relative
    # preference. This is one fitted degree of freedom. Own-brand unit volume is too
    # high at default settings, so reduce its relative probability in a no-offer week.
    def assess(offset):
        params = replace(
            defaults, asc_rival=defaults.asc_rival + offset, asc_other=defaults.asc_other + offset
        )
        volume = baseline_volume(population, params, panel["weeks"], demo_weeks)
        residual = [value / target - 1 for value in volume.values()]
        return max(map(abs, residual)), min(residual) + max(residual), params, volume

    best = assess(0.0)
    default_volume = best[3]
    if best[0] > TOLERANCE:
        # Moving the common intercept upwards lowers own-brand take-up. Balance the
        # opposite-signed percentage errors of the two horizons on a fixed seed panel.
        low, high = -4.0, 4.0
        for _ in range(12):
            midpoint = round((low + high) / 2, 6)
            outcome = assess(midpoint)
            if outcome[0] < best[0]:
                best = outcome
            if outcome[1] > 0:
                low = midpoint
            else:
                high = midpoint
    error, _, fitted, volume = best
    if error > TOLERANCE:
        raise ValueError(f"Own-brand volume target {target:g} not reached: {volume}")
    description = (
        f"Manufacturer {panel['manufacturer']} own-brand units per panel household-week "
        f"target {target:.12g} (within 10% over {demo_weeks} and {panel['weeks']} weeks). "
        f"Achieved is the {panel['weeks']}-week mean."
    )
    parameters = {}
    for field in fields(defaults):
        if field.name == "label":
            continue
        value = getattr(fitted, field.name)
        changed = value != getattr(defaults, field.name)
        parameters[field.name] = {
            "value": value,
            "label": "calibrated" if changed else "default",
            "target": description if changed else None,
            "achieved": volume["panel"] if changed else None,
        }
    return parameters, {
        "measure": "Own-brand units; WeekStats omits units for rival and other brands",
        "panel": panel,
        "demo_weeks": demo_weeks,
        "agents": AGENTS,
        "population_seed": 0,
        "paired_seeds": list(SEEDS),
        "households_per_agent": fitted.households_per_agent,
        "default_units_per_household_week": default_volume,
        "fitted_units_per_household_week": volume,
    }


def main():
    parameters, report = calibrate()
    path = loaders.PROCESSED / "shopper_params.json"
    draft = NamedTemporaryFile("w", dir=path.parent, encoding="utf-8", delete=False)
    tmp_path = Path(draft.name)
    try:
        with draft:
            json.dump(parameters, draft, ensure_ascii=False, indent=2, allow_nan=False)
            draft.write("\n")
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
