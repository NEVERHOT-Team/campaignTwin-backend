"""Choose weighted age and income representatives for archetype-level SSR scoring."""

from itertools import groupby
from math import fsum

import numpy as np

from data.etl.config import SEED

SAMPLE_PER_ARCHETYPE = 5


def _representatives(profiles, slots, covered, rng):
    # Break exact ties reproducibly, including when the input file is reordered.
    tie_order = rng.permutation(len(profiles))
    cells = [(p["age_band"], p["income_band"]) for p in profiles]
    bands = np.array(cells)
    distance = (bands[:, None, :] != bands[None, :, :]).sum(axis=2)
    weights = np.array([p["weight"] for p in profiles])

    selected = []
    nearest = np.full(len(profiles), 3)  # More than two mismatched bands.
    for _ in range(slots):
        # Within unrepresented cells, minimize the group's weighted band mismatches.
        costs = (weights[:, None] * np.minimum(nearest[:, None], distance)).sum(axis=0).round(12)
        pick = min(
            (i for i in range(len(profiles)) if i not in selected),
            key=lambda i: (cells[i] in covered, costs[i], -weights[i], tie_order[i]),
        )
        selected.append(pick)
        covered.add(cells[pick])
        nearest = np.minimum(nearest, distance[:, pick])

    # Every source profile contributes its weight, shared equally across nearest ties.
    matches = distance[:, selected] == nearest[:, None]
    masses = (weights[:, None] * matches / matches.sum(axis=1, keepdims=True)).sum(axis=0)
    return [(profiles[i], mass) for i, mass in zip(selected, masses, strict=True)]


def build_ssr_personas(personas: list[dict], *, seed: int = SEED) -> list[dict]:
    """Select five profiles per archetype and assign their represented population mass."""
    rng = np.random.default_rng(seed)
    ordered = sorted(personas, key=lambda p: (p["archetype"], p["persona_id"]))
    sample = []
    for archetype, profiles in groupby(ordered, key=lambda p: p["archetype"]):
        group = list(profiles)
        edges = [p for p in group if p["persona_id"].startswith("tt-edge-")]
        regular = [p for p in group if not p["persona_id"].startswith("tt-edge-")]
        if len(group) < SAMPLE_PER_ARCHETYPE or not edges:
            raise ValueError(f"{archetype} needs five distinct personas including an edge persona")
        regular_slots = min(len(regular), SAMPLE_PER_ARCHETYPE - 1)
        total = fsum(p["weight"] for p in group)
        covered = set()
        for pool, slots, kind in (
            (regular, regular_slots, "Regular"),
            (edges, SAMPLE_PER_ARCHETYPE - regular_slots, "Edge"),
        ):
            if not slots:
                continue
            for persona, mass in _representatives(pool, slots, covered, rng):
                age, income = persona["age_band"], persona["income_band"]
                sample.append(
                    {
                        "persona_id": persona["persona_id"],
                        "archetype": archetype,
                        "weight_in_archetype": round(float(mass / total), 12),
                        "reason": f"{kind} profile for age {age} and income {income}, "
                        "selected for weighted demographic coverage.",
                    }
                )
    return sample
