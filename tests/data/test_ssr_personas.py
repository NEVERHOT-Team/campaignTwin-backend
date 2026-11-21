import json
from collections import Counter

import pytest

from data import loaders
from data.etl.appeal import ARCHETYPES
from data.etl.ssr_personas import build_ssr_personas


def test_ssr_sample_covers_each_archetype():
    personas = {p["persona_id"]: p for p in loaders.load_personas()}
    sample = loaders.load_ssr_personas()

    assert len(sample) == len({p["persona_id"] for p in sample}) == 25
    assert Counter(p["archetype"] for p in sample) == dict.fromkeys(ARCHETYPES, 5)
    for archetype in ARCHETYPES:
        rows = [p for p in sample if p["archetype"] == archetype]
        profiles = [personas[p["persona_id"]] for p in rows]
        assert all(profile["archetype"] == archetype for profile in profiles)
        assert any(p["persona_id"].startswith("tt-edge-") for p in profiles)
        known_cells = {
            (p["age_band"], p["income_band"])
            for p in profiles
            if p["age_band"] != "Unknown" and p["income_band"] != "Unknown"
        }
        assert len(known_cells) > 1
        assert all(isinstance(p["reason"], str) and p["reason"].strip() for p in rows)
        weights = [p["weight_in_archetype"] for p in rows]
        assert all(0 < weight <= 1 for weight in weights)
        assert sum(weights) == pytest.approx(1)


def test_weighted_ssr_average_preserves_archetype_edge_share():
    # A score of 1 for edges and 0 for regular personas should reproduce the full-panel mean.
    personas = loaders.load_personas()
    sample = loaders.load_ssr_personas()
    for archetype in ARCHETYPES:
        full = [p for p in personas if p["archetype"] == archetype]
        edge_mass = sum(p["weight"] for p in full if p["persona_id"].startswith("tt-edge-"))
        sample_mean = sum(
            p["weight_in_archetype"]
            for p in sample
            if p["archetype"] == archetype and p["persona_id"].startswith("tt-edge-")
        )
        assert sample_mean == pytest.approx(edge_mass / sum(p["weight"] for p in full))


def test_representative_inherits_unselected_personas_weight_in_the_same_cell():
    # Six source personas in five distinct cells, including two in the same regular cell.
    personas = [
        {
            "persona_id": persona_id,
            "archetype": "loyal",
            "age_band": "35-44",
            "income_band": income,
            "weight": weight,
        }
        for persona_id, income, weight in [
            ("tt-1", "Under 15K", 0.20),
            ("tt-2", "Under 15K", 0.30),
            ("tt-3", "15-24K", 0.15),
            ("tt-4", "25-34K", 0.15),
            ("tt-5", "35-49K", 0.15),
            ("tt-edge-6", "50-74K", 0.05),
        ]
    ]
    weights = {p["persona_id"]: p["weight_in_archetype"] for p in build_ssr_personas(personas)}
    assert weights == pytest.approx(
        {"tt-2": 0.50, "tt-3": 0.15, "tt-4": 0.15, "tt-5": 0.15, "tt-edge-6": 0.05}
    )


def test_published_ssr_sample_is_reproducible():
    personas = loaders.load_personas()
    published = loaders.load_ssr_personas()
    assert build_ssr_personas(personas) == published
    assert build_ssr_personas(list(reversed(personas)), seed=42) == published


def test_loader_rejects_unnormalized_archetype_weights(tmp_path, monkeypatch):
    rows = [dict(row) for row in loaders.load_ssr_personas()]
    rows[0]["weight_in_archetype"] /= 2
    (tmp_path / "ssr_personas.json").write_text(json.dumps(rows), encoding="utf-8")
    monkeypatch.setattr(loaders, "PROCESSED", tmp_path)
    loaders.load_ssr_personas.cache_clear()
    with pytest.raises(ValueError, match="weights summing to one"):
        loaders.load_ssr_personas()
