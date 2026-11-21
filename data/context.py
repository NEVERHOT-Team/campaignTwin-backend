"""Report inputs assembled from the campaign library and rival model."""

from typing import Any

from contracts.models import CampaignSpec, Strategy
from data import loaders, rival


def competitive_outlook(spec: CampaignSpec, strategy: Strategy | None = None) -> dict[str, Any]:
    """Group scenario estimates by rival and include up to five sourced campaign examples.

    Depth is in percentage points, delay and duration are in weeks, and likelihood
    and its interval (ci_low, ci_high) are probabilities. prior_label describes
    the category prior. Each scenario retains its own evidence IDs.
    """
    by_rival: dict[str, dict[str, Any]] = {}
    for estimate in rival.get_rival_scenarios(spec, strategy):
        if estimate.rival not in by_rival:
            by_rival[estimate.rival] = {
                "rival": estimate.rival,
                "display_name": loaders.brand_display_name(estimate.rival),
                "rival_score": estimate.rival_score,
                "scenarios": {},
            }
        by_rival[estimate.rival]["scenarios"][estimate.scenario] = estimate.model_dump(
            exclude={"rival", "rival_score", "scenario"}
        )
    return {
        "prior_label": loaders.load_rival_prior()["source"],
        "rivals": list(by_rival.values()),
        "similar_campaigns": rival.similar_campaigns(spec, strategy, k=5),
    }
