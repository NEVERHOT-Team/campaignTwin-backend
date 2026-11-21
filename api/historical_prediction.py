"""Compare an optional engine-owner artifact with a run's pinned observation."""

import json

from api.v1_contract import validator

# Protocol v1 fixes the population and the control adjustment via the schema.
# These fields further bind a prediction to the exact case and holdout.
_CASE_FIELDS = (
    "campaign_id", "evidence_version", "metric", "unit",
    "fit_through_week", "baseline_weeks", "campaign_weeks",
)


def compare_prediction(reference: dict, artifact: dict | None = None) -> dict:
    """Use inclusive sensitivity bounds from an already verified HistoricalCase."""
    if artifact is None:
        return {"status": "reference_only", "reason": "prediction_missing", "prediction": None}

    incompatible = {"status": "reference_only", "reason": "incompatible_prediction",
                    "prediction": None}
    # JSON Schema's Python number type also accepts complex and decimal objects.
    try:
        json.dumps(artifact, allow_nan=False)
    except (TypeError, ValueError):
        return incompatible

    if (not validator("HistoricalPrediction").is_valid(artifact)
            or any(artifact[key] != reference[key] for key in _CASE_FIELDS)):
        return incompatible

    bounds = reference["observed_range"]
    inside = bounds["low"] <= artifact["value"] <= bounds["high"]
    return {"status": "inside_range" if inside else "outside_range",
            "reason": None, "prediction": artifact}
