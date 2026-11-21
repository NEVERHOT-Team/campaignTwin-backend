"""Cached fixture responses until a model provider is wired in."""

import json
from hashlib import sha256
from typing import Any, TypeVar

from pydantic import BaseModel

import db
from contracts.models import BriefParseResponse, Report
from mocks import load_mock

Response = TypeVar("Response", bound=BaseModel)


def call_llm(
    model: str, messages: list[dict[str, Any]], schema: type[Response]
) -> Response | None:
    prompt = {"messages": messages, "schema": schema.model_json_schema(), "provider": "fixture-v1"}
    encoded = json.dumps({"model": model, **prompt}, sort_keys=True, ensure_ascii=False)
    key = sha256(encoded.encode()).hexdigest()
    cached = db.cache_get(key)
    if cached is not None:
        return schema.model_validate(cached)

    fixture = {BriefParseResponse: "campaign_spec", Report: "report"}.get(schema)
    if fixture is None:
        return None
    result = schema.model_validate(load_mock(fixture))
    db.cache_put(key, model, prompt, result.model_dump(mode="json"))
    return result
