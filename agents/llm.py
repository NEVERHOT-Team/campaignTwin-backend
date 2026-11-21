"""Cached, typed OpenAI calls with explicit fixture/None fallbacks.

Run ``python -m agents.llm --check`` for one uncached provider call per model.
The probe never reports a cached or fixture response as a successful live call.
"""

import argparse
import json
import logging
import os
import re
from hashlib import sha256
from typing import Any, TypeVar

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

import db
from contracts.models import BriefParseResponse, Report
from mocks import load_mock

load_dotenv(db.ROOT / ".env")
logger = logging.getLogger(__name__)
Response = TypeVar("Response", bound=BaseModel)
PROMPT_VERSION = "structured-v1"


def resolve_model(model: str) -> str | None:
    name = model.strip()
    alias = {"strong": "MODEL_STRONG", "fast": "MODEL_FAST"}.get(name.lower())
    return (os.getenv(alias, "").strip() if alias else name) or None


def _safe_error(error: Exception) -> str:
    message = f"{type(error).__name__}: {error}"
    for variable in ("OPENAI_API_KEY", "DATABASE_URL"):
        value = os.getenv(variable)
        if value:
            message = message.replace(value, "[REDACTED]")
    message = re.sub(r"sk-[A-Za-z0-9_*\-]{4,}", "[REDACTED]", message)
    return message[:600]


def _identity(provider: str, model: str, messages: list, schema: type[BaseModel]):
    prompt = {
        "provider": provider,
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "messages": messages,
        "schema": schema.model_json_schema(),
    }
    encoded = json.dumps(prompt, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return sha256(encoded.encode()).hexdigest(), prompt


def _cache_read(key: str, model: str, schema: type[Response], provider: str) -> Response | None:
    try:
        value = db.cache_get(key)
        if value is not None:
            result = schema.model_validate(value)
            logger.info("llm path=cache provider=%s model=%s", provider, model)
            return result
    except Exception as error:
        logger.warning("llm cache_read_failed model=%s error=%s", model, _safe_error(error))
    return None


def _cache_write(key: str, model: str, prompt: dict, result: BaseModel) -> None:
    try:
        db.cache_put(key, model, prompt, result.model_dump(mode="json"))
    except Exception as error:
        logger.warning("llm cache_write_failed model=%s error=%s", model, _safe_error(error))


def _provider_call(model: str, messages: list, schema: type[Response]) -> Response:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=30.0, max_retries=1)
    try:
        completion = client.chat.completions.parse(
            model=model,
            messages=messages,
            response_format=schema,
        )
        if not completion.choices:
            raise ValueError("Provider returned no completion choices")
        message = completion.choices[0].message
        if message.refusal:
            raise ValueError("Provider refused the structured response")
        if message.parsed is None:
            raise ValueError("Provider returned no parsed structured response")
        return schema.model_validate(message.parsed)
    finally:
        client.close()


def call_llm(model: str, messages: list[dict[str, Any]], schema: type[Response]) -> Response | None:
    resolved = resolve_model(model)
    cache_model = resolved or model
    key, prompt = _identity("openai", cache_model, messages, schema)
    cached = _cache_read(key, cache_model, schema, "openai")
    if cached is not None:
        return cached

    if resolved and os.getenv("OPENAI_API_KEY", "").strip():
        try:
            result = _provider_call(resolved, messages, schema)
            _cache_write(key, resolved, prompt, result)
            logger.info("llm path=openai model=%s", resolved)
            return result
        except Exception as error:
            logger.warning("llm provider_failed model=%s error=%s", resolved, _safe_error(error))
    else:
        logger.info("llm provider_skipped model=%s reason=missing_key_or_model", cache_model)

    # A fallback is cached separately so it cannot masquerade as a live response
    # or prevent recovery when the provider becomes available again.
    fixture_key, fixture_prompt = _identity("fixture-v1", cache_model, messages, schema)
    cached = _cache_read(fixture_key, cache_model, schema, "fixture-v1")
    if cached is not None:
        return cached
    fixture = {BriefParseResponse: "campaign_spec", Report: "report"}.get(schema)
    if fixture is not None:
        try:
            result = schema.model_validate(load_mock(fixture))
            _cache_write(fixture_key, cache_model, fixture_prompt, result)
            logger.info("llm path=fixture model=%s fixture=%s", cache_model, fixture)
            return result
        except Exception as error:
            logger.warning("llm fixture_failed model=%s error=%s", cache_model, _safe_error(error))
    logger.info("llm path=none model=%s", cache_model)
    return None


class _ModelCheck(BaseModel):
    ok: bool


def check_models() -> int:
    failed = False
    for variable, alias in (("MODEL_STRONG", "strong"), ("MODEL_FAST", "fast")):
        resolved = resolve_model(alias)
        try:
            if not resolved:
                raise ValueError(f"{variable} is not configured")
            if not os.getenv("OPENAI_API_KEY", "").strip():
                raise ValueError("OPENAI_API_KEY is not configured")
            result = _provider_call(
                resolved,
                [{"role": "user", "content": 'Connectivity check: return {"ok":true}.'}],
                _ModelCheck,
            )
            if not result.ok:
                raise ValueError("Provider returned ok=false")
            logger.info("llm path=openai-check model=%s", resolved)
            print(f"{variable} ({resolved}): OK", flush=True)
        except Exception as error:
            failed = True
            print(f"{variable} ({resolved or 'unset'}): FAIL - {_safe_error(error)}", flush=True)
    return int(failed)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="probe both configured models once")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    return check_models() if args.check else 0


if __name__ == "__main__":
    raise SystemExit(main())
