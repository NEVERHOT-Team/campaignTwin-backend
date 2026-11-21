import inspect
import logging
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pydantic import BaseModel

from agents import llm
from contracts.models import BriefParseResponse
from mocks import load_mock


class Answer(BaseModel):
    value: int


def completion(parsed=None, refusal=None):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(parsed=parsed, refusal=refusal),
            )
        ]
    )


@pytest.fixture
def provider(monkeypatch):
    cache = {}
    writes = []
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-secret-value")
    monkeypatch.setenv("MODEL_STRONG", "strong-test")
    monkeypatch.setenv("MODEL_FAST", "fast-test")
    monkeypatch.setattr(llm.db, "cache_get", lambda key: cache.get(key))

    def put(key, model, prompt, response):
        writes.append({"key": key, "model": model, "prompt": prompt, "response": response})
        cache[key] = response

    monkeypatch.setattr(llm.db, "cache_put", put)
    parse = Mock(return_value=completion(Answer(value=7)))
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(parse=parse)), close=Mock()
    )
    factory = Mock(return_value=client)
    monkeypatch.setattr(llm, "OpenAI", factory)
    return SimpleNamespace(cache=cache, writes=writes, parse=parse, factory=factory, client=client)


def test_signature_aliases_and_cache_hit_skip_provider(provider, caplog):
    assert list(inspect.signature(llm.call_llm).parameters) == ["model", "messages", "schema"]
    messages = [{"role": "user", "content": "A tiny example"}]
    with caplog.at_level(logging.INFO):
        first = llm.call_llm("fast", messages, Answer)
        assert llm.call_llm("fast-test", messages, Answer) == first == Answer(value=7)
    provider.factory.assert_called_once_with(
        api_key="sk-test-secret-value",
        timeout=30.0,
        max_retries=1,
    )
    assert provider.parse.call_args.kwargs["response_format"] is Answer
    assert provider.parse.call_args.kwargs["model"] == "fast-test"
    assert provider.client.close.call_count == 1
    assert "path=openai" in caplog.text
    assert "path=cache provider=openai" in caplog.text


def test_cache_identity_includes_model_provider_prompt_and_schema(provider, monkeypatch):
    assert llm.call_llm("strong", [], Answer) == Answer(value=7)
    assert llm.call_llm("fast", [], Answer) == Answer(value=7)
    monkeypatch.setattr(llm, "PROMPT_VERSION", "a-new-version")
    assert llm.call_llm("fast", [], Answer) == Answer(value=7)
    assert provider.parse.call_count == 3
    assert len(provider.cache) == 3
    assert all(write["prompt"]["provider"] == "openai" for write in provider.writes)
    primary, _ = llm._identity("openai", "fast-test", [], Answer)
    fallback, _ = llm._identity("fixture-v1", "fast-test", [], Answer)
    assert primary != fallback


def test_provider_timeout_falls_back_without_poisoning_provider_cache(provider, caplog):
    provider.parse.side_effect = TimeoutError("timed out")
    expected = BriefParseResponse.model_validate(load_mock("campaign_spec"))
    with caplog.at_level(logging.INFO):
        assert llm.call_llm("strong", [], BriefParseResponse) == expected
    assert provider.writes[0]["prompt"]["provider"] == "fixture-v1"
    assert "path=fixture" in caplog.text
    provider.parse.side_effect = None
    provider.parse.return_value = completion(expected)
    assert llm.call_llm("strong", [], BriefParseResponse) == expected
    assert provider.parse.call_count == 2
    assert provider.writes[-1]["prompt"]["provider"] == "openai"


def test_missing_key_uses_fixture_or_none(provider, monkeypatch, caplog):
    monkeypatch.setenv("OPENAI_API_KEY", "")
    with caplog.at_level(logging.INFO):
        assert isinstance(llm.call_llm("fast", [], BriefParseResponse), BriefParseResponse)
        assert llm.call_llm("fast", [], Answer) is None
    provider.factory.assert_not_called()
    assert "path=fixture" in caplog.text and "path=none" in caplog.text


def test_live_cache_remains_usable_without_key(provider, monkeypatch):
    llm.call_llm("fast", [], Answer)
    monkeypatch.setenv("OPENAI_API_KEY", "")
    assert llm.call_llm("fast", [], Answer) == Answer(value=7)
    assert provider.parse.call_count == 1


@pytest.mark.parametrize(
    "value", [completion(), completion(refusal="refused"), completion({"value": "bad"})]
)
def test_invalid_or_refused_provider_output_is_never_cached(provider, value):
    provider.parse.return_value = value
    assert llm.call_llm("strong", [], Answer) is None
    assert provider.writes == []


def test_invalid_cache_is_replaced_with_valid_output(provider):
    key, _ = llm._identity("openai", "fast-test", [], Answer)
    provider.cache[key] = {"value": "invalid"}
    assert llm.call_llm("fast", [], Answer) == Answer(value=7)
    assert provider.parse.call_count == 1
    assert provider.cache[key] == {"value": 7}


def test_cache_failure_does_not_prevent_valid_provider_response(provider, monkeypatch):
    monkeypatch.setattr(llm.db, "cache_get", Mock(side_effect=OSError("storage unavailable")))
    monkeypatch.setattr(llm.db, "cache_put", Mock(side_effect=OSError("storage unavailable")))
    assert llm.call_llm("fast", [], Answer) == Answer(value=7)


def test_errors_redact_full_and_masked_keys(provider, caplog):
    provider.parse.side_effect = RuntimeError("invalid sk-test-secret-value and sk-proj-abc***xyz")
    llm.call_llm("fast", [], Answer)
    assert "sk-test-secret-value" not in caplog.text
    assert "sk-proj-abc" not in caplog.text
    assert "[REDACTED]" in caplog.text


def test_check_calls_each_live_model_once_without_cache_or_fixtures(provider, monkeypatch, capsys):
    provider.parse.return_value = completion({"ok": True})
    read_cache = Mock(side_effect=AssertionError("live probe must bypass cache"))
    monkeypatch.setattr(llm.db, "cache_get", read_cache)
    assert llm.check_models() == 0
    assert [call.kwargs["model"] for call in provider.parse.call_args_list] == [
        "strong-test",
        "fast-test",
    ]
    read_cache.assert_not_called()
    assert not provider.writes
    output = capsys.readouterr().out
    assert "MODEL_STRONG (strong-test): OK" in output
    assert "MODEL_FAST (fast-test): OK" in output


def test_check_reports_provider_error_and_continues_to_other_model(provider, capsys):
    provider.parse.side_effect = [RuntimeError("model not found"), completion({"ok": True})]
    assert llm.check_models() == 1
    output = capsys.readouterr().out
    assert "MODEL_STRONG (strong-test): FAIL - RuntimeError: model not found" in output
    assert "MODEL_FAST (fast-test): OK" in output
    assert provider.parse.call_count == 2
