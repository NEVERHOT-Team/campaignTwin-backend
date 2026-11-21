import json

import numpy as np
import pytest

from agents import llm, ssr
from contracts.models import Strategy
from data.loaders import load_anchors, load_personas

ANSWER_TOP = "answer-top"
ANSWER_BOTTOM = "answer-bottom"


def fake_embed(texts):
    """Anchor i -> one-hot e_i, so cosine similarity is exact and easy to reason about."""
    position = {}
    for group in load_anchors()["sets"]:
        for index, anchor in enumerate(group["anchors"]):
            position[anchor] = index
    position[ANSWER_TOP] = 4
    position[ANSWER_BOTTOM] = 0
    vectors = []
    for text in texts:
        vector = np.full(5, 0.1)
        vector[position.get(text, 2)] = 1.0
        vectors.append(vector.tolist())
    return vectors


def offer(discount=40.0, mechanic="percent_off"):
    return ssr.MessageOffer(
        message_id="m1",
        text="ลด 40% ทั้งร้าน",
        discount_pct=discount,
        mechanic=mechanic,
    )


@pytest.fixture
def persona():
    return load_personas()[0]


def test_pmf_from_similarities_shape_and_order():
    pmf = ssr.pmf_from_similarities([0.1, 0.2, 0.3, 0.4, 0.9])
    assert pmf.sum() == pytest.approx(1)
    assert pmf[0] == 0 and np.argmax(pmf) == 4
    assert ssr.pmf_from_similarities([0.5] * 5).tolist() == pytest.approx([0.2] * 5)
    with pytest.raises(ValueError):
        ssr.pmf_from_similarities([0.1, 0.2])


def test_answer_matching_top_anchor_scores_high():
    anchors = ssr.anchor_matrices(fake_embed)
    top = ssr.ssr_pmf(fake_embed([ANSWER_TOP])[0], anchors)
    bottom = ssr.ssr_pmf(fake_embed([ANSWER_BOTTOM])[0], anchors)
    assert np.argmax(top) == 4 and np.argmax(bottom) == 0
    assert ssr.appeal_from_pmf(top) > 0 > ssr.appeal_from_pmf(bottom)
    assert -1 <= ssr.appeal_from_pmf(top) <= 1


def test_rate_uses_llm_answer_when_available(persona):
    anchors = ssr.anchor_matrices(fake_embed)

    def ask(model, messages, schema):
        assert model == "fast" and schema is ssr.PersonaAnswer
        return ssr.PersonaAnswer(answer=ANSWER_TOP)

    result = ssr.rate(persona, offer(), anchors, ask=ask, embed=fake_embed)
    assert result.label == "llm_estimate"
    assert result.answer == ANSWER_TOP and result.appeal > 0
    assert sum(result.pmf) == pytest.approx(1)


@pytest.mark.parametrize("ask_result", [None, Exception("boom")])
def test_rate_falls_back_to_heuristic(persona, ask_result):
    anchors = ssr.anchor_matrices(fake_embed)

    def ask(*_):
        if isinstance(ask_result, Exception):
            raise ask_result
        return ask_result

    result = ssr.rate(persona, offer(), anchors, ask=ask, embed=fake_embed)
    assert result.label == "default" and result.answer is None
    assert -1 <= result.appeal <= 1
    assert sum(result.pmf) == pytest.approx(1)


def test_heuristic_prefers_deal_seekers_and_bigger_discounts():
    deal = {"persona_id": "d", "promo_share": 1.0, "brand_loyalty": 0.1,
            "traits": {"impulsiveness": 0.7}}
    loyal = {"persona_id": "l", "promo_share": 0.1, "brand_loyalty": 0.9,
             "traits": {"impulsiveness": 0.3}}
    assert ssr.heuristic_appeal(deal, offer(40)) > ssr.heuristic_appeal(loyal, offer(40))
    assert ssr.heuristic_appeal(deal, offer(40)) > ssr.heuristic_appeal(deal, offer(5))


def test_score_all_offline_is_complete_and_deterministic():
    personas = load_personas()[:5]
    offers = [offer(), ssr.MessageOffer(message_id="m2", text="ส่งฟรี", discount_pct=0,
                                        mechanic="free_shipping")]
    offline = lambda texts: None  # noqa: E731
    first = ssr.score_all(personas, offers, embed=offline)
    assert len(first) == 10
    assert {key[1] for key in first} == {"m1", "m2"}
    assert all(result.label == "default" for result in first.values())
    assert first == ssr.score_all(personas, offers, embed=offline)
    table = ssr.appeal_table(first)
    assert all(-1 <= value <= 1 for value in table.values())


def test_score_all_with_fake_model_runs_concurrently():
    personas = load_personas()[:4]
    ask = lambda *_: ssr.PersonaAnswer(answer=ANSWER_TOP)  # noqa: E731
    results = ssr.score_all(personas, [offer()], ask=ask, embed=fake_embed, max_workers=4)
    assert all(result.label == "llm_estimate" for result in results.values())


def test_persona_prompt_contains_profile_and_offer_only(persona):
    messages = ssr.persona_messages(persona, offer())
    payload = json.loads(messages[1]["content"])
    assert set(payload) == {"shopper", "offer"}
    assert payload["offer"]["discount_pct"] == 40
    assert "weight" not in payload["shopper"] and "source" not in payload["shopper"]


def test_offers_from_strategies_skip_baseline():
    def strategy(sid, baseline=False):
        return Strategy(
            id=sid, name=sid, discount_pct=0 if baseline else 25, mechanic="voucher",
            target_segments=[], message=f"{sid} message", channel_mix={"shopee": 1.0},
            timing={}, cap_per_user=None, budget_split={}, contingency_rules=[],
            rationale="", is_baseline=baseline,
        )

    offers = ssr.offers_from_strategies([strategy("baseline", True), strategy("g0-deep")])
    assert [o.message_id for o in offers] == ["g0-deep"]
    assert offers[0].text == "g0-deep message" and offers[0].channels == ["shopee"]


def test_embed_without_key_returns_none(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("EMBED_MODEL", "test-embed")
    monkeypatch.setattr(llm.db, "cache_get", lambda key: None)
    assert llm.embed(["hello"]) is None
    assert llm.embed([]) == []
