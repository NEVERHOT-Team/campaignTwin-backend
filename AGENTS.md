# CampaignTwin backend

CampaignTwin stress-tests e-commerce campaigns against simulated shopper and rival behavior before launch.

1. Stack and commands: `pip install -e ".[dev]"`, `uvicorn main:app --reload --host 0.0.0.0 --port 8000`, `pytest -q`, `ruff check .`.
2. Folder owners: api/agents/contracts = Big, engine = Mic, data/db = Gan. Stay in your folder.
3. `contracts/` is the API contract. Only Big edits it, and announces changes in team chat.
4. Read data only via `data/loaders.py`. Never open `data/processed/` or `data/raw/` directly.
5. No shared static DB. Static data lives in `data/processed/` (git). Postgres in Docker on Gan's Mac holds runs + LLM cache only. `DB_MODE=json` is the fallback.
6. Every value carries a source label: data | llm_estimate | default.
7. `data/processed/` holds aggregates (segments ≥ 50 households) and personas only. Never send household-level data to an LLM.
8. Engine is deterministic: numpy, vectorized, seeded; plan and baseline share seeds. No LLM calls inside the weekly loop.
9. All LLM calls go through `agents/llm.py` (cache by prompt hash, timeout, fallback).
10. Report numbers come only from results JSON; LLM writes commentary only.
11. Never commit `.env` or keys. Commit `data/processed/`, `cache/llm/`, `demo/replay.json`.
12. Small PRs, feature branch per person, merge at each checkpoint.
