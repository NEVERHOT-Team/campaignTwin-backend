# CampaignTwin backend

CampaignTwin is a pre-launch e-commerce campaign stress test. This scaffold
serves fixtures so the frontend, API, engine, and data work can start in parallel.
The replay follows three plans over four campaign weeks and two post-campaign
weeks. The deep 11.11 discount leads in S1 and breaks in week 3 in S3. The
targeted loyalty plan stays profitable relative to baseline in every scenario.

## Run locally

From this repo root, use Python 3.11 and an activated virtual environment.

```sh
pip install -e .
uvicorn main:app --reload --host 0.0.0.0 --port 8000
pytest -q
ruff check .
```

JSON storage is the default and needs no Docker or OpenAI key. Open
[OpenAPI docs](http://localhost:8000/docs) for request and response shapes and run:

```sh
curl -fsS http://localhost:8000/api/demo/replay
```

Use the `spec` from `POST /api/brief/parse` as the body for
`POST /api/plan/context`. Send a `SimulationPlan` to `POST /api/runs`,
poll `GET /api/runs/{run_id}` until it reports `done`, and then
request `GET /api/runs/{run_id}/report`. A mock run takes about ten seconds.
The submitted plan is validated and stored. Every legacy run returns the same demo
results while the engine is under development. The replay endpoint is immediate.

## Configure the AI interview

Set `OPENAI_API_KEY` in the backend repository's local `.env`. The backend loads this file at startup and Git ignores it. Keep the key on the server. The browser only calls CampaignTwin's API.

The interview uses `MODEL_FAST`, then `OPENAI_MODEL`, then `gpt-4.1-mini` when those settings are blank. Use a model available to your OpenAI project that supports structured Chat Completions. Restart the backend after changing the environment. Run `python -m agents.llm --check` in the backend environment for an uncached provider check.

## Versioned planner workflow

Use the review ID to resume a session after reloading.

1. POST a brief, structured fields, or both to `/api/v1/campaign-reviews`. The **syntheticRehearsal** example in the [v1 contract](docs/campaigntwin-api-v1.openapi.yaml) includes a complete runnable THB voucher.
2. GET `/api/v1/campaign-reviews/{review_id}` to recover the current facts, candidates, questions and revision. POST a natural-language reply to its `/revisions` endpoint, for example `{"base_revision": 1, "answer": "Our budget is 1000 THB."}`. Read the updated review, check proposed values against their quotes, and confirm their IDs using the new revision number. Structured `edits` remain available for manual corrections. A stale write returns the latest review.
3. POST the exact revision and `"confirmed": true` to its `/approvals` endpoint. A material edit requires another approval.
4. POST the returned `approval_id` and a client-generated UUID in `submission_id` to `/api/v1/runs`. Reuse that submission ID after a lost response. GET the returned `status_url`, then its `report_url`.

For the browser walkthrough, start FastAPI on port 8000. From the sibling `campaignTwin-frontend` repository run `npm ci` and `npm run dev`, then open `http://localhost:5173`. The development server proxies `/api` to FastAPI. A separately hosted frontend can set `VITE_API_BASE_URL` to the backend origin and the backend can list allowed frontend origins in `CORS_ORIGINS`.

Try **“I want to promote toilet tissue.”** Then reply **“Use the synthetic demo for all buyers, target incremental profit, budget 1000 THB, 10 THB voucher, next week.”** Check the proposed values and dates, confirm them, then approve the resulting revision. The interview interprets “next week” against the current Bangkok date. The displayed end date is exclusive.

The runnable catalog contains one explicitly synthetic 12-roll Shopee listing. The local screen compares the submitted offer, a smaller discount and a paired no-campaign baseline. Reviews display the assumed economics and evaluation policy before approval. The screen executes during submission. Commercial seller selections stay blocked until applicable evidence is available.

JSON mode persists sessions in the runs directory. For an existing PostgreSQL deployment, apply the updated `db/init.sql` to create `rehearsal_workspace`. Approval records use the trusted local workspace actor from `CAMPAIGNTWIN_PLANNER_ID` (default `local-planner`).

## CampaignSpec v0.1

The brief/context/run endpoints now use the nested CampaignSpec from the team's
draft: brand, objective, audience, offer, budget, channels, timing, constraints,
and per-field provenance. See [the contract and migration guide](docs/campaign-spec.md).
The fixture example is `mocks/campaign_spec.json`; the published contract is
`/openapi.json`. A run requires a spec with `status: "ready"` and no blocking
questions. The parser and context endpoints still return synthetic fixtures.

On Windows, use a Python 3.11+ virtual environment. If the default `python`
points to an older interpreter, create `.venv` with an explicit supported Python
path. Once it exists, use `.venv/Scripts/python.exe -m pip install -e .`,
`.venv/Scripts/python.exe -m uvicorn main:app --reload --port 8000`,
`.venv/Scripts/python.exe -m pytest -q`, and
`.venv/Scripts/python.exe -m ruff check .`.

## Use Postgres

Copy `.env.example` to `.env` and set `POSTGRES_PASSWORD`. In `.env`, set
`DB_MODE=postgres` and set `DATABASE_URL` to match the password and host.
Use `localhost` for your own Compose service or Gan's LAN host for the shared
run and cache database.

```sh
docker compose up -d
uvicorn main:app --reload --host 0.0.0.0 --port 8000
curl -fsS http://localhost:8000/api/health
```

The health response should be `{"ok":true,"db_mode":"postgres"}`. The static
fixtures remain local files in both storage modes. See [data fixtures](data/README.md)
for the loader interface and units.

## Ownership

- Big owns `api/`, `agents/`, and the Pydantic wire contracts in `contracts/`.
  FastAPI publishes those contracts at `/openapi.json`.
- Mic owns `engine/`.
- Gan owns `data/`, `db/`, and `db.py`.
- Phee owns the frontend in the separate CampaignTwin frontend repo.
