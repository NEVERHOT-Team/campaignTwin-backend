# CampaignTwin backend

CampaignTwin is a pre-launch e-commerce campaign stress test. This scaffold
serves fixtures so the frontend, API, engine, and data work can start in parallel.
The replay follows three plans over four campaign weeks and two post-campaign
weeks. The deep 11.11 discount leads in S1 and breaks in week 3 in S3. The
targeted loyalty plan stays profitable relative to baseline in every scenario.

## Run locally

From this repo root, use Python 3.11 and an activated virtual environment.

```sh
pip install -e ".[dev]"
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

## Deploy

### Render

1. In the Render Dashboard, select **New → Blueprint**, connect this backend repository,
   select `dev`, and use the root `render.yaml`. It defines one free Docker web service.
   [Render Blueprint setup](https://render.com/docs/infrastructure-as-code)
2. Supply `OPENAI_API_KEY` in Render's secret prompt for live model calls. The app also
   runs with the key unset, using cached responses and local fallbacks. Select
   **Deploy Blueprint**.
3. Copy the service's HTTPS URL from Render and check both endpoints. The Docker
   image serves the existing planner and the v2 seller API on the same port.
   The health endpoint reports the database mode. Render checks the seller
   discovery endpoint, which returns the v2 version and review URL.

   ```sh
   BACKEND_URL=https://your-service.onrender.com
   curl -fsS "$BACKEND_URL/api/health"
   curl -fsS "$BACKEND_URL/api/v2"
   ```

4. Set the frontend's `VITE_API_BASE_URL` to that service origin and rebuild the frontend.
   Local development on port 5173 and HTTPS Vercel previews are already allowed. For a
   custom frontend domain, set `CORS_ORIGINS` on the backend to a comma-separated list of
   exact origins, such as `https://planner.example.com,https://demo.example.com`.

Render builds from the Git checkout. Docker includes the processed data, demo and LLM
cache, plus any calibration files present in the build context. The local calibration
panel is Git-ignored, so a Render build starts with an empty `data/calib/`.
With `DB_MODE=json`, reviews, approvals, runs and new cache entries are lost when the
free instance restarts, redeploys or spins down. [Render free service limits](https://render.com/docs/free)

### Cloudflare Tunnel fallback

With Docker and `cloudflared` installed, start the backend from the repository root.

```sh
docker build -t campaigntwin-backend .
docker run --rm -p 127.0.0.1:8000:8000 -e DB_MODE=json campaigntwin-backend
```

In a second terminal, confirm the health response and start a temporary public tunnel.

```sh
curl -fsS http://127.0.0.1:8000/api/health
cloudflared tunnel --url http://127.0.0.1:8000
```

Use the printed `https://…trycloudflare.com` origin as `VITE_API_BASE_URL` and rebuild
the frontend. Keep both processes running for the demo. Each new tunnel gets its own
URL. [Cloudflare Quick Tunnels](https://developers.cloudflare.com/tunnel/get-started/quick-tunnels/)

## Configure the AI interview

Set `OPENAI_API_KEY` in the backend repository's local `.env`. The backend loads this file at startup and Git ignores it. Keep the key on the server. The browser only calls CampaignTwin's API.

The interview uses `MODEL_FAST`, then `OPENAI_MODEL`, then `gpt-4.1-mini` when those settings are blank. Use a model available to your OpenAI project that supports structured Chat Completions. Restart the backend after changing the environment. Run `python -m agents.llm --check` in the backend environment for an uncached provider check.

## Optional seller interview API

A separate entry point serves a guided seller interview, voucher economics and a seeded
comparison at `/api/v2`. To try it alongside the existing server, run
`DB_MODE=json python -m uvicorn api.v2.app:app --host 127.0.0.1 --port 8002`.
See the [seller walkthrough](docs/seller-interview-v2.md) and
[static v2 OpenAPI contract](docs/campaigntwin-api-v2.openapi.yaml).

## Versioned planner workflow

Use the review ID to resume a session after reloading.

1. POST a brief, structured fields, or both to `/api/v1/campaign-reviews`. The **syntheticRehearsal** example in the [v1 contract](docs/campaigntwin-api-v1.openapi.yaml) includes a complete runnable THB voucher.
2. GET `/api/v1/campaign-reviews/{review_id}` to recover the current facts, candidates, questions and revision. POST a natural-language reply to its `/revisions` endpoint, for example `{"base_revision": 1, "answer": "Our budget is 1000 THB."}`. Read the updated review, check proposed values against their quotes, and confirm their IDs using the new revision number. Structured `edits` remain available for manual corrections. A stale write returns the latest review.
3. POST the exact revision and `"confirmed": true` to its `/approvals` endpoint. A material edit requires another approval.
4. POST the returned `approval_id` and a client-generated UUID in `submission_id` to `/api/v1/runs`. Reuse that submission ID after a lost response. GET the returned `status_url`, then its `report_url`.

5. GET `/api/v1/runs/{run_id}/input` for the exact approved review passed to the evaluator, including planner facts, their origins, resolved offer and costs, aggregate audience, and evaluation policy. This snapshot remains available after another edit or a failed run.
6. GET `/api/v1/runs/{run_id}/evaluation` after completion to compare the submitted offer, smaller discount and paired no-campaign baseline. Each decision check shows the metric, observed value, comparison rule, approved threshold when applicable, and outcome. Use the report for each plan’s exact terms and seeds.
7. GET `/api/v1/runs/{run_id}/historical-comparison` to see the held-out observation pinned at submission. A matching engine prediction uses the published `HistoricalPrediction` schema. Until a producer supplies one, a verified reference returns `reference_only` with `prediction_missing`.

The approved `synthetic-tissue-v1` policy calls `engine.synthetic.evaluate(run_id, approval_id, review)`. The shared business evaluator on `dev`, `engine.evaluator.evaluate_plan`, takes paired plan and baseline weekly shopper statistics keyed by seed, strategy, scenario, market costs, budget and target. Reusing it for a future policy requires a simulation that produces those weekly inputs under the approved offer and budget. The current report stores final percentiles. Its subsidy leakage uses actual overlap in the paired daily draws, while the weekly evaluator estimates overlap from unit totals.

For historical scoring, `data.backtest.evaluate_backtest` consumes a predicted control-adjusted sales lift in percentage points and reads the current released case. The HTTP adapter applies the same inclusive range check to the version pinned for the run. Historical scoring requires a separately identified prediction for the same campaign, population, control protocol, fit cutoff and windows.

For example, the published synthetic rehearsal with a 10 THB voucher and a 1,000 THB budget returns **hold**. Across the eight paired seeds, the submitted offer has median incremental profit of **−600 THB** against the **0 THB** minimum. All eight seeds lose money, exceeding the approved **25%** loss-risk ceiling. A budget below the cost of a single voucher produces **insufficient evidence** because subsidy leakage cannot be measured. These results describe the synthetic S1 scenario.

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
path. Once it exists, use `.venv/Scripts/python.exe -m pip install -e ".[dev]"`,
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
