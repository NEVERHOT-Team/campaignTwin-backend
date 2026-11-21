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
The submitted plan is validated and stored. Every run returns the same demo
results while the engine is under development. The replay endpoint is immediate.

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
