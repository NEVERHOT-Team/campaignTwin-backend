CREATE TABLE IF NOT EXISTS runs (
    run_id text PRIMARY KEY,
    status text NOT NULL,
    progress real NOT NULL,
    generation int NOT NULL,
    plan jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS results (
    run_id text NOT NULL REFERENCES runs(run_id),
    strategy_id text NOT NULL,
    scenario text NOT NULL,
    payload jsonb NOT NULL,
    PRIMARY KEY (run_id, strategy_id, scenario)
);

CREATE TABLE IF NOT EXISTS llm_cache (
    hash text PRIMARY KEY,
    model text NOT NULL,
    prompt jsonb NOT NULL,
    response jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);
