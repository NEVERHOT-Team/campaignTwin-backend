CREATE TABLE IF NOT EXISTS rehearsal_workspace_v2 (
    id integer PRIMARY KEY CHECK (id = 1),
    state jsonb NOT NULL
);
