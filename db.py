"""Store mock runs and the LLM cache in JSON or Postgres."""

import json
import os
import re
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Literal

from dotenv import load_dotenv
from filelock import FileLock

ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT / "runs"
CACHE_DIR = ROOT / "cache" / "llm"
load_dotenv(ROOT / ".env")


def get_mode() -> Literal["json", "postgres"]:
    mode = os.getenv("DB_MODE", "json")
    if mode not in ("json", "postgres"):
        raise ValueError("DB_MODE must be json or postgres")
    return mode


def _jsonb(value):
    from psycopg.types.json import Jsonb

    return Jsonb(value)


def _connect():
    import psycopg
    from psycopg.rows import dict_row

    url = os.getenv("DATABASE_URL")
    if not url:
        raise ValueError("DATABASE_URL is required with DB_MODE=postgres")
    return psycopg.connect(url, row_factory=dict_row, connect_timeout=3)


def check_connection() -> None:
    if get_mode() == "postgres":
        try:
            import psycopg
        except ImportError as error:
            raise ConnectionError("Postgres driver unavailable") from error
        try:
            with _connect() as connection:
                connection.execute("SELECT 1")
        except psycopg.Error as error:
            raise ConnectionError("Postgres unavailable") from error


def _safe_key(key: str) -> str:
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,128}", key):
        raise ValueError("Storage key must be a short alphanumeric identifier")
    return key


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, mode="w", encoding="utf-8", delete=False) as handle:
        temp_path = Path(handle.name)
        try:
            json.dump(data, handle, ensure_ascii=False)
            handle.close()
            temp_path.replace(path)
        finally:
            temp_path.unlink(missing_ok=True)


def _run_dir(run_id: str) -> Path:
    return RUNS_DIR / _safe_key(run_id)


def save_run(run_id: str, plan: dict[str, Any]) -> None:
    if get_mode() == "postgres":
        with _connect() as connection:
            connection.execute(
                """INSERT INTO runs (run_id, status, progress, generation, plan)
                   VALUES (%s, 'queued', 0, 0, %s)""",
                (run_id, _jsonb(plan)),
            )
    else:
        _write_json(
            _run_dir(run_id) / "run.json",
            {"run_id": run_id, "status": "queued", "progress": 0, "generation": 0, "plan": plan},
        )


def update_progress(
    run_id: str, progress: float, *, status: str = "running", generation: int = 0
) -> None:
    if get_mode() == "postgres":
        with _connect() as connection:
            cursor = connection.execute(
                "UPDATE runs SET status = %s, progress = %s, generation = %s WHERE run_id = %s",
                (status, progress, generation, run_id),
            )
            if cursor.rowcount == 0:
                raise KeyError(run_id)
    else:
        path = _run_dir(run_id) / "run.json"
        run = _read_json(path)
        if run is None:
            raise KeyError(run_id)
        run.update(status=status, progress=progress, generation=generation)
        _write_json(path, run)


def get_run(run_id: str) -> dict[str, Any] | None:
    if get_mode() == "postgres":
        with _connect() as connection:
            return connection.execute("SELECT * FROM runs WHERE run_id = %s", (run_id,)).fetchone()
    return _read_json(_run_dir(run_id) / "run.json")


def save_result(run_id: str, payload: dict[str, Any]) -> None:
    strategy_id = payload["strategy_id"]
    scenario = payload["scenario"]
    if get_mode() == "postgres":
        with _connect() as connection:
            connection.execute(
                """INSERT INTO results (run_id, strategy_id, scenario, payload)
                   VALUES (%s, %s, %s, %s)
                   ON CONFLICT (run_id, strategy_id, scenario)
                   DO UPDATE SET payload = EXCLUDED.payload""",
                (run_id, strategy_id, scenario, _jsonb(payload)),
            )
    else:
        key = sha256(json.dumps([strategy_id, scenario]).encode()).hexdigest()
        _write_json(_run_dir(run_id) / "results" / f"{key}.json", payload)


def get_results(run_id: str) -> list[dict[str, Any]]:
    if get_mode() == "postgres":
        with _connect() as connection:
            rows = connection.execute(
                "SELECT payload FROM results WHERE run_id = %s ORDER BY scenario, strategy_id",
                (run_id,),
            ).fetchall()
        return [row["payload"] for row in rows]

    results = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (_run_dir(run_id) / "results").glob("*.json")
    ]
    return sorted(results, key=lambda result: (result["scenario"], result["strategy_id"]))


def cache_get(key: str) -> Any:
    if get_mode() == "postgres":
        with _connect() as connection:
            row = connection.execute(
                "SELECT response FROM llm_cache WHERE hash = %s", (key,)
            ).fetchone()
    else:
        row = _read_json(CACHE_DIR / f"{_safe_key(key)}.json")
    return row["response"] if row is not None else None


def cache_put(key: str, model: str, prompt: Any, response: Any) -> None:
    if get_mode() == "postgres":
        with _connect() as connection:
            connection.execute(
                """INSERT INTO llm_cache (hash, model, prompt, response) VALUES (%s, %s, %s, %s)
                   ON CONFLICT (hash) DO UPDATE
                   SET model = EXCLUDED.model, prompt = EXCLUDED.prompt,
                       response = EXCLUDED.response""",
                (key, model, _jsonb(prompt), _jsonb(response)),
            )
    else:
        _write_json(
            CACHE_DIR / f"{_safe_key(key)}.json",
            {"hash": key, "model": model, "prompt": prompt, "response": response},
        )


def export_cache(directory: str | Path) -> int:
    if get_mode() == "postgres":
        with _connect() as connection:
            records = connection.execute("SELECT hash, model, prompt, response FROM llm_cache")
            rows = records.fetchall()
    else:
        rows = [json.loads(path.read_text(encoding="utf-8")) for path in CACHE_DIR.glob("*.json")]

    for row in rows:
        _write_json(Path(directory) / f'{_safe_key(row["hash"])}.json', row)
    return len(rows)


@contextmanager
def v1_transaction():
    """Serialize review and run admission together, committing only on normal exit."""
    empty = {"reviews": {}, "approvals": {}, "runs": {}, "submissions": {}}
    if get_mode() == "postgres":
        with _connect() as connection:
            connection.execute(
                "INSERT INTO rehearsal_workspace (id, state) VALUES (1, %s) "
                "ON CONFLICT (id) DO NOTHING", (_jsonb(empty),)
            )
            state = connection.execute(
                "SELECT state FROM rehearsal_workspace WHERE id = 1 FOR UPDATE"
            ).fetchone()["state"]
            before = json.dumps(state)
            yield state
            if json.dumps(state) != before:
                connection.execute(
                    "UPDATE rehearsal_workspace SET state = %s WHERE id = 1", (_jsonb(state),)
                )
        return

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    path = RUNS_DIR / "v1-workspace.json"
    with FileLock(str(RUNS_DIR / ".v1.lock")):
        state = _read_json(path) or empty
        before = json.dumps(state)
        yield state
        if json.dumps(state) != before:
            _write_json(path, state)
