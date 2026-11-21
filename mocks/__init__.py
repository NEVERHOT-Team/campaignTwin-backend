"""Load canned API responses for the scaffold."""

import json
from pathlib import Path
from typing import Any


def load_mock(name: str) -> dict[str, Any]:
    path = Path(__file__).with_name(f"{name}.json")
    return json.loads(path.read_text(encoding="utf-8"))
