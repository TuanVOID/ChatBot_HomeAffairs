from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any


class JsonlStructuredLogger:
    """Append-only JSONL logger for observability events."""

    def __init__(self, file_path: Path):
        self._path = Path(file_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, payload: dict[str, Any]) -> None:
        line = json.dumps(payload, ensure_ascii=False)
        with self._lock:
            with self._path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

