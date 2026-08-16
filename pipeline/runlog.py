"""Run logging: timestamped events.jsonl + report.json per run directory.

This IS the contest's required clear-run log. Log everything: tool calls with
cache hit/live, LLM usage, validation results, fallbacks with reasons, mode flags.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

from pipeline.config import LOGS_DIR


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunLog:
    def __init__(self, run_dir: Path | None = None):
        stamp = datetime.now(timezone.utc).strftime("run-%Y-%m-%dT%H_%M_%S")
        self.dir = run_dir or (LOGS_DIR / stamp)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._events = self.dir / "events.jsonl"

    def event(self, kind: str, **data) -> None:
        with open(self._events, "a") as f:
            f.write(json.dumps({"ts": _now(), "kind": kind, **data}, default=str) + "\n")

    def save_report(self, report: dict) -> Path:
        path = self.dir / "report.json"
        with open(path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        return path
