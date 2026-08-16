"""Run logging: timestamped events.jsonl + report.json per run directory.

This IS the contest's required clear-run log. Log everything: tool calls with
cache hit/live, LLM usage, validation results, fallbacks with reasons, mode flags.
"""
import json
import threading
from datetime import datetime, timezone
from pathlib import Path

from pipeline.config import LOGS_DIR


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunLog:
    def __init__(self, run_dir: Path | None = None, echo: bool = True):
        stamp = datetime.now(timezone.utc).strftime("run-%Y-%m-%dT%H_%M_%S")
        self.dir = run_dir or (LOGS_DIR / stamp)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._events = self.dir / "events.jsonl"
        self.echo = echo
        self._lock = threading.Lock()  # events may arrive from worker threads

    def event(self, kind: str, **data) -> None:
        with self._lock, open(self._events, "a") as f:
            f.write(json.dumps({"ts": _now(), "kind": kind, **data}, default=str) + "\n")
        if self.echo:
            brief = " ".join(
                f"{k}={self._fmt(v)}" for k, v in data.items()
                if k not in ("quote", "text", "excerpt", "messages") and v is not None
            )
            print(f"[{_now()[11:19]}] {kind:<24} {brief[:200]}", flush=True)

    @staticmethod
    def _fmt(v) -> str:
        s = str(v)
        return s if len(s) <= 60 else s[:57] + "..."

    def save_report(self, report: dict) -> Path:
        path = self.dir / "report.json"
        with open(path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        return path
