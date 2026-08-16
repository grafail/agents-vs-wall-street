"""Read-through disk cache for every external fetch (tool belt + consensus).

Pre-baking == warming this cache during the day; the final run mostly hits it
but can still fetch live. Every call is logged hit/live by the caller via runlog.
"""
import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Callable

from pipeline.config import CACHE_DIR


def _key(tool: str, args: dict) -> str:
    blob = json.dumps([tool, args], sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:24]


def read_through(tool: str, args: dict, fetch: Callable[[], Any], refresh: bool = False) -> tuple[Any, bool]:
    """Returns (result, was_cache_hit). Cache entries carry fetched_at + args for audit."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{tool}-{_key(tool, args)}.json"
    if path.exists() and not refresh:
        with open(path) as f:
            return json.load(f)["result"], True
    result = fetch()
    with open(path, "w") as f:
        json.dump({
            "tool": tool, "args": args,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "result": result,
        }, f, indent=1, default=str)
    return result, False
