"""Make `pipeline` importable when the project is not installed as a package."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import pytest


@pytest.fixture(autouse=True)
def _env_isolated(monkeypatch):
    """Tests must never inherit live-run config from .env: dummy keys fail fast
    and identify any test that reaches for a real endpoint."""
    from pipeline import config
    monkeypatch.setenv("ESTIMATOR_PANEL", "")
    monkeypatch.setenv("OPENROUTER_PROVIDER", "")
    monkeypatch.setenv("PRICE_IN_PER_M", "0")
    monkeypatch.setenv("PRICE_OUT_PER_M", "0")
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-never-real")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-never-real")
    monkeypatch.setenv("MODEL_BIG", "test/model-big")
    monkeypatch.setenv("MODEL_SMALL", "test/model-small")
    config.settings.cache_clear()
    yield
    config.settings.cache_clear()
