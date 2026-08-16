"""Central config: paths + settings. No other module reads os.environ directly."""
import json
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]

CHALLENGE_DIR = ROOT / "challenge"
OFFLINE_DATA = CHALLENGE_DIR / "offline-data"
TEMPLATES_DIR = CHALLENGE_DIR / "templates"
ARTIFACTS_DIR = ROOT / "artifacts"          # committed research record
CACHE_DIR = ARTIFACTS_DIR / "cache"         # read-through tool cache (git-ignored)
FACTS_DIR = ARTIFACTS_DIR / "facts"         # extracted facts per company (committed)
LOGS_DIR = ROOT / "logs"
SUBMISSION_DIR = ROOT / "submission"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    llm_provider: Literal["openrouter", "openai"] = "openrouter"
    openrouter_api_key: str = ""
    openai_api_key: str = ""
    model_big: str = ""     # estimator + reconciler (pinned per run)
    model_small: str = ""   # extraction + vibe labeling
    enable_reconcile: bool = True
    enable_research: bool = True    # live research agent (news/peers digest) in the graph
    small_reasoning_effort: str = "low"   # OpenRouter reasoning effort for the small role ("" disables)
    extract_workers: int = 4              # parallel LLM parse batches during extraction
    price_in_per_m: float = 0.0           # USD per 1M input tokens (blended; 0 = unknown, cost shows —)
    price_out_per_m: float = 0.0          # USD per 1M output tokens
    openrouter_provider: str = ""         # pin ONE backend provider (cache continuity!); "" = no pin
                                          # e.g. "deepseek" for deepseek models, "openai" for gpt models
    estimator_panel: str = ""             # cross-provider estimator panel: comma-separated
                                          # "provider:model" entries (provider = openrouter|openai),
                                          # e.g. "openrouter:deepseek/deepseek-chat,openai:gpt-5".
                                          # "" = single call with llm_provider + MODEL_BIG (current behavior)

    @property
    def api_key(self) -> str:
        return self.openrouter_api_key if self.llm_provider == "openrouter" else self.openai_api_key

    @property
    def base_url(self) -> str | None:
        return "https://openrouter.ai/api/v1" if self.llm_provider == "openrouter" else None


@lru_cache
def settings() -> Settings:
    return Settings()


def load_companies() -> list[dict]:
    """The contest spec: 4 companies x 3 metrics, exact labels/units/output files."""
    with open(CHALLENGE_DIR / "companies.json") as f:
        return json.load(f)["companies"]
