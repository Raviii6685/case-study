"""Central configuration: paths, flag thresholds, retrieval knobs and LLM settings.

Every tunable lives here so the README can point at one file when explaining
"why this number".
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SHIPMENTS_CSV = ROOT / "shipment_records.csv"
NOTES_CSV = ROOT / "context_notes.csv"
SAMPLE_OUTPUT_CSV = ROOT / "sample_output_format_v2.csv"
OUTPUT_DIR = ROOT / "output"
CACHE_DIR = ROOT / ".cache"

# Output contract -- column names and order copied from sample_output_format_v2.csv.
OUTPUT_COLUMNS = [
    "route",
    "week_of",
    "cost_per_tonne_km",
    "vs_own_history",
    "vs_similar_routes",
    "flagged",
    "matched_note_id",
    "reason",
]

# ---- Baselines (fixed by the brief) -----------------------------------------
BASELINE_WEEKS = 8  # trailing window for "vs own history", strictly before the current week

# ---- Flag rule (data-derived, see README "Choosing the thresholds") ----------
# A week is flagged when cost is rising vs its own history AND it is out of the
# ordinary vs history OR vs peers.
OWN_HISTORY_THRESHOLD_PCT = 15.0  # ~3.5 sd of week-on-history noise (sd 4.2pp)
PEER_THRESHOLD_PCT = 20.0  # ~2.2 sd of peer-gap noise (sd 9.0pp), above the 98th pct

# ---- Retrieval ----------------------------------------------------------------
RETRIEVAL_TOP_K = 3
RETRIEVAL_ALPHA = 0.65  # weight on dense similarity; (1 - alpha) on BM25 keyword score
CANDIDATE_SLACK_WEEKS = 4  # metadata pre-filter: note window must be within +/- 4 weeks
OPEN_ENDED_NOTE_WEEKS = BASELINE_WEEKS  # "starting this week" changes are absorbed by the 8-week baseline
MAGNITUDE_TOLERANCE_PCT = 5.0  # a note stating "5-7%" can explain at most ~12%
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# ---- Q&A agent budgets (Section 8: loop / time / token budgets) ---------------
AGENT_MAX_STEPS = 6
AGENT_MAX_SECONDS = 60.0
AGENT_MAX_TOKENS = 40_000


@dataclass(frozen=True)
class ProviderPreset:
    base_url: str
    model: str
    key_env: str | None
    # Published USD rate per 1M tokens. Verify against the provider's pricing page
    # before quoting -- the token log multiplies these.
    input_per_m: float
    output_per_m: float


PROVIDERS: dict[str, ProviderPreset] = {
    # OpenAI's open-weight gpt-oss-120b on Groq (free tier available). Rates from
    # console.groq.com/docs/models, checked 2026-10-14. llama-3.3-70b-versatile is
    # still listed but returned model_not_found for a free-tier key.
    "groq": ProviderPreset(
        "https://api.groq.com/openai/v1", "openai/gpt-oss-120b", "GROQ_API_KEY", 0.15, 0.60
    ),
    # Fully local, open-source, $0.
    "ollama": ProviderPreset("http://localhost:11434/v1", "qwen2.5:7b-instruct", None, 0.0, 0.0),
    # Any other OpenAI-compatible endpoint; override everything through env vars.
    "openai": ProviderPreset("https://api.openai.com/v1", "gpt-4o-mini", "OPENAI_API_KEY", 0.15, 0.60),
}


@dataclass(frozen=True)
class LLMSettings:
    provider: str = "offline"  # offline | groq | ollama | openai
    base_url: str = ""
    model: str = ""
    api_key: str | None = None
    input_per_m: float = 0.0
    output_per_m: float = 0.0
    temperature: float = 0.0  # every call is deterministic-by-intent; see README "Reproducibility"
    seed: int = 7
    timeout_s: float = 30.0
    max_retries: int = 3
    use_cache: bool = True
    extra: dict = field(default_factory=dict)

    @property
    def enabled(self) -> bool:
        return self.provider != "offline"


def llm_settings_from_env(provider: str | None = None, use_cache: bool = True) -> LLMSettings:
    """Resolve LLM settings. Precedence: explicit arg > LLM_PROVIDER env > auto-detect > offline."""
    provider = provider or os.getenv("LLM_PROVIDER")
    if provider is None:
        provider = "groq" if os.getenv("GROQ_API_KEY") else "offline"
    if provider == "offline":
        return LLMSettings(use_cache=use_cache)
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown LLM provider '{provider}'. Choose from offline, {', '.join(PROVIDERS)}")

    preset = PROVIDERS[provider]
    api_key = os.getenv("LLM_API_KEY") or (os.getenv(preset.key_env) if preset.key_env else None)
    if preset.key_env and not api_key:
        raise ValueError(f"Provider '{provider}' needs {preset.key_env} (or LLM_API_KEY) to be set")
    return LLMSettings(
        provider=provider,
        base_url=os.getenv("LLM_BASE_URL", preset.base_url),
        model=os.getenv("LLM_MODEL", preset.model),
        api_key=api_key,
        input_per_m=float(os.getenv("LLM_INPUT_PER_M", preset.input_per_m)),
        output_per_m=float(os.getenv("LLM_OUTPUT_PER_M", preset.output_per_m)),
        use_cache=use_cache,
    )
