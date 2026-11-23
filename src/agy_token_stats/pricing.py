"""Approximate Gemini API token pricing catalog and per-step cost estimation.

Pricing figures follow Google AI Studio / Gemini API Paid Tier rates (USD per 1M
tokens) as documented at https://aistudio.google.com/docs/choosing-a-model and
https://ai.google.dev/gemini-api/docs/pricing:
    - Uncached input tokens are billed at the model's input rate.
    - Cached input tokens (`cache_read_tokens`) are billed at the model's context
      cache read rate (typically 90%-95% discounted vs uncached input).
    - Output tokens (`output_tokens`) already include internal thinking tokens and
      visible response/tool-call tokens, matching Gemini's combined output billing.

Invariants & Purity:
    - All lookup and estimation functions are pure and thread-safe.
    - Unrecognized or non-Gemini models (including `"Unknown"`) return `None` rather
      than guessing rates.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agy_token_stats.models import StepRecord

_MODEL_PREFIX_RE = re.compile(r"^models/")
_PAREN_QUALIFIER_RE = re.compile(r"\s*\([^)]*\)\s*")
_NON_ALNUM_DOT_RE = re.compile(r"[^a-z0-9.]+")
_TRAILING_VARIANT_RE = re.compile(
    r"(?:\s+(?:preview|latest|exp|experimental|high|medium|low|minimal|xhigh|thinking))+$"
)

GEMINI_4_INTRO_END_UTC = datetime(2026, 12, 1, 0, 0, 0, tzinfo=timezone.utc)
GEMINI_3_FLASH_INTRO_END_UTC = datetime(2027, 1, 1, 0, 0, 0, tzinfo=timezone.utc)


@dataclass(frozen=True)
class ModelPricing:
    """Per-1M-token USD pricing rule for a Gemini model tier."""

    model_key: str
    display_name: str
    input_per_1m: float
    cache_read_per_1m: float
    output_per_1m: float
    long_context_threshold: int | None = None
    long_input_per_1m: float | None = None
    long_cache_read_per_1m: float | None = None
    long_output_per_1m: float | None = None
    note: str = ""


@dataclass(frozen=True)
class StepCost:
    """Estimated USD cost components and prompt-cache savings for a single step."""

    input_cost_usd: float
    cache_cost_usd: float
    output_cost_usd: float
    cache_savings_usd: float

    @property
    def total_cost_usd(self) -> float:
        """Total estimated USD cost for this step."""
        return self.input_cost_usd + self.cache_cost_usd + self.output_cost_usd


# Static pricing rules for models without date-based introductory pricing transitions.
_STATIC_PRICING: tuple[tuple[str, ModelPricing], ...] = (
    (
        "gemini 4 hydrogen",
        ModelPricing(
            model_key="gemini-4-hydrogen",
            display_name="Gemini 4 Hydrogen",
            input_per_1m=1.50,
            cache_read_per_1m=0.075,
            output_per_1m=7.50,
        ),
    ),
    (
        "gemini 4 neon",
        ModelPricing(
            model_key="gemini-4-neon",
            display_name="Gemini 4 Neon",
            input_per_1m=3.00,
            cache_read_per_1m=0.15,
            output_per_1m=15.00,
        ),
    ),
    (
        "gemini 3.5 flash lite",
        ModelPricing(
            model_key="gemini-3.5-flash-lite",
            display_name="Gemini 3.5 Flash-Lite",
            input_per_1m=0.30,
            cache_read_per_1m=0.03,
            output_per_1m=2.50,
        ),
    ),
    (
        "gemini 3.5 flash",
        ModelPricing(
            model_key="gemini-3.5-flash",
            display_name="Gemini 3.5 Flash",
            input_per_1m=1.50,
            cache_read_per_1m=0.15,
            output_per_1m=9.00,
        ),
    ),
    (
        "gemini 3.1 flash lite",
        ModelPricing(
            model_key="gemini-3.1-flash-lite",
            display_name="Gemini 3.1 Flash-Lite",
            input_per_1m=0.25,
            cache_read_per_1m=0.025,
            output_per_1m=1.50,
        ),
    ),
    (
        "gemini 3.1 pro",
        ModelPricing(
            model_key="gemini-3.1-pro",
            display_name="Gemini 3.1 Pro",
            input_per_1m=2.00,
            cache_read_per_1m=0.20,
            output_per_1m=12.00,
            long_context_threshold=200_000,
            long_input_per_1m=4.00,
            long_cache_read_per_1m=0.40,
            long_output_per_1m=18.00,
            note="<=200k / >200k tiered context",
        ),
    ),
    (
        "gemini 3 pro",
        ModelPricing(
            model_key="gemini-3-pro",
            display_name="Gemini 3 Pro",
            input_per_1m=2.00,
            cache_read_per_1m=0.20,
            output_per_1m=12.00,
            long_context_threshold=200_000,
            long_input_per_1m=4.00,
            long_cache_read_per_1m=0.40,
            long_output_per_1m=18.00,
            note="<=200k / >200k tiered context",
        ),
    ),
    (
        "gemini 3 flash",
        ModelPricing(
            model_key="gemini-3-flash",
            display_name="Gemini 3 Flash",
            input_per_1m=0.50,
            cache_read_per_1m=0.05,
            output_per_1m=3.00,
        ),
    ),
    (
        "gemini 2.5 flash lite",
        ModelPricing(
            model_key="gemini-2.5-flash-lite",
            display_name="Gemini 2.5 Flash-Lite",
            input_per_1m=0.10,
            cache_read_per_1m=0.01,
            output_per_1m=0.40,
        ),
    ),
    (
        "gemini 2.5 flash",
        ModelPricing(
            model_key="gemini-2.5-flash",
            display_name="Gemini 2.5 Flash",
            input_per_1m=0.30,
            cache_read_per_1m=0.03,
            output_per_1m=2.50,
        ),
    ),
    (
        "gemini 2.5 pro",
        ModelPricing(
            model_key="gemini-2.5-pro",
            display_name="Gemini 2.5 Pro",
            input_per_1m=1.25,
            cache_read_per_1m=0.125,
            output_per_1m=10.00,
            long_context_threshold=200_000,
            long_input_per_1m=2.50,
            long_cache_read_per_1m=0.25,
            long_output_per_1m=15.00,
            note="<=200k / >200k tiered context",
        ),
    ),
    (
        "gemini 2.0 flash lite",
        ModelPricing(
            model_key="gemini-2.0-flash-lite",
            display_name="Gemini 2.0 Flash-Lite",
            input_per_1m=0.075,
            cache_read_per_1m=0.0075,
            output_per_1m=0.30,
        ),
    ),
    (
        "gemini 2.0 flash",
        ModelPricing(
            model_key="gemini-2.0-flash",
            display_name="Gemini 2.0 Flash",
            input_per_1m=0.10,
            cache_read_per_1m=0.025,
            output_per_1m=0.40,
        ),
    ),
)


def normalize_model_name(model_name: str) -> str:
    """Normalize a display or API model name into lowercase space-delimited tokens."""
    text = (model_name or "").strip().lower()
    if not text:
        return ""
    text = _MODEL_PREFIX_RE.sub("", text)
    text = _PAREN_QUALIFIER_RE.sub(" ", text)
    text = _NON_ALNUM_DOT_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = _TRAILING_VARIANT_RE.sub("", text).strip()
    # Normalize "gemini 3.0 ..." -> "gemini 3 ..." and "gemini 4.0 ..." -> "gemini 4 ..."
    text = re.sub(r"\bgemini (\d+)\.0\b", r"gemini \1", text)
    return text


def _is_before_cutoff(at_utc: datetime | None, cutoff_utc: datetime) -> bool:
    """Return True only when `at_utc` is given and strictly precedes `cutoff_utc`."""
    if at_utc is None:
        return False
    aware = at_utc if at_utc.tzinfo is not None else at_utc.replace(tzinfo=timezone.utc)
    return aware.astimezone(timezone.utc) < cutoff_utc


def resolve_model_pricing(
    model_name: str,
    *,
    at_utc: datetime | None = None,
) -> ModelPricing | None:
    """Look up approximate Gemini Paid Tier pricing for `model_name`.

    When `at_utc` is provided, models with scheduled introductory-to-standard
    transitions (Gemini 4 Argon/Carbon through 2026-11-30 UTC; Gemini 3.6/3.7/3.8
    Flash through 2026-12-31 UTC) select the rate active at that timestamp.
    When `at_utc` is `None`, standard post-intro rates are returned.

    Returns `None` for `"Unknown"` or non-Gemini models.
    """
    norm = normalize_model_name(model_name)
    if not norm or "gemini" not in norm:
        return None

    # Gemini 4 Argon & Carbon (introductory through Nov 30, 2026 UTC; standard from Dec 1, 2026 UTC)
    for element in ("argon", "carbon"):
        prefix = f"gemini 4 {element}"
        if norm == prefix or norm.startswith(prefix + " "):
            if _is_before_cutoff(at_utc, GEMINI_4_INTRO_END_UTC):
                return ModelPricing(
                    model_key=f"gemini-4-{element}",
                    display_name=f"Gemini 4 {element.capitalize()}",
                    input_per_1m=2.00,
                    cache_read_per_1m=0.10,
                    output_per_1m=10.00,
                    note="introductory rate (through 2026-11-30 UTC)",
                )
            return ModelPricing(
                model_key=f"gemini-4-{element}",
                display_name=f"Gemini 4 {element.capitalize()}",
                input_per_1m=4.00,
                cache_read_per_1m=0.20,
                output_per_1m=20.00,
                note="standard rate (from 2026-12-01 UTC)",
            )

    # Gemini 3.6 / 3.7 / 3.8 Flash (introductory through Dec 31, 2026 UTC; standard from Jan 1, 2027 UTC)
    for minor in ("3.8", "3.7", "3.6"):
        prefix = f"gemini {minor} flash"
        if norm == prefix or norm.startswith(prefix + " "):
            if _is_before_cutoff(at_utc, GEMINI_3_FLASH_INTRO_END_UTC):
                return ModelPricing(
                    model_key=f"gemini-{minor}-flash",
                    display_name=f"Gemini {minor} Flash",
                    input_per_1m=0.75,
                    cache_read_per_1m=0.075,
                    output_per_1m=3.75,
                    note="introductory rate (through 2026-12-31 UTC)",
                )
            return ModelPricing(
                model_key=f"gemini-{minor}-flash",
                display_name=f"Gemini {minor} Flash",
                input_per_1m=1.50,
                cache_read_per_1m=0.15,
                output_per_1m=7.50,
                note="standard rate (from 2027-01-01 UTC)",
            )

    for prefix, pricing in _STATIC_PRICING:
        if norm == prefix or norm.startswith(prefix + " "):
            return pricing

    return None


def estimate_step_cost(step: StepRecord, model_name: str) -> StepCost | None:
    """Estimate USD cost and prompt-cache savings for a `PLANNER_RESPONSE` step.

    Returns `None` if `model_name` does not map to a known Gemini pricing rule.
    """
    pricing = resolve_model_pricing(model_name, at_utc=step.created_at_utc)
    if pricing is None:
        return None

    gross_input = step.input_tokens + step.cache_read_tokens
    use_long = (
        pricing.long_context_threshold is not None
        and gross_input > pricing.long_context_threshold
    )

    in_rate = (
        pricing.long_input_per_1m
        if use_long and pricing.long_input_per_1m is not None
        else pricing.input_per_1m
    )
    cache_rate = (
        pricing.long_cache_read_per_1m
        if use_long and pricing.long_cache_read_per_1m is not None
        else pricing.cache_read_per_1m
    )
    out_rate = (
        pricing.long_output_per_1m
        if use_long and pricing.long_output_per_1m is not None
        else pricing.output_per_1m
    )

    in_cost = (step.input_tokens / 1_000_000.0) * in_rate
    cache_cost = (step.cache_read_tokens / 1_000_000.0) * cache_rate
    out_cost = (step.output_tokens / 1_000_000.0) * out_rate
    cache_savings = (step.cache_read_tokens / 1_000_000.0) * max(
        0.0, in_rate - cache_rate
    )
    return StepCost(
        input_cost_usd=in_cost,
        cache_cost_usd=cache_cost,
        output_cost_usd=out_cost,
        cache_savings_usd=cache_savings,
    )
