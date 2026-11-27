"""Anchored weekly quota estimates based on locally discovered Antigravity logs.

This module intentionally models an entered reset boundary as a recurring, fixed
168-hour schedule. It is not Google's rolling per-request quota meter.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone, tzinfo
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any

from agy_token_stats.models import StatsReport

WEEK = timedelta(hours=168)
DEFAULT_LIMIT_USD = 200.0
_ISO_WITH_OFFSET = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2}(?:\.\d{1,6})?)?(?:Z|[+-]\d{2}:\d{2})$"
)
_DURATION = re.compile(r"^(?:(\d+)d)?(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?$")
_BANNER = re.compile(
    r"^You've reached your individual weekly usage limit set by your administrator\. "
    r"It resets in ((?:\d+d)?(?:\d+h)?(?:\d+m)?(?:\d+s)?)\. "
    r"Contact your administrator if you need a higher limit\.$"
)
_STATE_FIELDS = {"version", "anchor_utc", "origin"}


@dataclass(frozen=True)
class QuotaState:
    """Immutable aware-UTC reset anchor and its non-empty provenance label.

    Instances contain no mutable shared state; callers must serialize updates
    through :func:`save_quota_state` if multiple threads or processes may write.
    """

    anchor_utc: datetime
    origin: str


def default_quota_path() -> Path:
    """Return the XDG quota file path, falling back to ``~/.config``."""
    configured = os.environ.get("XDG_CONFIG_HOME")
    base = Path(configured).expanduser() if configured else Path.home() / ".config"
    if not base.is_absolute():
        base = Path.home() / ".config"
    return base / "agy-token-stats" / "quota.json"


def parse_quota_limit(value: str) -> float:
    """Parse a positive, finite per-run USD budget override."""
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("quota limit must be a positive finite USD amount") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError("quota limit must be a positive finite USD amount")
    return parsed


def _parse_duration(value: str) -> timedelta:
    match = _DURATION.fullmatch(value)
    if not match or not any(part is not None for part in match.groups()):
        raise ValueError("countdown must use compact d/h/m/s units, e.g. 1h30m19s")
    days, hours, minutes, seconds = (int(part or 0) for part in match.groups())
    duration = timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)
    if duration <= timedelta(0):
        raise ValueError("countdown must be greater than zero")
    if duration > WEEK:
        raise ValueError("countdown cannot exceed 168 hours")
    return duration


def parse_reset_input(value: str, *, entered_at: datetime) -> tuple[datetime, str]:
    """Parse an offset timestamp or supported reset countdown into a UTC boundary.

    Accepted countdowns are compact d/h/m/s values and the exact observed
    Antigravity weekly-limit banner. Arbitrary prose and timezone-naive dates
    are rejected. Countdown durations are measured from ``entered_at``; only
    the full banner records ``manual_limit_notice`` evidence.
    """
    entered_utc = _as_utc(entered_at)
    clean = value.strip()
    banner = _BANNER.fullmatch(clean)
    if banner:
        duration_text = banner.group(1)
        duration = _parse_duration(duration_text)
        try:
            return entered_utc + duration, "manual_limit_notice"
        except OverflowError as exc:
            raise ValueError("countdown produces an unrepresentable reset timestamp") from exc
    if _DURATION.fullmatch(clean):
        duration = _parse_duration(clean)
        try:
            return entered_utc + duration, "manual_countdown"
        except OverflowError as exc:
            raise ValueError("countdown produces an unrepresentable reset timestamp") from exc
    if not _ISO_WITH_OFFSET.fullmatch(clean):
        raise ValueError(
            "reset must be an ISO-8601 date-time with Z/explicit offset, "
            "a compact countdown (e.g. 1h30m19s), or the full reset banner"
        )
    try:
        parsed = datetime.fromisoformat(clean[:-1] + "+00:00" if clean.endswith("Z") else clean)
        boundary = parsed.astimezone(timezone.utc)
        max_boundary = entered_utc + WEEK
    except OverflowError as exc:
        raise ValueError("reset timestamp is outside the supported datetime range") from exc
    except ValueError as exc:
        raise ValueError("reset timestamp is not a valid ISO-8601 date-time") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("reset timestamp must include Z or an explicit UTC offset")
    if boundary > max_boundary:
        raise ValueError("reset timestamp cannot be more than 168 hours in the future")
    return boundary, "manual_timestamp"


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("quota clock must be timezone-aware")
    return value.astimezone(timezone.utc)


def _rounded_finite(value: float, places: int) -> float | None:
    """Round a report number, mapping overflow/non-finite derived values to null."""
    return round(value, places) if math.isfinite(value) else None


def load_quota_state(path: Path) -> QuotaState | None:
    """Read and validate one state snapshot; return ``None`` when it is missing.

    This function never writes. Atomic replacements are safe to read, but there
    is no lock against concurrent writers. Malformed/unreadable files raise
    ``ValueError`` naming the path and the ``--quota-reset`` recovery option.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise ValueError(
            f"cannot read quota state at {path}: {exc}; use --quota-reset to replace it"
        ) from exc
    except UnicodeError as exc:
        raise ValueError(
            f"cannot decode quota state at {path}: {exc}; use --quota-reset to replace it"
        ) from exc
    try:
        payload = json.loads(raw)
        if not isinstance(payload, dict) or set(payload) != _STATE_FIELDS:
            raise ValueError("state must contain exactly version, anchor_utc, and origin")
        if type(payload["version"]) is not int or payload["version"] != 1:
            raise ValueError("unsupported state version (expected 1)")
        if not isinstance(payload["anchor_utc"], str) or not isinstance(payload["origin"], str):
            raise ValueError("anchor_utc and origin must be strings")
        anchor_text = payload["anchor_utc"]
        if not _ISO_WITH_OFFSET.fullmatch(anchor_text):
            raise ValueError("anchor_utc must be an ISO-8601 date-time with offset")
        anchor = datetime.fromisoformat(
            anchor_text[:-1] + "+00:00" if anchor_text.endswith("Z") else anchor_text
        )
        if anchor.utcoffset() != timedelta(0) or not payload["origin"].strip():
            raise ValueError("anchor_utc must be UTC and origin must be non-empty")
        return QuotaState(anchor.astimezone(timezone.utc), payload["origin"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"malformed quota state at {path}: {exc}; use --quota-reset to replace it"
        ) from exc


def validate_quota_state(state: QuotaState, *, path: Path, now_utc: datetime) -> None:
    """Reject a saved boundary beyond the supported 168-hour horizon.

    The horizon is checked at report time (after any interactive entry). The
    state file is never rewritten by this validation.
    """
    now = _as_utc(now_utc)
    try:
        latest = now + WEEK
    except OverflowError:
        latest = datetime.max.replace(tzinfo=timezone.utc)
    if state.anchor_utc > latest:
        raise ValueError(
            f"quota state at {path} has a reset more than 168 hours in the future; "
            "use --quota-reset to replace it"
        )


def save_quota_state(path: Path, state: QuotaState) -> None:
    """Atomically replace state via a same-directory temporary file.

    Concurrent writers are not locked; the last successful replacement wins.
    Filesystem failures raise ``ValueError`` with the target path.
    """
    payload = {
        "version": 1,
        "anchor_utc": _as_utc(state.anchor_utc).isoformat().replace("+00:00", "Z"),
        "origin": state.origin,
    }
    temp_name: str | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.",
            suffix=".tmp", delete=False,
        ) as temp:
            temp_name = temp.name
            json.dump(payload, temp, separators=(",", ":"))
            temp.write("\n")
            temp.flush()
            os.fsync(temp.fileno())
        os.replace(temp_name, path)
    except OSError as exc:
        if temp_name is not None:
            try:
                os.unlink(temp_name)
            except OSError:
                pass
        raise ValueError(f"cannot write quota state at {path}: {exc}") from exc


def next_reset_boundary(anchor_utc: datetime, now_utc: datetime) -> tuple[datetime, bool]:
    """Return the next strictly-future 168-hour boundary and whether it was projected."""
    anchor = _as_utc(anchor_utc)
    now = _as_utc(now_utc)
    if anchor > now:
        return anchor, False
    periods = int((now - anchor) // WEEK) + 1
    return anchor + periods * WEEK, True


def build_quota_report(
    stats: StatsReport | None,
    *,
    now_utc: datetime,
    local_tz: tzinfo | None,
    state: QuotaState | None,
    limit_usd: float = DEFAULT_LIMIT_USD,
    sources_scanned: Sequence[str] | None = None,
    sources_unavailable: Sequence[str] = (),
) -> dict[str, Any]:
    """Build the dedicated quota response from weekly stats or unknown state."""
    now = _as_utc(now_utc)
    if sources_scanned is not None:
        sources = list(sources_scanned)
    elif stats is not None:
        sources = stats.sources_scanned
    else:
        sources = []
    unavailable = list(sources_unavailable)
    coverage_note = (
        "Only discovered local logs and configured sources are included; "
        "--source and --brain-dir filters can narrow coverage. Root availability "
        "does not guarantee every nested transcript was readable."
    )
    if unavailable:
        coverage_note += (
            f" Unavailable source roots: {', '.join(unavailable)}; estimates use "
            "only scanned roots."
        )
    result: dict[str, Any] = {
        "quota_status": (
            "blocked"
            if state is not None
            and state.origin in ("manual_limit_notice", "manual_blocked")
            and now < state.anchor_utc
            else "unverified"
        ),
        "reset_local": None,
        "cycle_start_local": None,
        "seconds_until_reset": None,
        "reset_source": state.origin if state else None,
        "reset_projected": None,
        "limit_usd": limit_usd,
        "estimated_used_usd": None,
        "estimated_remaining_usd": None,
        "percent_used": None,
        "priced_calls": None,
        "unpriced_calls": None,
        "llm_calls": None,
        "estimated_token_steps": None,
        "average_usd_per_day": None,
        "projected_used_at_reset_usd": None,
        "sustainable_remaining_usd_per_day": None,
        "estimated_limit_hit_local": None,
        "sources_scanned": list(sources),
        "sources_unavailable": unavailable,
        "local_coverage_note": coverage_note,
        "quota_model_note": (
            "Approximate API pricing and discovered local logs are not the official Google "
            "quota meter; some usage may be missing or unpriced. The anchored recurring "
            "168-hour schedule assumption is not rolling per-request expiry; future observations "
            "may correct it."
        ),
    }
    if state is None:
        result["human_action"] = (
            "Reset boundary unknown. Register it with --quota-reset WHEN; current-cycle "
            "usage and pace estimates are unavailable until a reset is known."
        )
        return result

    next_reset, projected = next_reset_boundary(state.anchor_utc, now)
    cycle_start = next_reset - WEEK
    result["reset_local"] = next_reset.astimezone(local_tz).isoformat()
    result["cycle_start_local"] = cycle_start.astimezone(local_tz).isoformat()
    result["seconds_until_reset"] = math.ceil((next_reset - now).total_seconds())
    result["reset_projected"] = projected
    if stats is None:
        return result

    metrics = stats.overall
    if metrics.llm_calls == 0:
        used: float | None = 0.0
    elif metrics.priced_calls > 0 and math.isfinite(metrics.cost_usd):
        used = metrics.cost_usd
    else:
        used = None
    result.update(
        {
            "estimated_used_usd": _rounded_finite(used, 6) if used is not None else None,
            "estimated_remaining_usd": (
                _rounded_finite(max(0.0, limit_usd - used), 6)
                if used is not None
                else None
            ),
            "percent_used": _rounded_finite(used / limit_usd * 100, 2)
            if used is not None
            else None,
            "priced_calls": metrics.priced_calls,
            "unpriced_calls": metrics.unpriced_calls,
            "llm_calls": metrics.llm_calls,
            "estimated_token_steps": metrics.estimated_steps,
        }
    )
    if result["quota_status"] == "blocked":
        for field in (
            "estimated_remaining_usd",
            "percent_used",
            "average_usd_per_day",
            "projected_used_at_reset_usd",
            "sustainable_remaining_usd_per_day",
            "estimated_limit_hit_local",
        ):
            result[field] = None
        return result

    elapsed = (now - cycle_start).total_seconds()
    remaining_seconds = (next_reset - now).total_seconds()
    if elapsed >= 3600 and used is not None and used > 0:
        average = used / (elapsed / 86400)
        projected = used + average * (remaining_seconds / 86400)
        sustainable = max(0.0, limit_usd - used) / (remaining_seconds / 86400)
        result["average_usd_per_day"] = _rounded_finite(average, 6)
        result["projected_used_at_reset_usd"] = _rounded_finite(projected, 6)
        result["sustainable_remaining_usd_per_day"] = _rounded_finite(sustainable, 6)
        if (
            used < limit_usd
            and math.isfinite(average)
            and average > 0
            and used + average * (remaining_seconds / 86400) >= limit_usd
        ):
            hit_at = cycle_start + timedelta(days=(limit_usd - 0.0) / average)
            if hit_at < next_reset:
                result["estimated_limit_hit_local"] = hit_at.astimezone(local_tz).isoformat()
    return result


def build_trailing_usage_report(
    stats: StatsReport | None,
    *,
    start_utc: datetime,
    end_utc: datetime,
    sources_scanned: Sequence[str],
    sources_unavailable: Sequence[str],
) -> dict[str, Any]:
    """Describe local usage in the inclusive trailing 168-hour interval.

    ``None`` statistics mean no configured source root could be scanned and are
    distinct from a readable root with no matching calls.
    """
    start = _as_utc(start_utc)
    end = _as_utc(end_utc)
    if start > end:
        raise ValueError("trailing usage start must not follow its end")
    used: float | None = None
    metrics: Any = None
    if stats is not None:
        metrics = stats.overall
        if metrics.llm_calls == 0:
            used = 0.0
        elif metrics.priced_calls and math.isfinite(metrics.cost_usd):
            used = metrics.cost_usd
    return {
        "window": "trailing_7_day_usage",
        "start_utc": start.isoformat(),
        "end_utc": end.isoformat(),
        "estimated_used_usd": _rounded_finite(used, 6) if used is not None else None,
        "average_usd_per_day": _rounded_finite(used / 7, 6) if used is not None else None,
        "llm_calls": metrics.llm_calls if metrics is not None else None,
        "priced_calls": metrics.priced_calls if metrics is not None else None,
        "unpriced_calls": metrics.unpriced_calls if metrics is not None else None,
        "estimated_token_steps": metrics.estimated_steps if metrics is not None else None,
        "sources_scanned": list(sources_scanned),
        "sources_unavailable": list(sources_unavailable),
        "note": "Local usage estimate only; this is not your Google quota window.",
    }


def format_quota_json(report: dict[str, Any]) -> str:
    """Serialize the dedicated quota response as flat, stable JSON."""
    return json.dumps(report, indent=2, allow_nan=False)


def format_countdown(seconds: int) -> str:
    """Render integer remaining seconds as a compact human duration."""
    remainder = max(0, int(seconds))
    days, remainder = divmod(remainder, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds_left = divmod(remainder, 60)
    parts = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    if seconds_left or not parts:
        parts.append(f"{seconds_left}s")
    return " ".join(parts)
