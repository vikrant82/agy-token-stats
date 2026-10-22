"""Timestamp parsing and local-time window resolution for Antigravity token statistics.

Threading & Safety:
    - All functions in this module are pure (given explicit `now` and `local_tz`) and
      thread-safe.
    - Fractional-second ISO-8601 strings of any precision (e.g. `.1Z`, `.12345Z`,
      `.123456789+05:30`) are normalized to 6 microsecond digits for Python 3.9+
      `datetime.fromisoformat()` compatibility.
"""

from __future__ import annotations

from datetime import datetime, time, timedelta, timezone, tzinfo
import math
import re
from typing import Any

from agy_token_stats.models import TimeWindow

ISO_FRACTION_RE = re.compile(r"(\d{2}:\d{2}:\d{2})\.(\d+)")


def get_default_local_tz() -> tzinfo:
    """Return the system local timezone (falling back to UTC if unavailable)."""
    return datetime.now().astimezone().tzinfo or timezone.utc


def parse_iso_timestamp(raw: Any, local_tz: tzinfo) -> tuple[datetime, datetime] | None:
    """Parse an ISO-8601 timestamp string into `(utc_dt, local_dt)`.

    Accepts trailing `'Z'`, explicit timezone offsets, variable fractional
    seconds (e.g. `.5Z`, `.123456789Z`), or naive timestamps (which are
    interpreted in `local_tz`). Returns `None` if `raw` is not a valid ISO
    timestamp string.
    """
    if not isinstance(raw, str):
        return None
    cleaned = raw.strip()
    if not cleaned:
        return None
    if cleaned.endswith(("Z", "z")):
        cleaned = cleaned[:-1] + "+00:00"
    cleaned = ISO_FRACTION_RE.sub(
        lambda m: f"{m.group(1)}.{m.group(2)[:6].ljust(6, '0')}",
        cleaned,
        count=1,
    )
    try:
        dt = datetime.fromisoformat(cleaned)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt_local = dt.replace(tzinfo=local_tz)
        dt_utc = dt_local.astimezone(timezone.utc)
    else:
        dt_utc = dt.astimezone(timezone.utc)
        dt_local = dt.astimezone(local_tz)
    return dt_utc, dt_local


def parse_bound_datetime(raw: str, *, is_end: bool, local_tz: tzinfo) -> datetime:
    """Parse a user-supplied `--since` or `--until` date/datetime string in `local_tz`.

    Date-only strings (`YYYY-MM-DD`) expand to `00:00:00` when `is_end=False`
    and `23:59:59.999999` when `is_end=True`. Full ISO-8601 strings preserve
    their exact time (converting aware timestamps to `local_tz`).

    Raises:
        ValueError: If `raw` cannot be parsed as a valid date or ISO-8601 datetime.
    """
    cleaned = raw.strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", cleaned):
        try:
            d = datetime.strptime(cleaned, "%Y-%m-%d").date()
        except ValueError as exc:
            raise ValueError(f"Invalid date '{raw}': {exc}") from exc
        bound_time = time(23, 59, 59, 999999) if is_end else time(0, 0, 0)
        return datetime.combine(d, bound_time, tzinfo=local_tz)

    parsed = parse_iso_timestamp(cleaned, local_tz)
    if parsed is None:
        raise ValueError(
            f"Invalid timestamp '{raw}': expected YYYY-MM-DD or ISO-8601 datetime"
        )
    return parsed[1]


def resolve_time_window(
    *,
    now: datetime | None = None,
    today: bool = False,
    yesterday: bool = False,
    last_n_days: int | None = None,
    last_n_hours: float | None = None,
    since: str | None = None,
    until: str | None = None,
    local_tz: tzinfo | None = None,
) -> TimeWindow:
    """Resolve CLI time flags into a concrete local-time `TimeWindow`.

    Calendar-day rules (in `local_tz`):
        - `today=True`: `[today 00:00:00 .. now_local]`
        - `yesterday=True`: `[yesterday 00:00:00 .. yesterday 23:59:59.999999]`
        - `last_n_days=N`: `N` calendar days including today, i.e.
          `[(today 00:00:00 - (N - 1) days) .. now_local]`. Thus `N=1` matches `--today`
          and `N=7` covers today plus the 6 preceding calendar days.
        - `last_n_hours=H`: `[now_local - H hours .. now_local]`.
        - `since` / `until`: explicit lower/upper bounds (can be used alone or to refine).

    Raises:
        ValueError: If mutually exclusive presets are combined, if `last_n_days < 1`,
            if `last_n_hours <= 0` or non-finite, or if `start_local > end_local`.
    """
    tz = local_tz or (now.tzinfo if now and now.tzinfo else get_default_local_tz())
    now_local = (
        now.astimezone(tz)
        if (now and now.tzinfo)
        else (now.replace(tzinfo=tz) if now else datetime.now(tz))
    )
    today_midnight = datetime.combine(now_local.date(), time(0, 0, 0), tzinfo=tz)

    presets_used = sum(
        [
            bool(today),
            bool(yesterday),
            last_n_days is not None,
            last_n_hours is not None,
        ]
    )
    if presets_used > 1:
        raise ValueError(
            "Choose at most one preset window (--today, --yesterday, --last-n-days, --last-n-hours)"
        )

    start_local: datetime | None = None
    end_local: datetime | None = None
    label = "all-time"

    if today:
        start_local = today_midnight
        end_local = now_local
        label = "today"
    elif yesterday:
        y_date = now_local.date() - timedelta(days=1)
        start_local = datetime.combine(y_date, time(0, 0, 0), tzinfo=tz)
        end_local = datetime.combine(y_date, time(23, 59, 59, 999999), tzinfo=tz)
        label = "yesterday"
    elif last_n_days is not None:
        if last_n_days < 1:
            raise ValueError(f"--last-n-days must be >= 1 (got {last_n_days})")
        start_local = today_midnight - timedelta(days=last_n_days - 1)
        end_local = now_local
        label = f"last-{last_n_days}-day{'s' if last_n_days != 1 else ''}"
    elif last_n_hours is not None:
        if not math.isfinite(last_n_hours) or last_n_hours <= 0:
            raise ValueError(f"--last-n-hours must be > 0 (got {last_n_hours})")
        start_local = now_local - timedelta(hours=last_n_hours)
        end_local = now_local
        label = f"last-{last_n_hours:g}-hours"

    if since is not None:
        since_dt = parse_bound_datetime(since, is_end=False, local_tz=tz)
        start_local = since_dt if start_local is None else max(start_local, since_dt)
        if label == "all-time":
            label = f"since-{since.strip()}"

    if until is not None:
        until_dt = parse_bound_datetime(until, is_end=True, local_tz=tz)
        end_local = until_dt if end_local is None else min(end_local, until_dt)
        if since is not None and presets_used == 0:
            label = f"{since.strip()}..{until.strip()}"
        elif label == "all-time":
            label = f"until-{until.strip()}"

    if start_local is not None and end_local is not None and start_local > end_local:
        raise ValueError(
            f"Window start ({start_local.isoformat()}) must not be after window end ({end_local.isoformat()})"
        )

    return TimeWindow(label=label, start_local=start_local, end_local=end_local)
