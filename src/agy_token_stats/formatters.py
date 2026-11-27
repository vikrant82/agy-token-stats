"""Human-readable CLI table and JSON report formatters for Antigravity token statistics.

Threading & Purity:
    - All formatting functions in this module are pure and do not mutate `StatsReport`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
import json
import os
import re
import sys
from typing import Any, TextIO

from agy_token_stats.models import SessionSummary, StatsReport, TokenMetrics
from agy_token_stats.quota import WEEK, format_countdown

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_RED = "\033[31m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_BLUE = "\033[34m"
_MAGENTA = "\033[35m"
_CYAN = "\033[36m"
_GRAY = "\033[90m"
_BRIGHT_RED = "\033[91m"
_BRIGHT_GREEN = "\033[92m"
_BRIGHT_YELLOW = "\033[93m"
_BRIGHT_BLUE = "\033[94m"
_BRIGHT_MAGENTA = "\033[95m"
_BRIGHT_CYAN = "\033[96m"
_BRIGHT_WHITE = "\033[97m"

_BANNER_WIDTH = 112


def resolve_color_enabled(
    mode: str = "auto",
    *,
    stream: TextIO | None = None,
) -> bool:
    """Resolve whether ANSI color codes should be emitted for `mode` (`auto|always|never`)."""
    normalized = (mode or "auto").strip().lower()
    if normalized == "always":
        return True
    if normalized == "never":
        return False

    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("TERM", "").strip().lower() == "dumb":
        return False

    force_color = os.environ.get("FORCE_COLOR", "").strip()
    clicolor_force = os.environ.get("CLICOLOR_FORCE", "").strip()
    if (force_color and force_color != "0") or (
        clicolor_force and clicolor_force != "0"
    ):
        return True

    target = stream if stream is not None else sys.stdout
    try:
        return bool(target.isatty())
    except Exception:
        return False


def _paint(text: str, *codes: str, color: bool) -> str:
    """Wrap `text` in ANSI SGR `codes` when `color=True`."""
    if not color or not codes or not text:
        return text
    return f"{''.join(codes)}{text}{_RESET}"


def _visible_len(text: str) -> int:
    """Return display length of `text` excluding ANSI SGR escape sequences."""
    if "\x1b" not in text:
        return len(text)
    return len(_ANSI_RE.sub("", text))


def _pad_cell(text: str, width: int, *, right: bool) -> str:
    """Pad `text` to visible `width` while preserving ANSI escape sequences."""
    pad = max(0, width - _visible_len(text))
    if pad == 0:
        return text
    spaces = " " * pad
    return f"{spaces}{text}" if right else f"{text}{spaces}"


def _fmt_int(val: int) -> str:
    """Format an integer with thousand separators."""
    return f"{val:,}"


def _fmt_pct(val: float) -> str:
    """Format a percentage with 1 decimal place."""
    return f"{val:.1f}%"


def _fmt_usd(val: float, *, partial: bool = False) -> str:
    """Format a USD cost value cleanly for CLI display."""
    if val <= 0.0:
        out = "$0.00"
    elif val < 0.01:
        out = f"${val:.4f}"
    else:
        out = f"${val:,.2f}"
    return f"{out}*" if partial else out


def _fmt_metrics_cost(m: TokenMetrics) -> str:
    """Format approximate cost for a `TokenMetrics` bucket (`—` when unpriced)."""
    if m.llm_calls == 0 or m.priced_calls == 0:
        return "—"
    return _fmt_usd(m.cost_usd, partial=(m.unpriced_calls > 0))


def _fmt_local_dt(dt: datetime | None) -> str:
    """Format a local datetime cleanly as `YYYY-MM-DD HH:MM:SS`."""
    if dt is None:
        return "—"
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _fmt_short_dt(dt: datetime | None) -> str:
    """Format a local datetime as `YYYY-MM-DD HH:MM` for compact tables."""
    if dt is None:
        return "—"
    return dt.strftime("%Y-%m-%d %H:%M")


def _cache_color(pct: float) -> tuple[str, ...]:
    if pct >= 80.0:
        return (_BRIGHT_GREEN,)
    if pct >= 50.0:
        return (_YELLOW,)
    return (_RED,)


def _think_color(pct: float) -> tuple[str, ...]:
    if pct >= 40.0:
        return (_BRIGHT_MAGENTA,)
    if pct > 0.0:
        return (_MAGENTA,)
    return (_GRAY,)


def _cost_color(m: TokenMetrics) -> tuple[str, ...]:
    if m.llm_calls == 0 or m.priced_calls == 0:
        return (_GRAY,)
    val = m.cost_usd
    if val >= 25.0:
        return (_BOLD, _BRIGHT_RED)
    if val >= 1.0:
        return (_BRIGHT_YELLOW,)
    return (_BRIGHT_GREEN,)


def _source_color(src: str) -> tuple[str, ...]:
    low = src.lower()
    if low == "cli":
        return (_BOLD, _CYAN)
    if low == "ide":
        return (_BOLD, _BRIGHT_BLUE)
    return (_BOLD, _BRIGHT_WHITE)


def _role_color(role: str) -> tuple[str, ...]:
    low = role.lower()
    if low == "root" or low.startswith("root "):
        return (_BOLD, _CYAN)
    if "implementer" in low:
        return (_GREEN,)
    if "oracle" in low:
        return (_MAGENTA,)
    if "reviewer" in low:
        return (_YELLOW,)
    if "bash" in low or "runner" in low:
        return (_BRIGHT_BLUE,)
    return (_CYAN,)


def _model_color(model_name: str) -> tuple[str, ...]:
    low = model_name.lower()
    if not low or low == "unknown":
        return (_GRAY,)
    if "carbon" in low:
        return (_BRIGHT_MAGENTA,)
    if "argon" in low:
        return (_BRIGHT_BLUE,)
    if "neon" in low:
        return (_CYAN,)
    if "hydrogen" in low:
        return (_GREEN,)
    if "flash" in low:
        return (_BRIGHT_GREEN,)
    if "pro" in low:
        return (_YELLOW,)
    if "claude" in low:
        return (_BRIGHT_YELLOW,)
    return (_BRIGHT_WHITE,)


def _fmt_share_bar(share_pct: float, *, color: bool, width: int = 8) -> str:
    """Render a compact progress bar + percentage string (`████░░░░  50.0%`)."""
    clamped = max(0.0, min(100.0, share_pct))
    filled = int(round((clamped / 100.0) * width))
    empty = width - filled
    bar_filled = _paint("█" * filled, _CYAN, color=color)
    bar_empty = _paint("░" * empty, _GRAY, color=color)
    pct_str = _paint(f"{clamped:5.1f}%", _BOLD, color=color)
    return f"{bar_filled}{bar_empty} {pct_str}"


def _render_aligned_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    right_align_cols: set[int],
    *,
    color: bool = False,
) -> list[str]:
    """Render an aligned CLI table with ANSI-aware column width measurement."""
    widths = [_visible_len(h) for h in headers]
    for r in rows:
        for idx, cell in enumerate(r):
            vis = _visible_len(cell)
            if vis > widths[idx]:
                widths[idx] = vis

    def _fmt_row(cells: Sequence[str], *, is_header: bool = False) -> str:
        parts = []
        for idx, cell in enumerate(cells):
            padded = _pad_cell(cell, widths[idx], right=(idx in right_align_cols))
            if is_header:
                padded = _paint(padded, _BOLD, _BRIGHT_CYAN, color=color)
            parts.append(padded)
        return "  ".join(parts)

    sep_parts = [("─" * widths[idx]) for idx in range(len(headers))]
    sep_line = _paint("  ".join(sep_parts), _GRAY, color=color)
    out = [_fmt_row(headers, is_header=True), sep_line]
    for r in rows:
        out.append(_fmt_row(r))
    return out


def _render_breakdown_table(
    title: str,
    first_col_header: str,
    buckets: Mapping[str, TokenMetrics],
    session_counts: Mapping[str, int],
    total_tokens_overall: int,
    *,
    label_styler: object,
    color: bool,
) -> list[str]:
    """Render a standardized breakdown table (Source / Role / Model) with cost and share bar."""
    out = ["", _paint(f"▸ {title}", _BOLD, _BRIGHT_CYAN, color=color)]
    headers = [
        first_col_header,
        "Sessions",
        "LLM Calls",
        "Uncached In",
        "Cache Read",
        "Cache %",
        "Output",
        "Thinking (est)",
        "Think %",
        "Net Tokens",
        "Total Tokens",
        "Peak Ctx",
        "Est. Cost",
        "Share %",
    ]
    rows: list[list[str]] = []
    for key, m in buckets.items():
        share = (
            (m.total_tokens / total_tokens_overall * 100.0)
            if total_tokens_overall > 0
            else 0.0
        )
        key_codes = label_styler(key)  # type: ignore[operator]
        cost_str = _paint(_fmt_metrics_cost(m), *_cost_color(m), color=color)
        rows.append(
            [
                _paint(key, *key_codes, color=color),
                _fmt_int(session_counts.get(key, 0)),
                _fmt_int(m.llm_calls),
                _paint(_fmt_int(m.uncached_input), _CYAN, color=color),
                _paint(_fmt_int(m.cache_read), _GREEN, color=color),
                _paint(_fmt_pct(m.cache_hit_pct), *_cache_color(m.cache_hit_pct), color=color),
                _paint(_fmt_int(m.output_tokens), _YELLOW, color=color),
                _paint(_fmt_int(m.thinking_tokens), _MAGENTA, color=color),
                _paint(_fmt_pct(m.thinking_pct), *_think_color(m.thinking_pct), color=color),
                _fmt_int(m.net_tokens),
                _paint(_fmt_int(m.total_tokens), _BOLD, color=color),
                _paint(_fmt_int(m.peak_context), _BLUE, color=color),
                cost_str,
                _fmt_share_bar(share, color=color),
            ]
        )
    out.extend(
        _render_aligned_table(
            headers,
            rows,
            right_align_cols={1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13},
            color=color,
        )
    )
    return out


def format_json_report(report: StatsReport, *, indent: int = 2) -> str:
    """Serialize `StatsReport` to a formatted JSON string."""
    return json.dumps(report.to_dict(), indent=indent)


def format_quota_human(report: Mapping[str, Any], *, color: bool = False) -> str:
    """Render a quota snapshot as ANSI-aware tables without changing the report."""
    def title(text: str) -> str:
        return _paint(text, _BOLD, _BRIGHT_CYAN, color=color)

    def local_time(value: str, *, projected: bool = False) -> str:
        instant = datetime.fromisoformat(value)
        offset = instant.strftime("%z")
        offset = f"{offset[:3]}:{offset[3:]}" if offset else ""
        text = f"{instant:%a %b %d, %H:%M:%S} {offset}".rstrip()
        return f"{text} (projected)" if projected else text

    blocked = report["quota_status"] == "blocked"
    if blocked:
        status = (
            "BLOCKED — weekly limit notice"
            if report["reset_source"] == "manual_limit_notice"
            else "BLOCKED — manually recorded"
        )
    else:
        status = "Unverified"
    reset = report["reset_local"]
    rows = [("Status", _paint(status, _BOLD, _BRIGHT_RED if blocked else _YELLOW, color=color))]
    if reset is None:
        rows.append(("Reset at", "Unknown"))
    else:
        countdown = format_countdown(report["seconds_until_reset"])
        rows.extend(
            [
                ("Resets in", _paint(countdown, _BOLD, _BRIGHT_CYAN, color=color)),
                ("Reset at", local_time(reset, projected=report["reset_projected"])),
            ]
        )
    lines = [title("Antigravity weekly quota"), ""]
    lines.extend(_render_aligned_table(("Quota", "Value"), rows, set(), color=color))
    if reset is None:
        lines.extend(["", "Register a reset with --quota-reset WHEN."])
        return "\n".join(lines)

    used = report["estimated_used_usd"]
    spend = _paint(f"${used:,.2f}", _BOLD, _MAGENTA, color=color) if used is not None else "Unavailable"
    budget_rows = [("Spend in local logs", spend)]
    if not blocked:
        budget_rows.append(("Planning budget", f"${report['limit_usd']:,.2f}"))
        remaining = report["estimated_remaining_usd"]
        if remaining is not None:
            budget_rows.append(("Est. budget left until reset", f"${remaining:,.2f}"))
        hit = report["estimated_limit_hit_local"]
        if hit is not None:
            budget_rows.append(("Budget hit at current pace", local_time(hit)))
    coverage = report["sources_scanned"]
    if coverage:
        budget_rows.append(("Sources", ", ".join(coverage)))
    lines.extend(["", title("▸ Local usage estimate")])
    lines.extend(_render_aligned_table(("Metric", "Value"), budget_rows, {1}, color=color))
    if not blocked:
        average = report["average_usd_per_day"]
        full_cycle_rate = report["limit_usd"] / WEEK.days
        headers = ["Avg. local spend / day", f"Budget / day ({WEEK.days} days)"]
        values = [
            f"${average:,.2f}" if average is not None else "—",
            f"${full_cycle_rate:,.2f}",
        ]
        remaining_rate = report["sustainable_remaining_usd_per_day"]
        if remaining_rate is not None and report["seconds_until_reset"] >= 86400:
            headers.append("Est. budget left / day")
            values.append(f"${remaining_rate:,.2f}")
        lines.extend(["", title("▸ Daily averages")])
        lines.extend(_render_aligned_table(headers, [values], set(range(len(headers))), color=color))
    lines.extend(["", "Local estimates can miss usage; this is not Google's quota meter."])
    if report["reset_projected"]:
        lines.append("Projected reset assumes a 168-hour cycle; not verified by Google.")
    if report["sources_unavailable"]:
        missing = ", ".join(report["sources_unavailable"])
        lines.append(_paint(f"Unavailable sources: {missing}", _YELLOW, color=color))
    if report["unpriced_calls"]:
        lines.append(f"Unpriced calls excluded: {report['unpriced_calls']:,}")
    if report["estimated_token_steps"]:
        lines.append(f"Calls with estimated tokens: {report['estimated_token_steps']:,}")
    return "\n".join(lines)


def format_trailing_usage_human(
    report: Mapping[str, Any], *, color: bool = False
) -> str:
    """Render the no-reset first-run preview of trailing local seven-day usage."""
    title = _paint("Last 7 days · local usage", _BOLD, _BRIGHT_CYAN, color=color)
    used = report["estimated_used_usd"]
    average = report["average_usd_per_day"]
    rows = [
        (
            "Local spend",
            f"${used:,.2f}" if used is not None else "Unavailable",
        ),
        (
            "Average / day",
            f"${average:,.2f}" if average is not None else "Unavailable",
        ),
    ]
    if report["llm_calls"] is not None:
        rows.append(
            (
                "Calls",
                f"{report['llm_calls']:,} total; {report['priced_calls']:,} priced; "
                f"{report['unpriced_calls']:,} unpriced",
            )
        )
    if report["sources_scanned"]:
        rows.append(("Sources", ", ".join(report["sources_scanned"])))
    lines = [title]
    lines.extend(_render_aligned_table(("Metric", "Value"), rows, {1}, color=color))
    lines.append("Local estimates can miss usage; this is not your Google quota window.")
    if report["sources_unavailable"]:
        missing = ", ".join(report["sources_unavailable"])
        lines.append(_paint(f"Unavailable sources: {missing}", _YELLOW, color=color))
    if report["unpriced_calls"]:
        lines.append(f"Unpriced calls excluded: {report['unpriced_calls']:,}")
    if report["estimated_token_steps"]:
        lines.append(f"Calls with estimated tokens: {report['estimated_token_steps']:,}")
    return "\n".join(lines)


def format_human_report(
    report: StatsReport,
    *,
    top_n: int = 20,
    show_daily: bool | None = None,
    show_subagents: bool = False,
    color: bool = False,
) -> str:
    """Format a `StatsReport` as a human-readable, optionally colorized CLI report.

    Includes:
        1. Header & Overall Summary (window bounds, sources, sessions, LLM calls,
           uncached/cached/gross input, cache hit %, output, estimated thinking & response
           tokens, net/total tokens, peak context, approximate Gemini USD cost & savings).
        2. Breakdown by Source (`cli` vs `ide`), Breakdown by Role (`root` vs subagent
           `typeName`), and Breakdown by Model.
        3. Daily Breakdown Table (shown when `show_daily=True` or when `show_daily is None`
           and active steps span multiple calendar dates).
        4. Top Sessions Table (with `Model`, `Thinking`, `Est. Cost` columns, plus
           optional indented child subagent tree when `show_subagents=True`).
    """
    lines: list[str] = []
    w = report.window
    start_str = _fmt_local_dt(w.start_local) if w.start_local else "beginning"
    end_str = _fmt_local_dt(w.end_local) if w.end_local else "now"
    tz_name = ""
    if w.start_local and w.start_local.tzinfo:
        tz_name = f" ({w.start_local.strftime('%Z')})"
    elif w.end_local and w.end_local.tzinfo:
        tz_name = f" ({w.end_local.strftime('%Z')})"

    mode_str = "session-start" if report.by_session_mode else "step-timestamp"
    hier_str = "flat" if report.flat_mode else "tree (subagents rolled into root)"
    ov = report.overall

    if ov.estimated_steps == 0:
        accuracy_str = (
            "100% exact native transcript counters "
            "(thinking split estimated proportionally from step thinking text within native output_tokens)"
        )
    else:
        accuracy_str = (
            f"includes {ov.estimated_steps:,} fallback-estimated step(s) "
            f"out of {ov.llm_calls:,} LLM calls "
            f"(thinking split estimated proportionally from step thinking text within output_tokens)"
        )

    heavy_rule = _paint("━" * _BANNER_WIDTH, _CYAN, color=color)
    thin_rule = _paint("─" * _BANNER_WIDTH, _GRAY, color=color)

    lines.append(heavy_rule)
    title_left = _paint(
        "◆ Antigravity (agy) Token Usage Report",
        _BOLD,
        _BRIGHT_CYAN,
        color=color,
    )
    title_right = _paint(f"Window: {w.label}", _BOLD, _BRIGHT_YELLOW, color=color)
    lines.append(f"{title_left} — {title_right}")
    lines.append(heavy_rule)

    def _meta_line(label: str, value: str) -> str:
        lbl = _paint(f"  {label:<15}: ", _GRAY, color=color)
        return f"{lbl}{value}"

    lines.append(
        _meta_line(
            "Time Window",
            f"[{_paint(start_str, _BRIGHT_WHITE, color=color)} .. "
            f"{_paint(end_str, _BRIGHT_WHITE, color=color)}]{tz_name} "
            f"{_paint(f'({mode_str})', _DIM, color=color)}",
        )
    )
    sources_val = (
        ", ".join(_paint(s, *_source_color(s), color=color) for s in report.sources_scanned)
        if report.sources_scanned
        else _paint("none", _GRAY, color=color)
    )
    lines.append(_meta_line("Sources Scanned", sources_val))
    lines.append(_meta_line("Hierarchy Mode", hier_str))
    if report.model_filter:
        lines.append(
            _meta_line(
                "Model Filter",
                _paint(report.model_filter, _BOLD, _BRIGHT_YELLOW, color=color),
            )
        )
    lines.append(
        _meta_line(
            "Active Sessions",
            f"{_paint(f'{report.active_root_sessions:,}', _BOLD, _CYAN, color=color)} root session(s), "
            f"{_paint(f'{report.active_subagent_sessions:,}', _BOLD, _MAGENTA, color=color)} subagent session(s)",
        )
    )
    lines.append(
        _meta_line(
            "Activity Counts",
            f"{_paint(f'{ov.llm_calls:,}', _BOLD, color=color)} LLM calls "
            f"({_paint(f'{ov.thinking_steps:,}', _MAGENTA, color=color)} w/ thinking) | "
            f"{_paint(f'{ov.user_turns:,}', _BOLD, color=color)} user turns | "
            f"{_paint(f'{ov.tool_calls:,}', _BOLD, color=color)} tool calls",
        )
    )
    lines.append(_meta_line("Counter Source", _paint(accuracy_str, _DIM, color=color)))
    lines.append(thin_rule)

    def _kpi_row(
        left_label: str,
        left_val: str,
        left_unit: str,
        right_label: str,
        right_val: str,
        right_extra: str = "",
    ) -> str:
        l_lbl = _paint(f"  {left_label:<15}: ", _GRAY, color=color)
        r_lbl = _paint(f"    {right_label:<22}: ", _GRAY, color=color)
        l_unit = f" {left_unit}" if left_unit else "       "
        return f"{l_lbl}{left_val}{l_unit}{r_lbl}{right_val}{right_extra}"

    uncached_cell = _paint(_pad_cell(_fmt_int(ov.uncached_input), 14, right=True), _CYAN, color=color)
    cache_cell = _paint(_pad_cell(_fmt_int(ov.cache_read), 14, right=True), _GREEN, color=color)
    gross_cell = _paint(_pad_cell(_fmt_int(ov.gross_input), 14, right=True), _BRIGHT_CYAN, color=color)
    hit_cell = _paint(
        _pad_cell(_fmt_pct(ov.cache_hit_pct), 14, right=True),
        *_cache_color(ov.cache_hit_pct),
        color=color,
    )
    peak_cell = _paint(_pad_cell(_fmt_int(ov.peak_context), 14, right=True), _BRIGHT_BLUE, color=color)

    out_cell = _paint(_pad_cell(_fmt_int(ov.output_tokens), 14, right=True), _YELLOW, color=color)
    think_cell = _paint(_pad_cell(_fmt_int(ov.thinking_tokens), 14, right=True), _MAGENTA, color=color)
    resp_cell = _paint(_pad_cell(_fmt_int(ov.response_tokens), 14, right=True), _BRIGHT_YELLOW, color=color)
    net_cell = _paint(_pad_cell(_fmt_int(ov.net_tokens), 14, right=True), _BOLD, color=color)
    tot_cell = _paint(_pad_cell(_fmt_int(ov.total_tokens), 14, right=True), _BOLD, _BRIGHT_WHITE, color=color)

    lines.append(
        _kpi_row("Uncached Input", uncached_cell, "tokens", "Output Tokens", out_cell, " tokens")
    )
    lines.append(
        _kpi_row(
            "Cache Read",
            cache_cell,
            "tokens",
            "Thinking Tokens (est.)",
            think_cell,
            f" tokens ({_paint(_fmt_pct(ov.thinking_pct), *_think_color(ov.thinking_pct), color=color)} of output, "
            f"{_fmt_int(ov.thinking_steps)}/{_fmt_int(ov.llm_calls)} calls)",
        )
    )
    lines.append(
        _kpi_row(
            "Gross Input",
            gross_cell,
            "tokens",
            "Response Tokens (est.)",
            resp_cell,
            " tokens (content + tool calls)",
        )
    )
    lines.append(
        _kpi_row(
            "Cache Hit Rate",
            hit_cell,
            "",
            "Net Tokens",
            net_cell,
            " tokens (uncached in + out)",
        )
    )
    lines.append(
        _kpi_row(
            "Peak Context",
            peak_cell,
            "tokens",
            "Total Tokens",
            tot_cell,
            " tokens (gross in + out)",
        )
    )

    if ov.priced_calls > 0:
        cost_val_str = _paint(
            _pad_cell(_fmt_metrics_cost(ov), 14, right=True),
            *_cost_color(ov),
            color=color,
        )
        in_c = _paint(_fmt_usd(ov.input_cost_usd), _CYAN, color=color)
        ca_c = _paint(_fmt_usd(ov.cache_cost_usd), _GREEN, color=color)
        out_c = _paint(_fmt_usd(ov.output_cost_usd), _YELLOW, color=color)
        saved_c = _paint(_fmt_usd(ov.cache_savings_usd), _BRIGHT_GREEN, color=color)
        coverage_note = ""
        if ov.unpriced_calls > 0:
            coverage_note = _paint(
                f" ({ov.priced_calls:,}/{ov.llm_calls:,} calls priced)",
                _GRAY,
                color=color,
            )
        lines.append(
            _kpi_row(
                "Est. Cost (USD)",
                cost_val_str,
                "approx",
                "Cost Breakdown",
                f"in {in_c} + cache {ca_c} + out {out_c}",
                f" | cache saved ~{saved_c}{coverage_note}",
            )
        )
    elif ov.llm_calls > 0:
        unpriced_cell = _paint(_pad_cell("—", 14, right=True), _GRAY, color=color)
        lines.append(
            _kpi_row(
                "Est. Cost (USD)",
                unpriced_cell,
                "",
                "Pricing Coverage",
                _paint(
                    f"no Gemini pricing match ({ov.unpriced_calls:,} unpriced call(s))",
                    _GRAY,
                    color=color,
                ),
            )
        )

    lines.append(heavy_rule)

    if not ov.has_activity and not report.sessions:
        lines.append("")
        lines.append(
            _paint(
                "No active Antigravity sessions or steps matched the selected filters.",
                _YELLOW,
                color=color,
            )
        )
        return "\n".join(lines)

    # 2a. Breakdown by Source
    if report.by_source:
        lines.extend(
            _render_breakdown_table(
                "Breakdown by Source",
                "Source",
                report.by_source,
                report.by_source_sessions,
                ov.total_tokens,
                label_styler=_source_color,
                color=color,
            )
        )

    # 2b. Breakdown by Role
    if report.by_role:
        lines.extend(
            _render_breakdown_table(
                "Breakdown by Agent Role",
                "Role / Type",
                report.by_role,
                report.by_role_sessions,
                ov.total_tokens,
                label_styler=_role_color,
                color=color,
            )
        )

    # 2c. Breakdown by Model
    if report.by_model:
        lines.extend(
            _render_breakdown_table(
                "Breakdown by Model",
                "Model",
                report.by_model,
                report.by_model_sessions,
                ov.total_tokens,
                label_styler=_model_color,
                color=color,
            )
        )

    # 3. Daily Breakdown Table
    should_show_daily = (
        show_daily if show_daily is not None else (len(report.daily) > 1)
    )
    if should_show_daily and report.daily:
        lines.append("")
        lines.append(_paint("▸ Daily Breakdown (Local Time)", _BOLD, _BRIGHT_CYAN, color=color))
        day_headers = [
            "Date",
            "Root Sess",
            "Subagents",
            "LLM Calls",
            "Uncached In",
            "Cache Read",
            "Cache %",
            "Output",
            "Thinking (est)",
            "Think %",
            "Net Tokens",
            "Total Tokens",
            "Est. Cost",
        ]
        day_rows: list[list[str]] = []
        for d in report.daily:
            dm = d.metrics
            day_rows.append(
                [
                    _paint(d.date_str, _BOLD, _BRIGHT_WHITE, color=color),
                    _fmt_int(d.active_root_sessions),
                    _fmt_int(d.active_subagent_sessions),
                    _fmt_int(dm.llm_calls),
                    _paint(_fmt_int(dm.uncached_input), _CYAN, color=color),
                    _paint(_fmt_int(dm.cache_read), _GREEN, color=color),
                    _paint(_fmt_pct(dm.cache_hit_pct), *_cache_color(dm.cache_hit_pct), color=color),
                    _paint(_fmt_int(dm.output_tokens), _YELLOW, color=color),
                    _paint(_fmt_int(dm.thinking_tokens), _MAGENTA, color=color),
                    _paint(_fmt_pct(dm.thinking_pct), *_think_color(dm.thinking_pct), color=color),
                    _fmt_int(dm.net_tokens),
                    _paint(_fmt_int(dm.total_tokens), _BOLD, color=color),
                    _paint(_fmt_metrics_cost(dm), *_cost_color(dm), color=color),
                ]
            )
        lines.extend(
            _render_aligned_table(
                day_headers,
                day_rows,
                right_align_cols={1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
                color=color,
            )
        )

    # 4. Top Sessions Table
    if report.sessions:
        lines.append("")
        displayed_sessions = (
            report.sessions[:top_n] if top_n > 0 else report.sessions
        )
        total_sess = len(report.sessions)
        if top_n > 0 and total_sess > top_n:
            sess_title = f"Top Sessions (showing {len(displayed_sessions)} of {total_sess})"
        else:
            sess_title = f"Sessions ({total_sess})"
        lines.append(_paint(f"▸ {sess_title}", _BOLD, _BRIGHT_CYAN, color=color))

        sess_headers = [
            "Session ID",
            "Src",
            "Role",
            "Model",
            "Last Active",
            "Subs",
            "Calls",
            "Uncached In",
            "Cache Read",
            "Cache %",
            "Output",
            "Thinking",
            "Net Tokens",
            "Total Tokens",
            "Peak Ctx",
            "Est. Cost",
            "Title / Prompt Preview",
        ]
        sess_rows: list[list[str]] = []

        def _fmt_model_cell(raw_model: str, max_len: int = 32) -> str:
            label = (raw_model or "").strip() or "Unknown"
            if len(label) > max_len:
                label = label[: max_len - 1].rstrip() + "…"
            return _paint(label, *_model_color(label), color=color)

        def _append_session_rows(s: SessionSummary, depth: int = 0) -> None:
            tm = s.tree_metrics if depth == 0 else s.self_metrics
            short_id = s.conversation_id[:8]
            if depth > 0:
                branch = _paint("  " * (depth - 1) + "└─ ", _GRAY, color=color)
                id_col = f"{branch}{_paint(short_id, *_role_color(s.agent_type), color=color)}"
            else:
                id_col = _paint(short_id, _BOLD, _BRIGHT_CYAN, color=color)

            title_preview = s.title or s.workspace or "—"
            if len(title_preview) > 48:
                title_preview = title_preview[:47].rstrip() + "…"

            sess_rows.append(
                [
                    id_col,
                    _paint(s.source_kind, *_source_color(s.source_kind), color=color),
                    _paint(s.agent_type, *_role_color(s.agent_type), color=color),
                    _fmt_model_cell(
                        s.model if depth == 0 else (s.self_model or s.model)
                    ),
                    _paint(_fmt_short_dt(s.last_active_local), _DIM, color=color),
                    _fmt_int(s.active_subagent_count) if depth == 0 else "—",
                    _fmt_int(tm.llm_calls),
                    _paint(_fmt_int(tm.uncached_input), _CYAN, color=color),
                    _paint(_fmt_int(tm.cache_read), _GREEN, color=color),
                    _paint(_fmt_pct(tm.cache_hit_pct), *_cache_color(tm.cache_hit_pct), color=color),
                    _paint(_fmt_int(tm.output_tokens), _YELLOW, color=color),
                    _paint(_fmt_int(tm.thinking_tokens), _MAGENTA, color=color),
                    _fmt_int(tm.net_tokens),
                    _paint(_fmt_int(tm.total_tokens), _BOLD, color=color),
                    _paint(_fmt_int(tm.peak_context), _BLUE, color=color),
                    _paint(_fmt_metrics_cost(tm), *_cost_color(tm), color=color),
                    title_preview if depth == 0 else _paint(title_preview, _DIM, color=color),
                ]
            )
            if show_subagents and s.children:
                if depth == 0 and s.self_metrics.has_activity:
                    sm = s.self_metrics
                    branch = _paint("├─ ", _GRAY, color=color)
                    self_id = f"{branch}{_paint(f'{short_id} (self)', _CYAN, color=color)}"
                    self_role = f"{s.agent_type} (self)"
                    sess_rows.append(
                        [
                            self_id,
                            _paint(s.source_kind, *_source_color(s.source_kind), color=color),
                            _paint(self_role, *_role_color(self_role), color=color),
                            _fmt_model_cell(s.self_model or s.model),
                            _paint(_fmt_short_dt(s.last_active_local), _DIM, color=color),
                            "—",
                            _fmt_int(sm.llm_calls),
                            _paint(_fmt_int(sm.uncached_input), _CYAN, color=color),
                            _paint(_fmt_int(sm.cache_read), _GREEN, color=color),
                            _paint(_fmt_pct(sm.cache_hit_pct), *_cache_color(sm.cache_hit_pct), color=color),
                            _paint(_fmt_int(sm.output_tokens), _YELLOW, color=color),
                            _paint(_fmt_int(sm.thinking_tokens), _MAGENTA, color=color),
                            _fmt_int(sm.net_tokens),
                            _paint(_fmt_int(sm.total_tokens), _BOLD, color=color),
                            _paint(_fmt_int(sm.peak_context), _BLUE, color=color),
                            _paint(_fmt_metrics_cost(sm), *_cost_color(sm), color=color),
                            _paint("(parent session excluding subagents)", _DIM, color=color),
                        ]
                    )
                for ch in s.children:
                    _append_session_rows(ch, depth=depth + 1)

        for s in displayed_sessions:
            _append_session_rows(s, depth=0)

        lines.extend(
            _render_aligned_table(
                sess_headers,
                sess_rows,
                right_align_cols={5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
                color=color,
            )
        )

    if ov.priced_calls > 0:
        lines.append("")
        lines.append(
            _paint(
                "Note: Approx costs use Google AI Studio Gemini API Paid Tier rates "
                "(USD/1M tokens; output includes thinking): https://aistudio.google.com/docs/choosing-a-model",
                _GRAY,
                color=color,
            )
        )
        if ov.unpriced_calls > 0:
            lines.append(
                _paint(
                    "* Partial or unpriced rows exclude non-Gemini or Unknown model calls (shown as —).",
                    _GRAY,
                    color=color,
                )
            )

    return "\n".join(lines)
