"""Human-readable CLI table and JSON report formatters for Antigravity token statistics.

Threading & Purity:
    - All formatting functions in this module are pure and do not mutate `StatsReport`.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
import json

from agy_token_stats.models import SessionSummary, StatsReport


def _fmt_int(val: int) -> str:
    """Format an integer with thousand separators."""
    return f"{val:,}"


def _fmt_pct(val: float) -> str:
    """Format a percentage with 1 decimal place."""
    return f"{val:.1f}%"


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


def _render_aligned_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    right_align_cols: set[int],
) -> list[str]:
    """Render an aligned plain-text table."""
    widths = [len(h) for h in headers]
    for r in rows:
        for idx, cell in enumerate(r):
            if len(cell) > widths[idx]:
                widths[idx] = len(cell)

    def _fmt_row(cells: Sequence[str]) -> str:
        parts = []
        for idx, cell in enumerate(cells):
            w = widths[idx]
            parts.append(cell.rjust(w) if idx in right_align_cols else cell.ljust(w))
        return "  ".join(parts)

    sep_parts = [("-" * widths[idx]) for idx in range(len(headers))]
    out = [_fmt_row(headers), "  ".join(sep_parts)]
    for r in rows:
        out.append(_fmt_row(r))
    return out


def format_json_report(report: StatsReport, *, indent: int = 2) -> str:
    """Serialize `StatsReport` to a formatted JSON string."""
    return json.dumps(report.to_dict(), indent=indent)


def format_human_report(
    report: StatsReport,
    *,
    top_n: int = 20,
    show_daily: bool | None = None,
    show_subagents: bool = False,
) -> str:
    """Format a `StatsReport` as a human-readable, aligned CLI report.

    Includes:
        1. Header & Overall Summary (window bounds, sources, sessions, LLM calls,
           uncached/cached/gross input, cache hit %, output, estimated thinking & response
           tokens, net/total tokens, peak context).
        2. Breakdown by Source (`cli` vs `ide`), Breakdown by Role (`root` vs subagent
           `typeName`), and Breakdown by Model.
        3. Daily Breakdown Table (shown when `show_daily=True` or when `show_daily is None`
           and active steps span multiple calendar dates).
        4. Top Sessions Table (with `Model` and `Thinking` columns, plus optional indented
           child subagent tree when `show_subagents=True`).
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

    lines.append("=" * 108)
    lines.append(f"Antigravity (agy) Token Usage Report — Window: {w.label}")
    lines.append("=" * 108)
    lines.append(f"  Time Window    : [{start_str} .. {end_str}]{tz_name} ({mode_str})")
    lines.append(
        f"  Sources Scanned: {', '.join(report.sources_scanned) if report.sources_scanned else 'none'}"
    )
    lines.append(f"  Hierarchy Mode : {hier_str}")
    if report.model_filter:
        lines.append(f"  Model Filter   : {report.model_filter}")
    lines.append(
        f"  Active Sessions: {report.active_root_sessions:,} root session(s), "
        f"{report.active_subagent_sessions:,} subagent session(s)"
    )
    lines.append(
        f"  Activity Counts: {ov.llm_calls:,} LLM calls ({ov.thinking_steps:,} w/ thinking) | "
        f"{ov.user_turns:,} user turns | {ov.tool_calls:,} tool calls"
    )
    lines.append(f"  Counter Source : {accuracy_str}")
    lines.append("-" * 108)
    lines.append(
        f"  Uncached Input : {_fmt_int(ov.uncached_input):>14} tokens"
        f"    Output Tokens         : {_fmt_int(ov.output_tokens):>14} tokens"
    )
    lines.append(
        f"  Cache Read     : {_fmt_int(ov.cache_read):>14} tokens"
        f"    Thinking Tokens (est.): {_fmt_int(ov.thinking_tokens):>14} tokens"
        f" ({_fmt_pct(ov.thinking_pct)} of output, {_fmt_int(ov.thinking_steps)}/{_fmt_int(ov.llm_calls)} calls)"
    )
    lines.append(
        f"  Gross Input    : {_fmt_int(ov.gross_input):>14} tokens"
        f"    Response Tokens (est.): {_fmt_int(ov.response_tokens):>14} tokens (content + tool calls)"
    )
    lines.append(
        f"  Cache Hit Rate : {_fmt_pct(ov.cache_hit_pct):>14}       "
        f"    Net Tokens            : {_fmt_int(ov.net_tokens):>14} tokens (uncached in + out)"
    )
    lines.append(
        f"  Peak Context   : {_fmt_int(ov.peak_context):>14} tokens"
        f"    Total Tokens          : {_fmt_int(ov.total_tokens):>14} tokens (gross in + out)"
    )
    lines.append("=" * 108)

    if not ov.has_activity and not report.sessions:
        lines.append("")
        lines.append(
            "No active Antigravity sessions or steps matched the selected filters."
        )
        return "\n".join(lines)

    # 2a. Breakdown by Source
    if report.by_source:
        lines.append("")
        lines.append("Breakdown by Source")
        src_headers = [
            "Source",
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
            "Share %",
        ]
        src_rows: list[list[str]] = []
        for src, m in report.by_source.items():
            share = (
                (m.total_tokens / ov.total_tokens * 100.0)
                if ov.total_tokens > 0
                else 0.0
            )
            src_rows.append(
                [
                    src,
                    _fmt_int(report.by_source_sessions.get(src, 0)),
                    _fmt_int(m.llm_calls),
                    _fmt_int(m.uncached_input),
                    _fmt_int(m.cache_read),
                    _fmt_pct(m.cache_hit_pct),
                    _fmt_int(m.output_tokens),
                    _fmt_int(m.thinking_tokens),
                    _fmt_pct(m.thinking_pct),
                    _fmt_int(m.net_tokens),
                    _fmt_int(m.total_tokens),
                    _fmt_int(m.peak_context),
                    _fmt_pct(share),
                ]
            )
        lines.extend(
            _render_aligned_table(
                src_headers,
                src_rows,
                right_align_cols={1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
            )
        )

    # 2b. Breakdown by Role
    if report.by_role:
        lines.append("")
        lines.append("Breakdown by Agent Role")
        role_headers = [
            "Role / Type",
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
            "Share %",
        ]
        role_rows: list[list[str]] = []
        for role, m in report.by_role.items():
            share = (
                (m.total_tokens / ov.total_tokens * 100.0)
                if ov.total_tokens > 0
                else 0.0
            )
            role_rows.append(
                [
                    role,
                    _fmt_int(report.by_role_sessions.get(role, 0)),
                    _fmt_int(m.llm_calls),
                    _fmt_int(m.uncached_input),
                    _fmt_int(m.cache_read),
                    _fmt_pct(m.cache_hit_pct),
                    _fmt_int(m.output_tokens),
                    _fmt_int(m.thinking_tokens),
                    _fmt_pct(m.thinking_pct),
                    _fmt_int(m.net_tokens),
                    _fmt_int(m.total_tokens),
                    _fmt_int(m.peak_context),
                    _fmt_pct(share),
                ]
            )
        lines.extend(
            _render_aligned_table(
                role_headers,
                role_rows,
                right_align_cols={1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
            )
        )

    # 2c. Breakdown by Model
    if report.by_model:
        lines.append("")
        lines.append("Breakdown by Model")
        model_headers = [
            "Model",
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
            "Share %",
        ]
        model_rows: list[list[str]] = []
        for model_name, m in report.by_model.items():
            share = (
                (m.total_tokens / ov.total_tokens * 100.0)
                if ov.total_tokens > 0
                else 0.0
            )
            model_rows.append(
                [
                    model_name,
                    _fmt_int(report.by_model_sessions.get(model_name, 0)),
                    _fmt_int(m.llm_calls),
                    _fmt_int(m.uncached_input),
                    _fmt_int(m.cache_read),
                    _fmt_pct(m.cache_hit_pct),
                    _fmt_int(m.output_tokens),
                    _fmt_int(m.thinking_tokens),
                    _fmt_pct(m.thinking_pct),
                    _fmt_int(m.net_tokens),
                    _fmt_int(m.total_tokens),
                    _fmt_int(m.peak_context),
                    _fmt_pct(share),
                ]
            )
        lines.extend(
            _render_aligned_table(
                model_headers,
                model_rows,
                right_align_cols={1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
            )
        )

    # 3. Daily Breakdown Table
    should_show_daily = (
        show_daily if show_daily is not None else (len(report.daily) > 1)
    )
    if should_show_daily and report.daily:
        lines.append("")
        lines.append("Daily Breakdown (Local Time)")
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
        ]
        day_rows: list[list[str]] = []
        for d in report.daily:
            dm = d.metrics
            day_rows.append(
                [
                    d.date_str,
                    _fmt_int(d.active_root_sessions),
                    _fmt_int(d.active_subagent_sessions),
                    _fmt_int(dm.llm_calls),
                    _fmt_int(dm.uncached_input),
                    _fmt_int(dm.cache_read),
                    _fmt_pct(dm.cache_hit_pct),
                    _fmt_int(dm.output_tokens),
                    _fmt_int(dm.thinking_tokens),
                    _fmt_pct(dm.thinking_pct),
                    _fmt_int(dm.net_tokens),
                    _fmt_int(dm.total_tokens),
                ]
            )
        lines.extend(
            _render_aligned_table(
                day_headers,
                day_rows,
                right_align_cols={1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
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
            lines.append(
                f"Top Sessions (showing {len(displayed_sessions)} of {total_sess})"
            )
        else:
            lines.append(f"Sessions ({total_sess})")

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
            "Title / Prompt Preview",
        ]
        sess_rows: list[list[str]] = []

        def _fmt_model_cell(raw_model: str, max_len: int = 32) -> str:
            label = (raw_model or "").strip() or "Unknown"
            if len(label) > max_len:
                return label[: max_len - 1].rstrip() + "…"
            return label

        def _append_session_rows(s: SessionSummary, depth: int = 0) -> None:
            tm = s.tree_metrics if depth == 0 else s.self_metrics
            short_id = s.conversation_id[:8]
            if depth > 0:
                prefix = "  " * (depth - 1) + "└─ "
                id_col = f"{prefix}{short_id}"
            else:
                id_col = short_id

            title_preview = s.title or s.workspace or "—"
            if len(title_preview) > 48:
                title_preview = title_preview[:47].rstrip() + "…"

            sess_rows.append(
                [
                    id_col,
                    s.source_kind,
                    s.agent_type,
                    _fmt_model_cell(
                        s.model if depth == 0 else (s.self_model or s.model)
                    ),
                    _fmt_short_dt(s.last_active_local),
                    _fmt_int(s.active_subagent_count) if depth == 0 else "—",
                    _fmt_int(tm.llm_calls),
                    _fmt_int(tm.uncached_input),
                    _fmt_int(tm.cache_read),
                    _fmt_pct(tm.cache_hit_pct),
                    _fmt_int(tm.output_tokens),
                    _fmt_int(tm.thinking_tokens),
                    _fmt_int(tm.net_tokens),
                    _fmt_int(tm.total_tokens),
                    _fmt_int(tm.peak_context),
                    title_preview,
                ]
            )
            if show_subagents and s.children:
                if depth == 0 and s.self_metrics.has_activity:
                    sm = s.self_metrics
                    sess_rows.append(
                        [
                            f"├─ {short_id} (self)",
                            s.source_kind,
                            f"{s.agent_type} (self)",
                            _fmt_model_cell(s.self_model or s.model),
                            _fmt_short_dt(s.last_active_local),
                            "—",
                            _fmt_int(sm.llm_calls),
                            _fmt_int(sm.uncached_input),
                            _fmt_int(sm.cache_read),
                            _fmt_pct(sm.cache_hit_pct),
                            _fmt_int(sm.output_tokens),
                            _fmt_int(sm.thinking_tokens),
                            _fmt_int(sm.net_tokens),
                            _fmt_int(sm.total_tokens),
                            _fmt_int(sm.peak_context),
                            "(parent session excluding subagents)",
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
                right_align_cols={5, 6, 7, 8, 9, 10, 11, 12, 13, 14},
            )
        )

    return "\n".join(lines)
