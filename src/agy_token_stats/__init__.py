"""Antigravity (`agy`) token usage statistics and subagent tree analyzer."""

from __future__ import annotations

from agy_token_stats.aggregator import (
    _collect_descendants,
    _find_root_session_id,
    _resolve_root_id,
    analyze_sessions,
    compute_stats_report,
)
from agy_token_stats.cli import (
    _resolve_brain_sources,
    build_arg_parser,
    main,
    resolve_brain_sources,
)
from agy_token_stats.discovery import (
    DEFAULT_CLI_BRAIN_DIR,
    DEFAULT_CLI_HISTORY_PATH,
    DEFAULT_IDE_BRAIN_DIR,
    SKIPPED_BRAIN_DIRS,
    _load_subagent_descriptors,
    _safe_is_dir,
    discover_sessions,
    infer_source_kind,
    load_cli_history_metadata,
)
from agy_token_stats.formatters import (
    format_human_report,
    format_json_report,
)
from agy_token_stats.models import (
    DailyStats,
    SessionRecord,
    SessionSummary,
    StatsReport,
    StepRecord,
    SubagentLink,
    TimeWindow,
    TokenMetrics,
)
from agy_token_stats.parser import (
    _clean_prompt_preview,
    _safe_is_file,
    _step_text_chars,
    parse_transcript_file,
)
from agy_token_stats.time_window import (
    get_default_local_tz,
    parse_bound_datetime,
    parse_iso_timestamp,
    resolve_time_window,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "DEFAULT_CLI_BRAIN_DIR",
    "DEFAULT_CLI_HISTORY_PATH",
    "DEFAULT_IDE_BRAIN_DIR",
    "SKIPPED_BRAIN_DIRS",
    "DailyStats",
    "SessionRecord",
    "SessionSummary",
    "StatsReport",
    "StepRecord",
    "SubagentLink",
    "TimeWindow",
    "TokenMetrics",
    "_clean_prompt_preview",
    "_collect_descendants",
    "_find_root_session_id",
    "_load_subagent_descriptors",
    "_resolve_brain_sources",
    "_resolve_root_id",
    "_safe_is_dir",
    "_safe_is_file",
    "_step_text_chars",
    "analyze_sessions",
    "build_arg_parser",
    "compute_stats_report",
    "discover_sessions",
    "format_human_report",
    "format_json_report",
    "get_default_local_tz",
    "infer_source_kind",
    "load_cli_history_metadata",
    "main",
    "parse_bound_datetime",
    "parse_iso_timestamp",
    "parse_transcript_file",
    "resolve_brain_sources",
    "resolve_time_window",
]
