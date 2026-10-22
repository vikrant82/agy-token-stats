"""Command-line interface and argument parsing for `agy-token-stats`.

Error Behavior:
    - Invalid time window combinations or bounds invoke `parser.error(...)`, exiting with
      status code `2` and a descriptive message on stderr.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import datetime, tzinfo
from pathlib import Path

from agy_token_stats.aggregator import compute_stats_report
from agy_token_stats.discovery import (
    DEFAULT_CLI_BRAIN_DIR,
    DEFAULT_CLI_HISTORY_PATH,
    DEFAULT_IDE_BRAIN_DIR,
    discover_sessions,
    infer_source_kind,
)
from agy_token_stats.formatters import format_human_report, format_json_report
from agy_token_stats.time_window import get_default_local_tz, resolve_time_window


def build_arg_parser() -> argparse.ArgumentParser:
    """Create and return the `argparse.ArgumentParser` for `agy-token-stats`."""
    parser = argparse.ArgumentParser(
        description=(
            "Estimate and report token usage statistics across Antigravity (agy) "
            "CLI and IDE sessions, including subagent trees."
        )
    )
    parser.add_argument(
        "--source",
        choices=("all", "cli", "ide"),
        default="all",
        help="Brain source(s) to scan when --brain-dir is not specified (default: all)",
    )
    parser.add_argument(
        "--brain-dir",
        action="append",
        dest="brain_dirs",
        default=None,
        metavar="[LABEL=]PATH",
        help=(
            "Override brain root directory to scan (repeatable). "
            "Accepts PATH or LABEL=PATH (e.g. cli=/path/to/brain)."
        ),
    )
    parser.add_argument(
        "--history-file",
        type=Path,
        default=DEFAULT_CLI_HISTORY_PATH,
        help=f"Path to CLI history.jsonl (default: {DEFAULT_CLI_HISTORY_PATH})",
    )

    time_group = parser.add_argument_group(
        "Time Window Filtering (step timestamp in local time)"
    )
    time_group.add_argument(
        "--today",
        action="store_true",
        help="Include activity from today 00:00:00 local time to now",
    )
    time_group.add_argument(
        "--yesterday",
        action="store_true",
        help="Include activity from yesterday 00:00:00 to 23:59:59 local time",
    )
    time_group.add_argument(
        "--last-n-days",
        "--days",
        type=int,
        dest="last_n_days",
        default=None,
        metavar="N",
        help=(
            "Include activity from the last N calendar days in local time including today "
            "(start = today 00:00:00 minus (N-1) days; N=1 equals --today, N=7 is 7 calendar days)"
        ),
    )
    time_group.add_argument(
        "--last-n-hours",
        "--hours",
        type=float,
        dest="last_n_hours",
        default=None,
        metavar="H",
        help="Include activity from the rolling last H hours up to now",
    )
    time_group.add_argument(
        "--since",
        type=str,
        default=None,
        metavar="DATE_OR_ISO",
        help="Lower bound in local time (YYYY-MM-DD at 00:00:00 or ISO-8601 timestamp)",
    )
    time_group.add_argument(
        "--until",
        type=str,
        default=None,
        metavar="DATE_OR_ISO",
        help="Upper bound in local time (YYYY-MM-DD at 23:59:59.999999 or ISO-8601 timestamp)",
    )
    time_group.add_argument(
        "--by-session",
        action="store_true",
        help="Filter whole sessions by session start time instead of individual step timestamps",
    )

    view_group = parser.add_argument_group("Grouping & Output Options")
    view_group.add_argument(
        "--session",
        type=str,
        default=None,
        metavar="ID_OR_PREFIX",
        help="Filter to a specific conversation UUID (or prefix) and display its subagent tree",
    )
    view_group.add_argument(
        "--model",
        type=str,
        default=None,
        metavar="PATTERN",
        help=(
            "Filter steps/sessions by model name using a case-insensitive substring match "
            "(e.g. --model claude, --model flash, --model argon)"
        ),
    )
    view_group.add_argument(
        "--flat",
        action="store_true",
        help="Treat every conversation UUID independently without rolling subagents into parent roots",
    )
    view_group.add_argument(
        "--show-subagents",
        action="store_true",
        help="Display child subagent rows beneath each parent session in the sessions table",
    )
    view_group.add_argument(
        "--top",
        type=int,
        default=20,
        metavar="N",
        help="Number of sessions to display in the sessions table (0 for all; default: 20)",
    )
    daily_group = view_group.add_mutually_exclusive_group()
    daily_group.add_argument(
        "--daily",
        dest="show_daily",
        action="store_true",
        default=None,
        help="Always show the daily breakdown table",
    )
    daily_group.add_argument(
        "--no-daily",
        dest="show_daily",
        action="store_false",
        help="Hide the daily breakdown table",
    )
    view_group.add_argument(
        "--json",
        action="store_true",
        help="Emit structured JSON output instead of the human-readable report",
    )
    return parser


def resolve_brain_sources(
    source_flag: str, brain_dirs_arg: Sequence[str] | None
) -> tuple[list[tuple[str, Path]], list[str]]:
    """Resolve CLI `--source` and `--brain-dir` flags into `(brain_sources, source_labels)`."""
    if brain_dirs_arg:
        resolved: list[tuple[str, Path]] = []
        labels: list[str] = []
        for raw in brain_dirs_arg:
            if "=" in raw:
                label, path_str = raw.split("=", 1)
                label = label.strip() or "custom"
                path = Path(path_str.strip()).expanduser()
            else:
                path = Path(raw.strip()).expanduser()
                label = (
                    source_flag
                    if source_flag in ("cli", "ide")
                    else infer_source_kind(path)
                )
            resolved.append((label, path))
            if label not in labels:
                labels.append(label)
        return resolved, labels

    sources: list[tuple[str, Path]] = []
    labels = []
    if source_flag in ("all", "cli"):
        sources.append(("cli", DEFAULT_CLI_BRAIN_DIR))
        labels.append("cli")
    if source_flag in ("all", "ide"):
        sources.append(("ide", DEFAULT_IDE_BRAIN_DIR))
        labels.append("ide")
    return sources, labels


_resolve_brain_sources = resolve_brain_sources


def main(
    argv: Sequence[str] | None = None,
    *,
    now: datetime | None = None,
    local_tz: tzinfo | None = None,
) -> int:
    """CLI entry point for `agy-token-stats`. Returns exit code (`0` on success)."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    tz = local_tz or (now.tzinfo if now and now.tzinfo else get_default_local_tz())
    try:
        window = resolve_time_window(
            now=now,
            today=args.today,
            yesterday=args.yesterday,
            last_n_days=args.last_n_days,
            last_n_hours=args.last_n_hours,
            since=args.since,
            until=args.until,
            local_tz=tz,
        )
    except ValueError as exc:
        parser.error(str(exc))

    brain_sources, source_labels = resolve_brain_sources(
        args.source, args.brain_dirs
    )
    sessions = discover_sessions(
        brain_sources,
        history_path=args.history_file,
        local_tz=tz,
    )
    report = compute_stats_report(
        sessions,
        window=window,
        by_session=args.by_session,
        flat=args.flat,
        session_filter=args.session,
        model_filter=args.model,
        sources_scanned=source_labels,
    )

    if args.json:
        print(format_json_report(report))
    else:
        show_subs = bool(args.show_subagents or args.session)
        print(
            format_human_report(
                report,
                top_n=args.top,
                show_daily=args.show_daily,
                show_subagents=show_subs,
            )
        )
    return 0
