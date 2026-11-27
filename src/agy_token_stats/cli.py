"""Command-line interface and argument parsing for `agy-token-stats`.

Error Behavior:
    - Invalid time window combinations or bounds invoke `parser.error(...)`, exiting with
      status code `2` and a descriptive message on stderr.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone, tzinfo
import os
from pathlib import Path
import sys

from agy_token_stats.aggregator import compute_stats_report
from agy_token_stats.discovery import (
    DEFAULT_CLI_BRAIN_DIR,
    DEFAULT_CLI_HISTORY_PATH,
    DEFAULT_IDE_BRAIN_DIR,
    discover_sessions,
    infer_source_kind,
)
from agy_token_stats.formatters import (
    format_human_report,
    format_json_report,
    format_quota_human,
    format_trailing_usage_human,
    resolve_color_enabled,
)
from agy_token_stats.models import SessionRecord, StatsReport, TimeWindow
from agy_token_stats.time_window import get_default_local_tz, resolve_time_window
from agy_token_stats.quota import (
    DEFAULT_LIMIT_USD,
    QuotaState,
    build_quota_report,
    build_trailing_usage_report,
    default_quota_path,
    format_quota_json,
    load_quota_state,
    parse_quota_limit,
    parse_reset_input,
    next_reset_boundary,
    save_quota_state,
    validate_quota_state,
)


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

    quota_group = parser.add_argument_group("Weekly Quota Estimate")
    quota_group.add_argument(
        "--quota",
        action="store_true",
        help="Show a local estimate for the manually anchored weekly quota cycle",
    )
    quota_group.add_argument(
        "--quota-reset",
        metavar="WHEN",
        help=(
            "Register the next reset as an offset ISO-8601 timestamp, compact countdown "
            "(e.g. 1h30m19s), or the full Antigravity reset banner"
        ),
    )
    quota_group.add_argument(
        "--quota-blocked",
        action="store_true",
        help="Mark the registered future reset as currently blocked by a manual observation",
    )
    quota_group.add_argument(
        "--quota-limit",
        metavar="USD",
        type=parse_quota_limit,
        default=None,
        help="Assumed USD budget for this report only (default: $200)",
    )
    quota_group.add_argument(
        "--quota-file",
        type=Path,
        default=None,
        metavar="PATH",
        help="Quota state file (default: XDG config directory)",
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
    color_group = view_group.add_mutually_exclusive_group()
    color_group.add_argument(
        "--color",
        choices=("auto", "always", "never"),
        default="auto",
        help="Colorize human-readable terminal output (default: auto)",
    )
    color_group.add_argument(
        "--no-color",
        dest="color",
        action="store_const",
        const="never",
        help="Disable ANSI color codes in human-readable output",
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


def _available_brain_sources(
    brain_sources: Sequence[tuple[str, Path]], source_labels: Sequence[str]
) -> tuple[list[tuple[str, Path]], list[str], list[str]]:
    """Keep searchable roots and report each configured label's availability."""
    available: list[tuple[str, Path]] = []
    available_labels: set[str] = set()
    unavailable_labels: set[str] = set()
    for label, path in brain_sources:
        if not os.access(path, os.X_OK):
            unavailable_labels.add(label)
            continue
        try:
            with os.scandir(path):
                pass
        except OSError:
            unavailable_labels.add(label)
            continue
        available.append((label, path))
        available_labels.add(label)
    scanned = [label for label in source_labels if label in available_labels]
    unavailable = [label for label in source_labels if label in unavailable_labels]
    return available, scanned, unavailable


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
    if args.quota:
        incompatible_filters = (
            args.today
            or args.yesterday
            or args.last_n_days is not None
            or args.last_n_hours is not None
            or args.since is not None
            or args.until is not None
            or args.by_session
            or args.session is not None
            or args.model is not None
        )
        if incompatible_filters:
            parser.error(
                "--quota cannot be combined with time-window, --by-session, "
                "--session, or --model filtering"
            )
        if now is not None and (now.tzinfo is None or now.utcoffset() is None):
            parser.error("--quota requires a timezone-aware clock")

        def quota_clock_now() -> datetime:
            """Use the injected deterministic clock or sample current UTC time."""
            return now.astimezone(timezone.utc) if now is not None else datetime.now(timezone.utc)

        command_now = (
            now.astimezone(timezone.utc)
            if now is not None
            else quota_clock_now() if args.quota_reset is not None else None
        )
        quota_path = (args.quota_file or default_quota_path()).expanduser()
        limit_usd = DEFAULT_LIMIT_USD if args.quota_limit is None else args.quota_limit
        explicit_state: QuotaState | None = None
        if args.quota_reset is not None:
            try:
                assert command_now is not None
                anchor, origin = parse_reset_input(args.quota_reset, entered_at=command_now)
            except ValueError as exc:
                parser.error(str(exc))
            explicit_state = QuotaState(anchor, origin)

        state = explicit_state
        should_save = explicit_state is not None
        if state is None:
            try:
                state = load_quota_state(quota_path)
            except ValueError as exc:
                parser.error(str(exc))
        observation_now = quota_clock_now()
        interactive = (
            not args.json
            and getattr(sys.stdin, "isatty", lambda: False)()
            and getattr(sys.stdout, "isatty", lambda: False)()
        )
        if args.quota_blocked and state is None and not interactive:
            parser.error(
                "--quota-blocked needs a registered future reset; provide a fresh "
                "notice/countdown or exact future time with --quota-reset"
            )

        def scan_quota_sources() -> tuple[
            dict[str, SessionRecord] | None, list[str], list[str]
        ]:
            """Resolve availability and discover the selected local brain roots."""
            configured_sources, configured_labels = resolve_brain_sources(
                args.source, args.brain_dirs
            )
            brain_sources, scanned_labels, missing_labels = _available_brain_sources(
                configured_sources, configured_labels
            )
            found = (
                discover_sessions(
                    brain_sources,
                    history_path=args.history_file,
                    local_tz=tz or timezone.utc,
                )
                if brain_sources
                else None
            )
            return found, scanned_labels, missing_labels

        def aggregate_trailing_usage(
            found: dict[str, SessionRecord] | None,
            labels: list[str],
            cutoff: datetime,
        ) -> StatsReport | None:
            if found is None:
                return None
            return compute_stats_report(
                found,
                window=TimeWindow(
                    label="trailing 7-day local usage",
                    start_local=cutoff - timedelta(hours=168),
                    end_local=cutoff,
                ),
                by_session=False,
                flat=False,
                sources_scanned=labels,
            )

        if state is None:
            if args.quota_blocked and not interactive:
                parser.error(
                    "--quota-blocked needs a registered future reset; provide a fresh "
                    "notice/countdown or exact future time with --quota-reset"
                )
            preview_sessions, preview_sources, preview_unavailable = scan_quota_sources()
            preview_now = quota_clock_now()
            trailing_stats = aggregate_trailing_usage(
                preview_sessions, preview_sources, preview_now
            )
            trailing_report = build_trailing_usage_report(
                trailing_stats,
                start_utc=preview_now - timedelta(hours=168),
                end_utc=preview_now,
                sources_scanned=preview_sources,
                sources_unavailable=preview_unavailable,
            )
            if interactive:
                print(
                    format_trailing_usage_human(
                        trailing_report, color=resolve_color_enabled(args.color)
                    )
                )
                try:
                    entered = input(
                        "\nNo weekly reset is registered. Enter one of these:\n"
                        "  A fresh countdown from Antigravity, e.g. 1h30m19s\n"
                        "  An exact reset time with UTC offset, e.g. 2026-11-27T20:15:00+05:30\n"
                        "  The full Antigravity weekly-limit notice (with a fresh countdown)\n"
                        "Reset (blank to skip): "
                    )
                except EOFError:
                    entered = ""
                if not entered.strip():
                    if args.quota_blocked:
                        parser.error(
                            "--quota-blocked requires a registered future reset; provide "
                            "a fresh notice/countdown or exact future time"
                        )
                    return 0
                observation_now = quota_clock_now()
                try:
                    anchor, origin = parse_reset_input(
                        entered.strip(), entered_at=observation_now
                    )
                except ValueError as exc:
                    parser.error(str(exc))
                state = QuotaState(anchor, origin)
                should_save = True
                # The preview predates the user's input and may miss logs written
                # while the prompt was open. Anchored usage gets a fresh scan below.
            else:
                report = build_quota_report(
                    None,
                    now_utc=preview_now,
                    local_tz=(local_tz if local_tz is not None else now.tzinfo if now else None),
                    state=None,
                    limit_usd=limit_usd,
                    sources_scanned=preview_sources,
                    sources_unavailable=preview_unavailable,
                )
                for field in (
                    "estimated_used_usd",
                    "estimated_remaining_usd",
                    "percent_used",
                    "priced_calls",
                    "unpriced_calls",
                    "llm_calls",
                    "estimated_token_steps",
                    "average_usd_per_day",
                    "projected_used_at_reset_usd",
                    "sustainable_remaining_usd_per_day",
                    "estimated_limit_hit_local",
                ):
                    report.pop(field, None)
                report["trailing_7_day_usage"] = trailing_report
                if args.json:
                    print(format_quota_json(report))
                else:
                    print(
                        format_trailing_usage_human(
                            trailing_report, color=resolve_color_enabled(args.color)
                        )
                    )
                    print(
                        "Reset boundary unknown. Register it with --quota-reset WHEN; "
                        "current-cycle usage and pace estimates are unavailable until a reset is known."
                    )
                return 0

        if args.quota_blocked:
            if state is None:
                parser.error(
                    "--quota-blocked needs a registered future reset; provide a fresh "
                    "notice/countdown or exact future time with --quota-reset"
                )
            if state.anchor_utc <= observation_now:
                parser.error(
                    "--quota-blocked needs a future reset boundary; provide a fresh "
                    "notice/countdown or exact future time with --quota-reset"
                )
            origin = (
                "manual_limit_notice"
                if state.origin == "manual_limit_notice"
                else "manual_blocked"
            )
            state = QuotaState(state.anchor_utc, origin)
            should_save = True
        assert state is not None
        try:
            validate_quota_state(state, path=quota_path, now_utc=observation_now)
            if should_save:
                save_quota_state(quota_path, state)
        except ValueError as exc:
            parser.error(str(exc))

        stats: StatsReport | None = None
        sessions, source_labels, unavailable_labels = scan_quota_sources()
        report_now = quota_clock_now()
        if sessions is not None:
            def aggregate_current_cycle(cutoff: datetime) -> StatsReport:
                next_reset, _ = next_reset_boundary(state.anchor_utc, cutoff)
                return compute_stats_report(
                    sessions,
                    window=TimeWindow(
                        label="anchored weekly quota cycle",
                        start_local=next_reset - timedelta(hours=168),
                        end_local=cutoff,
                    ),
                    by_session=False,
                    flat=False,
                    sources_scanned=source_labels,
                )

            first_boundary, _ = next_reset_boundary(state.anchor_utc, report_now)
            stats = aggregate_current_cycle(report_now)
            final_now = quota_clock_now()
            final_boundary, _ = next_reset_boundary(state.anchor_utc, final_now)
            if final_boundary != first_boundary:
                report_now = final_now
                stats = aggregate_current_cycle(report_now)
            else:
                report_now = final_now
        else:
            report_now = quota_clock_now()
        display_tz = (
            local_tz
            if local_tz is not None
            else now.tzinfo if now is not None else None
        )
        report = build_quota_report(
            stats,
            now_utc=report_now,
            local_tz=display_tz,
            state=state,
            limit_usd=limit_usd,
            sources_scanned=source_labels,
            sources_unavailable=unavailable_labels,
        )
        if args.json:
            print(format_quota_json(report))
        else:
            print(format_quota_human(report, color=resolve_color_enabled(args.color)))
        return 0
    if (
        args.quota_blocked
        or args.quota_reset is not None
        or args.quota_limit is not None
        or args.quota_file is not None
    ):
        parser.error("--quota-reset, --quota-blocked, --quota-limit, and --quota-file require --quota")
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
        use_color = resolve_color_enabled(args.color)
        print(
            format_human_report(
                report,
                top_n=args.top,
                show_daily=args.show_daily,
                show_subagents=show_subs,
                color=use_color,
            )
        )
    return 0
