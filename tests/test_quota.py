"""Public CLI behavior tests for the manually anchored weekly quota estimate."""

from __future__ import annotations

import contextlib
from datetime import datetime, timedelta, timezone, tzinfo
import errno
import io
import json
import os
from pathlib import Path
import pty
import re
import select
import sys
import time
from typing import Any
import unittest
from unittest import mock
from zoneinfo import ZoneInfo

from tests.helpers import TempBrainTestCase, write_jsonl

from agy_token_stats.cli import main

UTC = timezone.utc


class QuotaCliTests(TempBrainTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.brain = self.tmp_path / "brain"
        self.state = self.tmp_path / "quota.json"
        self.history = self.tmp_path / "missing-history.jsonl"

    def _invoke(
        self,
        now: datetime | None,
        *extra: str,
        json_output: bool = True,
        local_tz: tzinfo | None = UTC,
    ) -> str:
        output = io.StringIO()
        argv = [
            "--quota",
            "--quota-file",
            str(self.state),
            "--brain-dir",
            f"local={self.brain}",
            "--history-file",
            str(self.history),
            *extra,
        ]
        if json_output:
            argv.append("--json")
        with contextlib.redirect_stdout(output):
            self.assertEqual(main(argv, now=now, local_tz=local_tz), 0)
        return output.getvalue()

    def _write_calls(self, calls: list[tuple[datetime, str, int | None, int | None]]) -> None:
        records: list[dict | str] = []
        for index, (at, model, input_tokens, output_tokens) in enumerate(calls):
            records.append(
                {
                    "step_index": index * 2,
                    "source": "USER",
                    "type": "USER_INPUT",
                    "status": "DONE",
                    "created_at": at.isoformat().replace("+00:00", "Z"),
                    "content": (
                        "<USER_SETTINGS_CHANGE>The user changed setting Model Selection "
                        f"from Previous to {model}.</USER_SETTINGS_CHANGE>\n"
                        "<USER_REQUEST>Quota accounting</USER_REQUEST>"
                    ),
                }
            )
            response: dict[str, object] = {
                "step_index": index * 2 + 1,
                "source": "MODEL",
                "type": "PLANNER_RESPONSE",
                "status": "DONE",
                "created_at": at.isoformat().replace("+00:00", "Z"),
            }
            if input_tokens is not None:
                response["input_tokens"] = input_tokens
            if output_tokens is not None:
                response["output_tokens"] = output_tokens
            records.append(response)
        write_jsonl(
            self.brain / "session-1" / ".system_generated/logs/transcript.jsonl",
            records,
        )

    def _payload(self, text: str) -> dict[str, Any]:
        return json.loads(text)

    def test_iso_reset_persists_and_cycle_filters_steps_and_prices_pace(self) -> None:
        now = datetime(2026, 11, 27, 12, tzinfo=UTC)
        cycle_start = datetime(2026, 11, 21, 12, tzinfo=UTC)
        self._write_calls(
            [
                (cycle_start - timedelta(seconds=1), "Gemini 3.1 Pro", 100_000, 0),
                (cycle_start, "Gemini 3.1 Pro", 100_000, 0),
                (now, "Gemini 3.1 Pro", 100_000, 0),
                (now + timedelta(seconds=1), "Gemini 3.1 Pro", 100_000, 0),
            ]
        )

        report = self._payload(
            self._invoke(
                now,
                "--quota-reset",
                "2026-11-28T17:30:00+05:30",
                "--quota-limit",
                "0.45",
            )
        )

        self.assertEqual(report["cycle_start_local"], "2026-11-21T12:00:00+00:00")
        self.assertEqual(report["reset_local"], "2026-11-28T12:00:00+00:00")
        self.assertEqual(report["seconds_until_reset"], 86400)
        self.assertEqual(report["reset_source"], "manual_timestamp")
        self.assertEqual(report["quota_status"], "unverified")
        self.assertFalse(report["reset_projected"])
        self.assertEqual(report["sources_scanned"], ["local"])
        self.assertEqual(report["priced_calls"], 2)
        self.assertEqual(report["unpriced_calls"], 0)
        self.assertEqual(report["estimated_used_usd"], 0.4)
        self.assertEqual(report["estimated_remaining_usd"], 0.05)
        self.assertEqual(report["percent_used"], 88.89)
        self.assertEqual(report["average_usd_per_day"], round(0.4 / 6, 6))
        self.assertEqual(report["projected_used_at_reset_usd"], round(0.4 + 0.4 / 6, 6))
        self.assertEqual(report["sustainable_remaining_usd_per_day"], 0.05)
        self.assertEqual(report["estimated_limit_hit_local"], "2026-11-28T06:00:00+00:00")

        stored = json.loads(self.state.read_text(encoding="utf-8"))
        self.assertEqual(
            stored,
            {
                "version": 1,
                "anchor_utc": "2026-11-28T12:00:00Z",
                "origin": "manual_timestamp",
            },
        )

    def test_full_banner_countdown_is_anchored_once_and_rerun_does_not_move_it(self) -> None:
        now = datetime(2026, 11, 27, 12, tzinfo=UTC)
        self._write_calls([(now - timedelta(hours=1), "Gemini 3.1 Pro", 100_000, 0)])
        banner = (
            "You've reached your individual weekly usage limit set by your administrator. "
            "It resets in 1h30m19s. Contact your administrator if you need a higher limit."
        )
        first = self._payload(self._invoke(now, "--quota-reset", banner))
        saved = self.state.read_text(encoding="utf-8")
        self.assertEqual(json.loads(saved)["origin"], "manual_limit_notice")
        self.assertEqual(first["reset_source"], "manual_limit_notice")
        self.assertEqual(first["quota_status"], "blocked")
        self.assertEqual(first["reset_local"], "2026-11-27T13:30:19+00:00")
        self.assertEqual(first["estimated_used_usd"], 0.2)
        self.assertEqual(first["priced_calls"], 1)
        self.assertEqual(first["estimated_token_steps"], 0)
        for field in (
            "estimated_remaining_usd",
            "percent_used",
            "average_usd_per_day",
            "projected_used_at_reset_usd",
            "sustainable_remaining_usd_per_day",
            "estimated_limit_hit_local",
        ):
            self.assertIsNone(first[field], field)
        second = self._payload(self._invoke(now + timedelta(minutes=1)))
        self.assertEqual(second["reset_local"], first["reset_local"])
        self.assertEqual(second["seconds_until_reset"], 89 * 60 + 19)
        self.assertEqual(second["quota_status"], "blocked")
        self.assertEqual(self.state.read_text(encoding="utf-8"), saved)

        human = self._invoke(now, "--no-color", json_output=False)
        self.assertRegex(human, r"Status\s{2,}BLOCKED — weekly limit notice")
        self.assertRegex(human, r"Resets in\s{2,}1h 30m 19s")
        self.assertRegex(human, r"Reset at\s{2,}Fri Nov 27, 13:30:19 \+00:00")
        self.assertRegex(human, r"Spend in local logs\s{2,}\$0\.20")
        self.assertNotIn("Daily averages", human)
        for forbidden in ("budget left", "current pace", "total spend at reset", "/day"):
            self.assertNotIn(forbidden, human.lower())
        self.assertNotIn("manual_limit_notice", human)

        plain_reset = self._payload(
            self._invoke(now, "--quota-reset", "1h30m19s")
        )
        self.assertEqual(plain_reset["quota_status"], "unverified")
        self.assertEqual(plain_reset["reset_source"], "manual_countdown")

    def test_legacy_countdown_can_be_marked_blocked_without_reentering_anchor(self) -> None:
        now = datetime(2026, 11, 27, 12, tzinfo=UTC)
        anchor = "2026-11-27T13:30:00Z"
        self.state.write_text(
            json.dumps(
                {"version": 1, "anchor_utc": anchor, "origin": "manual_countdown"}
            ),
            encoding="utf-8",
        )
        report = self._payload(self._invoke(now, "--quota-blocked"))
        saved = json.loads(self.state.read_text(encoding="utf-8"))
        self.assertEqual(report["quota_status"], "blocked")
        self.assertEqual(saved["origin"], "manual_blocked")
        self.assertEqual(saved["anchor_utc"], anchor)

    def test_explicit_blocked_flag_works_with_future_reset_and_preserves_notice_evidence(self) -> None:
        now = datetime(2026, 11, 27, 12, tzinfo=UTC)
        explicit = self._payload(
            self._invoke(now, "--quota-reset", "1h", "--quota-blocked")
        )
        self.assertEqual(explicit["quota_status"], "blocked")
        self.assertEqual(explicit["reset_source"], "manual_blocked")

        exact_time = self._payload(
            self._invoke(
                now,
                "--quota-reset",
                "2026-11-27T13:00:00Z",
                "--quota-blocked",
            )
        )
        self.assertEqual(exact_time["quota_status"], "blocked")
        self.assertEqual(exact_time["seconds_until_reset"], 3600)

        notice = (
            "You've reached your individual weekly usage limit set by your administrator. "
            "It resets in 1h30m19s. Contact your administrator if you need a higher limit."
        )
        noticed = self._payload(
            self._invoke(now, "--quota-reset", notice, "--quota-blocked")
        )
        self.assertEqual(noticed["quota_status"], "blocked")
        self.assertEqual(noticed["reset_source"], "manual_limit_notice")

    def test_bare_countdown_and_exact_reset_are_unverified_not_blocked(self) -> None:
        now = datetime(2026, 11, 27, 12, tzinfo=UTC)
        self._write_calls([(now - timedelta(hours=1), "Gemini 3.1 Pro", 100_000, 0)])
        report = self._payload(self._invoke(now, "--quota-reset", "1h30m19s"))
        self.assertEqual(report["quota_status"], "unverified")
        self.assertEqual(report["estimated_remaining_usd"], 199.8)
        self.assertIsNotNone(report["average_usd_per_day"])

        anchor = now + timedelta(hours=1)
        self.state.write_text(
            json.dumps(
                {
                    "version": 1,
                    "anchor_utc": anchor.isoformat().replace("+00:00", "Z"),
                    "origin": "manual_blocked",
                }
            ),
            encoding="utf-8",
        )
        exact = self._payload(self._invoke(anchor))
        self.assertEqual(exact["quota_status"], "unverified")
        self.assertTrue(exact["reset_projected"])
        self.assertEqual(exact["seconds_until_reset"], 604800)
        later = self._payload(self._invoke(anchor + timedelta(seconds=1)))
        self.assertEqual(later["quota_status"], "unverified")
        self.assertTrue(later["reset_projected"])
        self.assertEqual(later["seconds_until_reset"], 604799)

    def test_blocked_flag_rejects_missing_or_expired_anchor_without_writing(self) -> None:
        now = datetime(2026, 11, 27, 12, tzinfo=UTC)
        with contextlib.redirect_stderr(io.StringIO()) as missing_error:
            with self.assertRaises(SystemExit) as missing:
                self._invoke(now, "--quota-blocked")
        self.assertEqual(missing.exception.code, 2)
        self.assertFalse(self.state.exists())
        self.assertIn("fresh notice/countdown or exact future time", missing_error.getvalue())

        original = json.dumps(
            {
                "version": 1,
                "anchor_utc": (now - timedelta(seconds=1)).isoformat(),
                "origin": "manual_countdown",
            }
        )
        self.state.write_text(original, encoding="utf-8")
        with contextlib.redirect_stderr(io.StringIO()) as expired_error:
            with self.assertRaises(SystemExit) as expired:
                self._invoke(now, "--quota-blocked")
        self.assertEqual(expired.exception.code, 2)
        self.assertIn("future", expired_error.getvalue())
        self.assertEqual(self.state.read_text(encoding="utf-8"), original)

        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as past_override:
                self._invoke(
                    now,
                    "--quota-reset",
                    "2026-11-27T11:59:59Z",
                    "--quota-blocked",
                )
        self.assertEqual(past_override.exception.code, 2)
        self.assertEqual(self.state.read_text(encoding="utf-8"), original)

    def test_exact_anchor_starts_new_period_and_step_at_anchor_is_included(self) -> None:
        now = datetime(2026, 11, 27, 12, tzinfo=UTC)
        self._write_calls(
            [
                (now - timedelta(seconds=1), "Gemini 3.1 Pro", 100_000, 0),
                (now, "Gemini 3.1 Pro", 100_000, 0),
                (now + timedelta(seconds=1), "Gemini 3.1 Pro", 100_000, 0),
            ]
        )
        report = self._payload(
            self._invoke(now, "--quota-reset", "2026-11-27T12:00:00Z")
        )
        self.assertEqual(report["cycle_start_local"], now.isoformat())
        self.assertEqual(report["reset_local"], (now + timedelta(days=7)).isoformat())
        self.assertTrue(report["reset_projected"])
        self.assertEqual(report["seconds_until_reset"], 604800)
        self.assertEqual(report["priced_calls"], 1)
        self.assertEqual(report["estimated_used_usd"], 0.2)
        self.assertIsNone(report["average_usd_per_day"])

    def test_partial_and_only_unpriced_usage_are_not_misreported_as_zero(self) -> None:
        now = datetime(2026, 11, 27, 12, tzinfo=UTC)
        reset = "2026-11-28T12:00:00Z"
        self._write_calls(
            [
                (now - timedelta(hours=2), "Gemini 3.1 Pro", 100_000, 0),
                (now - timedelta(hours=1), "Claude Opus 4.6", None, None),
            ]
        )
        partial = self._payload(self._invoke(now, "--quota-reset", reset))
        self.assertEqual(partial["estimated_used_usd"], 0.2)
        self.assertEqual(partial["priced_calls"], 1)
        self.assertEqual(partial["unpriced_calls"], 1)
        self.assertEqual(partial["estimated_token_steps"], 1)
        self.assertEqual(partial["average_usd_per_day"], round(0.2 / 6, 6))

        self._write_calls([(now - timedelta(hours=2), "Claude Opus 4.6", 1_000_000, 0)])
        unpriced = self._payload(self._invoke(now, "--quota-reset", reset))
        self.assertIsNone(unpriced["estimated_used_usd"])
        self.assertIsNone(unpriced["estimated_remaining_usd"])
        self.assertEqual(unpriced["unpriced_calls"], 1)
        self.assertIsNone(unpriced["average_usd_per_day"])

    def test_missing_noninteractive_json_is_unknown_and_does_not_create_state(self) -> None:
        now = datetime(2026, 11, 27, 12, tzinfo=UTC)
        self.brain.mkdir()
        report = self._payload(self._invoke(now))
        self.assertEqual(report["quota_status"], "unverified")
        self.assertIsNone(report["reset_local"])
        self.assertIsNone(report["cycle_start_local"])
        self.assertIsNone(report["seconds_until_reset"])
        self.assertIn("--quota-reset WHEN", report["human_action"])
        self.assertIn("current-cycle usage and pace estimates", report["human_action"])
        trailing = report["trailing_7_day_usage"]
        self.assertEqual(trailing["estimated_used_usd"], 0.0)
        self.assertEqual(trailing["average_usd_per_day"], 0.0)
        self.assertEqual(trailing["llm_calls"], 0)
        self.assertEqual(trailing["sources_scanned"], ["local"])
        self.assertNotIn("estimated_used_usd", report)
        self.assertNotIn("estimated_remaining_usd", report)
        self.assertNotIn("average_usd_per_day", report)
        self.assertNotIn("projected_used_at_reset_usd", report)
        self.assertEqual(report["sources_scanned"], ["local"])
        self.assertEqual(report["sources_unavailable"], [])
        self.assertFalse(self.state.exists())

    def test_trailing_usage_uses_inclusive_168h_bounds_and_local_call_counts(self) -> None:
        now = datetime(2026, 11, 27, 12, tzinfo=UTC)
        start = now - timedelta(days=7)
        self._write_calls(
            [
                (start - timedelta(seconds=1), "Gemini 3.1 Pro", 100_000, 0),
                (start, "Gemini 3.1 Pro", 100_000, 0),
                (now, "Gemini 3.1 Pro", 100_000, 0),
                (now + timedelta(seconds=1), "Gemini 3.1 Pro", 100_000, 0),
            ]
        )

        report = self._payload(self._invoke(now))
        trailing = report["trailing_7_day_usage"]
        self.assertEqual(trailing["start_utc"], start.isoformat())
        self.assertEqual(trailing["end_utc"], now.isoformat())
        self.assertEqual(trailing["estimated_used_usd"], 0.4)
        self.assertEqual(trailing["average_usd_per_day"], round(0.4 / 7, 6))
        self.assertEqual(trailing["llm_calls"], 2)
        self.assertEqual(trailing["priced_calls"], 2)
        self.assertEqual(trailing["unpriced_calls"], 0)
        self.assertEqual(trailing["sources_unavailable"], [])
        self.assertFalse(self.state.exists())

    def test_trailing_usage_with_no_accessible_roots_is_unknown_not_zero(self) -> None:
        now = datetime(2026, 11, 27, 12, tzinfo=UTC)
        report = self._payload(self._invoke(now))
        trailing = report["trailing_7_day_usage"]
        self.assertIsNone(trailing["estimated_used_usd"])
        self.assertIsNone(trailing["average_usd_per_day"])
        self.assertIsNone(trailing["llm_calls"])
        self.assertEqual(trailing["sources_scanned"], [])
        self.assertEqual(trailing["sources_unavailable"], ["local"])
        self.assertFalse(self.state.exists())

    def test_noninteractive_human_fallback_guides_reset_registration(self) -> None:
        self.brain.mkdir()
        output = self._invoke(
            datetime(2026, 11, 27, 12, tzinfo=UTC),
            json_output=False,
        )
        self.assertIn("Last 7 days · local usage", output)
        self.assertIn(
            "Reset boundary unknown. Register it with --quota-reset WHEN; current-cycle "
            "usage and pace estimates are unavailable until a reset is known.",
            output,
        )

    def test_available_empty_and_unavailable_source_roots_are_distinguished(self) -> None:
        now = datetime(2026, 11, 27, 12, tzinfo=UTC)
        missing = self._payload(
            self._invoke(now, "--quota-reset", "2026-11-28T12:00:00Z")
        )
        self.assertEqual(missing["sources_scanned"], [])
        self.assertEqual(missing["sources_unavailable"], ["local"])
        self.assertIsNone(missing["llm_calls"])
        self.assertIsNone(missing["estimated_used_usd"])

        self.brain.mkdir()
        empty = self._payload(self._invoke(now))
        self.assertEqual(empty["sources_scanned"], ["local"])
        self.assertEqual(empty["sources_unavailable"], [])
        self.assertEqual(empty["llm_calls"], 0)
        self.assertEqual(empty["estimated_used_usd"], 0.0)

        self._write_calls([(now - timedelta(hours=2), "Gemini 3.1 Pro", 100_000, 0)])
        partial = self._payload(
            self._invoke(
                now,
                "--brain-dir",
                f"missing={self.tmp_path / 'not-created'}",
            )
        )
        self.assertEqual(partial["sources_scanned"], ["local"])
        self.assertEqual(partial["sources_unavailable"], ["missing"])
        self.assertEqual(partial["llm_calls"], 1)
        self.assertEqual(partial["estimated_used_usd"], 0.2)
        self.assertIn("missing", partial["local_coverage_note"])

        same_label = self._payload(
            self._invoke(
                now,
                "--brain-dir",
                f"local={self.tmp_path / 'same-label-missing'}",
            )
        )
        self.assertEqual(same_label["sources_scanned"], ["local"])
        self.assertEqual(same_label["sources_unavailable"], ["local"])
        self.assertEqual(same_label["estimated_used_usd"], 0.2)

    @unittest.skipUnless(os.name == "posix", "directory traversal permissions require POSIX")
    def test_root_without_search_permission_is_unavailable(self) -> None:
        now = datetime(2026, 11, 27, 12, tzinfo=UTC)
        root = self.tmp_path / "no-search"
        root.mkdir()
        original_mode = root.stat().st_mode & 0o777
        try:
            root.chmod(0o400)
            if os.access(root, os.X_OK):
                self.skipTest("effective user can traverse a mode-0400 root")
            report_text = io.StringIO()
            argv = [
                "--quota",
                "--quota-file",
                str(self.state),
                "--brain-dir",
                f"locked={root}",
                "--history-file",
                str(self.history),
                "--quota-reset",
                "2026-11-28T12:00:00Z",
                "--json",
            ]
            with contextlib.redirect_stdout(report_text):
                self.assertEqual(main(argv, now=now, local_tz=UTC), 0)
            report = self._payload(report_text.getvalue())
            self.assertEqual(report["sources_scanned"], [])
            self.assertEqual(report["sources_unavailable"], ["locked"])
            self.assertIsNone(report["llm_calls"])
            self.assertIsNone(report["estimated_used_usd"])
        finally:
            root.chmod(original_mode)

    def test_saved_future_anchor_and_invalid_utf8_fail_with_recovery_path(self) -> None:
        now = datetime(2026, 11, 27, 12, tzinfo=UTC)
        too_far = {
            "version": 1,
            "anchor_utc": "2026-12-05T13:00:00Z",
            "origin": "manual_timestamp",
        }
        self.state.write_text(json.dumps(too_far), encoding="utf-8")
        for bad_state in (json.dumps(too_far).encode(), b"\xff\xfe\xfa"):
            self.state.write_bytes(bad_state)
            with contextlib.redirect_stderr(io.StringIO()) as stderr:
                with self.assertRaises(SystemExit) as raised:
                    self._invoke(now)
            self.assertEqual(raised.exception.code, 2)
            self.assertIn(str(self.state), stderr.getvalue())
            self.assertIn("--quota-reset", stderr.getvalue())
            recovered = self._payload(
                self._invoke(now, "--quota-reset", "2026-11-28T12:00:00Z")
            )
            self.assertEqual(recovered["reset_source"], "manual_timestamp")

    def test_dst_localization_uses_offsets_at_each_cycle_boundary(self) -> None:
        from agy_token_stats.quota import QuotaState, build_quota_report

        previous_tz = os.environ.get("TZ")
        os.environ["TZ"] = "America/New_York"
        time.tzset()
        try:
            now = datetime(2026, 11, 1, 1, 30, tzinfo=ZoneInfo("America/New_York"), fold=1)
            reset = datetime(2026, 11, 7, 6, 30, tzinfo=UTC)
            report = build_quota_report(
                None,
                now_utc=now.astimezone(UTC),
                local_tz=None,
                state=QuotaState(reset, "manual_timestamp"),
                sources_scanned=[],
            )
            self.assertEqual(report["cycle_start_local"], "2026-10-31T02:30:00-04:00")
            self.assertEqual(report["reset_local"], "2026-11-07T01:30:00-05:00")
        finally:
            if previous_tz is None:
                os.environ.pop("TZ", None)
            else:
                os.environ["TZ"] = previous_tz
            time.tzset()

    @unittest.skipUnless(os.name == "posix", "interactive prompt timing requires a POSIX PTY")
    def test_interactive_countdown_anchor_uses_entry_time(self) -> None:
        self.brain.mkdir()
        pid, master = pty.fork()
        if pid == 0:
            source_dir = str(Path(__file__).resolve().parents[1] / "src")
            child_env = os.environ.copy()
            inherited_pythonpath = child_env.get("PYTHONPATH")
            child_env["PYTHONPATH"] = os.pathsep.join(
                part for part in (source_dir, inherited_pythonpath) if part
            )
            child_code = "from agy_token_stats.cli import main; raise SystemExit(main())"
            child_argv = [
                sys.executable,
                "-c",
                child_code,
                "--quota",
                    "--quota-file",
                    str(self.state),
                    "--brain-dir",
                    f"local={self.brain}",
                "--history-file",
                str(self.history),
            ]
            os.execvpe(sys.executable, child_argv, child_env)
            os._exit(127)

        output = bytearray()
        status: int | None = None

        def read_pty() -> bytes:
            try:
                return os.read(master, 4096)
            except OSError as exc:
                if exc.errno == errno.EIO:
                    return b""
                raise

        try:
            deadline = time.monotonic() + 4
            while b"blank to skip" not in output:
                child_pid, child_status = os.waitpid(pid, os.WNOHANG)
                if child_pid:
                    status = child_status
                    break
                readable, _, _ = select.select([master], [], [], 0.5)
                if readable:
                    output.extend(read_pty())
                if time.monotonic() >= deadline:
                    self.fail(f"quota prompt did not appear: {output.decode(errors='replace')}")
            if b"blank to skip" not in output:
                self.fail(f"quota child exited before prompt: {output.decode(errors='replace')}")
            preview_position = output.find(b"Last 7 days")
            prompt_position = output.find(b"Reset (blank to skip):")
            self.assertGreaterEqual(preview_position, 0, output.decode(errors="replace"))
            self.assertLess(preview_position, prompt_position)
            self.assertIn(b"not your Google quota window", output)
            time.sleep(0.35)
            self._write_calls(
                [(datetime.now(UTC), "Gemini 3.1 Pro", 100_000, 0)]
            )
            entered_at = datetime.now(UTC)
            os.write(master, b"5s\n")
            deadline = time.monotonic() + 4
            while status is None and time.monotonic() < deadline:
                child_pid, child_status = os.waitpid(pid, os.WNOHANG)
                if child_pid:
                    status = child_status
                    continue
                readable, _, _ = select.select([master], [], [], 0.1)
                if readable:
                    output.extend(read_pty())
            self.assertIsNotNone(status, f"quota command hung: {output.decode(errors='replace')}")
        finally:
            if status is None:
                try:
                    os.kill(pid, 9)
                    os.waitpid(pid, 0)
                except OSError:
                    pass
            os.close(master)

        assert status is not None
        self.assertEqual(os.waitstatus_to_exitcode(status), 0, output.decode(errors="replace"))
        output_text = output.decode(errors="replace")
        self.assertLess(
            output_text.find("Last 7 days · local usage"),
            output_text.find("Reset (blank to skip):"),
        )
        self.assertEqual(output_text.count("Antigravity weekly quota"), 1)
        self.assertIn("$0.20", output_text)
        payload = json.loads(self.state.read_text(encoding="utf-8"))
        self.assertEqual(payload["origin"], "manual_countdown")
        anchor = datetime.fromisoformat(payload["anchor_utc"].replace("Z", "+00:00"))
        estimated_entry = anchor - timedelta(seconds=5)
        self.assertGreaterEqual(estimated_entry, entered_at - timedelta(milliseconds=100))
        self.assertLessEqual(estimated_entry, entered_at + timedelta(seconds=1))

    @unittest.skipUnless(os.name == "posix", "interactive prompt timing requires a POSIX PTY")
    def test_interactive_fallback_blank_exits_without_repeating_preview_or_writing(self) -> None:
        self.brain.mkdir()
        pid, master = pty.fork()
        if pid == 0:
            source_dir = str(Path(__file__).resolve().parents[1] / "src")
            child_env = os.environ.copy()
            child_env["PYTHONPATH"] = os.pathsep.join(
                part for part in (source_dir, child_env.get("PYTHONPATH")) if part
            )
            child_code = "from agy_token_stats.cli import main; raise SystemExit(main())"
            os.execvpe(
                sys.executable,
                [
                    sys.executable,
                    "-c",
                    child_code,
                    "--quota",
                    "--quota-file",
                    str(self.state),
                    "--brain-dir",
                    f"local={self.brain}",
                    "--history-file",
                    str(self.history),
                ],
                child_env,
            )
            os._exit(127)

        output = bytearray()
        status: int | None = None
        try:
            deadline = time.monotonic() + 4
            while b"Reset (blank to skip):" not in output:
                child_pid, child_status = os.waitpid(pid, os.WNOHANG)
                if child_pid:
                    status = child_status
                    break
                readable, _, _ = select.select([master], [], [], 0.5)
                if readable:
                    try:
                        output.extend(os.read(master, 4096))
                    except OSError as exc:
                        if exc.errno != errno.EIO:
                            raise
                if time.monotonic() >= deadline:
                    self.fail(f"quota prompt did not appear: {output.decode(errors='replace')}")
            self.assertIn(b"Last 7 days", output)
            self.assertLess(output.find(b"Last 7 days"), output.find(b"Reset (blank to skip):"))
            os.write(master, b"\n")
            deadline = time.monotonic() + 4
            while status is None and time.monotonic() < deadline:
                child_pid, child_status = os.waitpid(pid, os.WNOHANG)
                if child_pid:
                    status = child_status
                    continue
                readable, _, _ = select.select([master], [], [], 0.1)
                if readable:
                    try:
                        output.extend(os.read(master, 4096))
                    except OSError as exc:
                        if exc.errno != errno.EIO:
                            raise
            self.assertIsNotNone(status, f"quota command hung: {output.decode(errors='replace')}")
        finally:
            if status is None:
                try:
                    os.kill(pid, 9)
                    os.waitpid(pid, 0)
                except OSError:
                    pass
            os.close(master)

        assert status is not None
        self.assertEqual(os.waitstatus_to_exitcode(status), 0, output.decode(errors="replace"))
        self.assertEqual(output.count(b"Last 7 days"), 1)
        self.assertFalse(self.state.exists())

    def test_human_countdown_and_times_use_seconds_without_losing_json_precision(self) -> None:
        now = datetime(2026, 11, 27, 12, tzinfo=UTC)
        precise = self._payload(
            self._invoke(now, "--quota-reset", "2026-11-28T12:00:00.123456Z")
        )
        self.assertEqual(precise["reset_local"], "2026-11-28T12:00:00.123456+00:00")
        human = self._invoke(
            now,
            "--quota-reset",
            "2026-11-28T12:00:00.123456Z",
            json_output=False,
        )
        self.assertIn("1d", human)
        self.assertIn("Sat Nov 28, 12:00:00 +00:00", human)
        self.assertNotIn(".123456", human)

        report = self._payload(
            self._invoke(now, "--quota-reset", "1h30m19s")
        )
        human_countdown = self._invoke(
            now,
            "--quota-reset",
            "1h30m19s",
            json_output=False,
        )
        self.assertEqual(report["seconds_until_reset"], 5419)
        self.assertIn("1h 30m 19s", human_countdown)

    def test_cli_abbreviated_quota_modifiers_still_require_quota(self) -> None:
        modifier_args = (("--quota-l", "200"), ("--quota-b",))
        for modifier in modifier_args:
            with self.subTest(modifier=modifier):
                argv = [
                    *modifier,
                    "--brain-dir",
                    f"local={self.brain}",
                    "--history-file",
                    str(self.history),
                ]
                with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaises(SystemExit) as raised:
                        main(argv, now=datetime(2026, 11, 27, 12, tzinfo=UTC), local_tz=UTC)
                self.assertEqual(raised.exception.code, 2)

    def test_finite_extreme_limits_produce_strict_json_values(self) -> None:
        now = datetime(2026, 11, 27, 11, 59, 59, tzinfo=UTC)
        self._write_calls([(now - timedelta(hours=2), "Gemini 3.1 Pro", 100_000, 0)])
        for limit in ("1e308", "1e-320"):
            with self.subTest(limit=limit):
                report = self._payload(
                    self._invoke(
                        now,
                        "--quota-reset",
                        "2026-11-27T12:00:00Z",
                        "--quota-limit",
                        limit,
                    )
                )
                self.assertIsNone(report["sustainable_remaining_usd_per_day"] if limit == "1e308" else report["percent_used"])

    def test_human_report_qualifies_local_budget_estimate(self) -> None:
        now = datetime(2026, 11, 27, 12, tzinfo=UTC)
        self._write_calls([(now, "Gemini 3.1 Pro", 100_000, 0)])
        output = self._invoke(
            now,
            "--quota-reset",
            "2026-11-28T12:00:00Z",
            "--no-color",
            json_output=False,
        )
        self.assertIn("Antigravity weekly quota", output)
        self.assertRegex(output, r"Status\s{2,}Unverified")
        self.assertIn("Local usage estimate", output)
        self.assertRegex(output, r"Sources\s{2,}local")
        self.assertIn("not Google's quota meter", output)
        self.assertRegex(output, r"Est\. budget left until reset\s{2,}\$199\.80")
        self.assertNotIn("total spend at reset", output)
        self.assertIn("Daily averages", output)
        self.assertIn("Avg. local spend / day", output)
        self.assertIn("Budget / day (7 days)", output)
        self.assertIn("Est. budget left / day", output)
        self.assertRegex(output, r"\$0\.03\s{2,}\$28\.57\s{2,}\$199\.80")
        self.assertNotIn("manual_timestamp", output)
        self.assertNotIn("168-hour", output)

    def test_human_report_uses_dollars_left_not_a_daily_allowance_near_reset(self) -> None:
        now = datetime(2026, 11, 27, 12, tzinfo=UTC)
        self._write_calls([(now, "Gemini 3.1 Pro", 100_000, 0)])
        output = self._invoke(now, "--quota-reset", "1h", "--no-color", json_output=False)
        self.assertRegex(output, r"Est\. budget left until reset\s{2,}\$199\.80")
        self.assertNotIn("Est. budget left / day", output)

    def test_quota_tables_keep_alignment_with_and_without_color(self) -> None:
        now = datetime(2026, 11, 27, 12, tzinfo=UTC)
        args = ("--quota-blocked", "--quota-reset", "1h30m19s")
        colored = self._invoke(now, *args, "--color", "always", json_output=False)
        plain = self._invoke(now, *args, "--no-color", json_output=False)
        self.assertRegex(colored, r"\x1b\[1;?m\x1b\[91mBLOCKED")
        self.assertNotIn("\x1b[", plain)
        self.assertEqual(re.sub(r"\x1b\[[0-9;]*m", "", colored), plain)

    def test_invalid_reset_filter_and_malformed_state_never_clobber_state(self) -> None:
        now = datetime(2026, 11, 27, 12, tzinfo=UTC)
        original = '{"preserve":"this"}\n'
        self.state.write_text(original, encoding="utf-8")
        for reset in (
            "2026-11-28",
            "tomorrow",
            "0s",
            "169h",
            "2026-12-05T12:00:00Z",
            "0001-01-01T00:00:00+05:30",
        ):
            with self.subTest(reset=reset):
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as raised:
                        self._invoke(now, "--quota-reset", reset)
                self.assertEqual(raised.exception.code, 2)
                self.assertEqual(self.state.read_text(encoding="utf-8"), original)

        for filter_args in (("--today",), ("--by-session",), ("--session", "abc"), ("--model", "gemini")):
            with self.subTest(filter_args=filter_args):
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as raised:
                        self._invoke(
                            now,
                            "--quota-reset",
                            "2026-11-28T12:00:00Z",
                            *filter_args,
                        )
                self.assertEqual(raised.exception.code, 2)
                self.assertEqual(self.state.read_text(encoding="utf-8"), original)

        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                self._invoke(now)
        self.assertEqual(raised.exception.code, 2)
        self.assertEqual(self.state.read_text(encoding="utf-8"), original)

    def test_explicit_reset_replaces_malformed_state_and_modifiers_need_quota(self) -> None:
        now = datetime(2026, 11, 27, 12, tzinfo=UTC)
        self.state.write_text("{broken", encoding="utf-8")
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                main(
                    [
                        "--quota",
                        "--quota-file",
                        str(self.state),
                        "--brain-dir",
                        f"local={self.brain}",
                        "--history-file",
                        str(self.history),
                        "--json",
                    ],
                    now=now,
                    local_tz=UTC,
                )
        self.assertEqual(self.state.read_text(encoding="utf-8"), "{broken")

        report = self._payload(
            self._invoke(now, "--quota-reset", "2026-11-28T12:00:00Z")
        )
        self.assertEqual(report["reset_source"], "manual_timestamp")
        self.assertEqual(json.loads(self.state.read_text())["version"], 1)

        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                main(["--quota-limit", "200"], now=now, local_tz=UTC)
        self.assertEqual(raised.exception.code, 2)

    def test_reset_crossing_during_aggregation_recomputes_new_cycle_once(self) -> None:
        anchor = datetime(2026, 11, 27, 12, 0, 1, tzinfo=UTC)
        observation_now = anchor - timedelta(seconds=2)
        snapshot_now = anchor - timedelta(seconds=1)
        final_now = anchor + timedelta(seconds=1)
        self.state.write_text(
            json.dumps(
                {
                    "version": 1,
                    "anchor_utc": anchor.isoformat().replace("+00:00", "Z"),
                    "origin": "manual_limit_notice",
                }
            ),
            encoding="utf-8",
        )
        self._write_calls(
            [
                (anchor - timedelta(seconds=1), "Gemini 3.1 Pro", 100_000, 0),
                (anchor + timedelta(seconds=1), "Gemini 3.1 Pro", 100_000, 0),
            ]
        )

        class SequencedDateTime(datetime):
            samples = [observation_now, snapshot_now, final_now]

            @classmethod
            def now(cls, tz: tzinfo | None = None) -> datetime:
                sample = cls.samples.pop(0)
                return sample.astimezone(tz) if tz is not None else sample

        output = io.StringIO()
        argv = [
            "--quota",
            "--quota-file",
            str(self.state),
            "--brain-dir",
            f"local={self.brain}",
            "--history-file",
            str(self.history),
            "--json",
        ]
        before_state = self.state.read_bytes()
        with mock.patch("agy_token_stats.cli.datetime", SequencedDateTime):
            with contextlib.redirect_stdout(output):
                self.assertEqual(main(argv, local_tz=UTC), 0)

        report = self._payload(output.getvalue())
        self.assertEqual(report["quota_status"], "unverified")
        self.assertTrue(report["reset_projected"])
        self.assertEqual(report["seconds_until_reset"], 604799)
        self.assertEqual(report["cycle_start_local"], anchor.isoformat())
        self.assertEqual(report["llm_calls"], 1)
        self.assertEqual(report["priced_calls"], 1)
        self.assertEqual(report["estimated_used_usd"], 0.2)
        self.assertEqual(self.state.read_bytes(), before_state)

    def test_reset_crossing_during_state_sync_uses_the_new_cycle(self) -> None:
        anchor = datetime(2026, 11, 27, 12, tzinfo=UTC)
        self._write_calls(
            [
                (anchor - timedelta(seconds=1), "Gemini 3.1 Pro", 200_000, 0),
                (anchor, "Gemini 3.1 Pro", 100_000, 0),
            ]
        )
        clock = anchor - timedelta(seconds=1)
        fsync = os.fsync

        class SimulatedDateTime(datetime):
            @classmethod
            def now(cls, tz: tzinfo | None = None) -> datetime:
                return clock.astimezone(tz) if tz is not None else clock

        def sync_across_reset(fd: int) -> None:
            nonlocal clock
            fsync(fd)
            clock = anchor + timedelta(seconds=1)
            path = self.brain / "session-1" / ".system_generated/logs/transcript.jsonl"
            created_at = clock.isoformat().replace("+00:00", "Z")
            with path.open("a", encoding="utf-8") as transcript:
                transcript.write(
                    json.dumps(
                        {
                            "step_index": 2,
                            "source": "USER",
                            "type": "USER_INPUT",
                            "status": "DONE",
                            "created_at": created_at,
                            "content": (
                                "<USER_SETTINGS_CHANGE>The user changed setting Model Selection "
                                "from Previous to Gemini 3.1 Pro.</USER_SETTINGS_CHANGE>\n"
                                "<USER_REQUEST>Written during state sync</USER_REQUEST>"
                            ),
                        }
                    )
                    + "\n"
                )
                transcript.write(
                    json.dumps(
                        {
                            "step_index": 3,
                            "source": "MODEL",
                            "type": "PLANNER_RESPONSE",
                            "status": "DONE",
                            "created_at": created_at,
                            "input_tokens": 100_000,
                            "output_tokens": 0,
                        }
                    )
                    + "\n"
                )

        with mock.patch("agy_token_stats.cli.datetime", SimulatedDateTime):
            with mock.patch("agy_token_stats.quota.os.fsync", sync_across_reset):
                report = self._payload(
                    self._invoke(None, "--quota-blocked", "--quota-reset", "1s")
                )

        self.assertEqual(report["quota_status"], "unverified")
        self.assertEqual(report["cycle_start_local"], anchor.isoformat())
        self.assertEqual(report["seconds_until_reset"], 604799)
        self.assertEqual(report["llm_calls"], 2)
        self.assertEqual(report["priced_calls"], 2)
        self.assertEqual(report["estimated_used_usd"], 0.4)


if __name__ == "__main__":
    unittest.main()
