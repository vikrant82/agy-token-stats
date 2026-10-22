"""Behavior tests for timestamp parsing and time window filtering (`agy_token_stats.time_window`)."""

from __future__ import annotations

from datetime import datetime
import unittest

from tests.helpers import IST, TempBrainTestCase, write_jsonl

import agy_token_stats as stats


class TimeWindowTests(TempBrainTestCase):
    """Behavior tests for ISO timestamp parsing, window presets, and step vs session filtering."""

    def test_time_window_presets_and_step_vs_session_filtering(self) -> None:
        cli_brain = self.tmp_path / "antigravity-cli/brain"
        sess_spanning = "dddddddd-dddd-dddd-dddd-dddddddddddd"
        sess_old = "eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee"

        # Fixed reference `now`: 2026-04-19 15:00:00 IST (09:30:00 UTC)
        now_ist = datetime(2026, 4, 19, 15, 0, 0, tzinfo=IST)

        # sess_spanning starts yesterday in IST (2026-04-18 23:00:00+05:30)
        # and has a second step today in IST (2026-04-19 10:00:00+05:30).
        write_jsonl(
            cli_brain
            / sess_spanning
            / ".system_generated/logs/transcript.jsonl",
            [
                {
                    "step_index": 0,
                    "source": "MODEL",
                    "type": "PLANNER_RESPONSE",
                    "status": "DONE",
                    "created_at": "2026-04-18T23:00:00+05:30",
                    "input_tokens": 1000,
                    "cache_read_tokens": 0,
                    "output_tokens": 100,
                },
                {
                    "step_index": 1,
                    "source": "MODEL",
                    "type": "PLANNER_RESPONSE",
                    "status": "DONE",
                    "created_at": "2026-04-19T10:00:00+05:30",
                    "input_tokens": 2000,
                    "cache_read_tokens": 500,
                    "output_tokens": 200,
                },
            ],
        )

        # sess_old happened 5 days ago (2026-04-14 12:00:00+05:30)
        write_jsonl(
            cli_brain / sess_old / ".system_generated/logs/transcript.jsonl",
            [
                {
                    "step_index": 0,
                    "source": "MODEL",
                    "type": "PLANNER_RESPONSE",
                    "status": "DONE",
                    "created_at": "2026-04-14T12:00:00+05:30",
                    "input_tokens": 4000,
                    "cache_read_tokens": 0,
                    "output_tokens": 400,
                },
            ],
        )

        sessions = stats.discover_sessions(
            [("cli", cli_brain)], history_path=None, local_tz=IST
        )

        # 1. --today (step-timestamp mode): includes only step 1 of sess_spanning (2700 total)
        win_today = stats.resolve_time_window(
            now=now_ist, today=True, local_tz=IST
        )
        rep_today = stats.compute_stats_report(sessions, window=win_today)
        self.assertEqual(rep_today.overall.total_tokens, 2700)

        # 2. --last-n-days 1 matches --today
        win_1d = stats.resolve_time_window(
            now=now_ist, last_n_days=1, local_tz=IST
        )
        self.assertEqual(win_1d.start_local, win_today.start_local)
        self.assertEqual(win_1d.end_local, win_today.end_local)

        # 3. --today with --by-session: sess_spanning started yesterday, so 0 tokens today
        rep_today_by_sess = stats.compute_stats_report(
            sessions, window=win_today, by_session=True
        )
        self.assertEqual(rep_today_by_sess.overall.total_tokens, 0)

        # 4. --yesterday with --by-session: includes BOTH steps of sess_spanning (1100 + 2700 = 3800)
        win_yesterday = stats.resolve_time_window(
            now=now_ist, yesterday=True, local_tz=IST
        )
        rep_yest_by_sess = stats.compute_stats_report(
            sessions, window=win_yesterday, by_session=True
        )
        self.assertEqual(rep_yest_by_sess.overall.total_tokens, 3800)

        # 5. --last-n-days 7 (starts 2026-04-13 00:00:00 IST): includes sess_spanning + sess_old (3800 + 4400 = 8200)
        win_7d = stats.resolve_time_window(
            now=now_ist, last_n_days=7, local_tz=IST
        )
        self.assertEqual(
            win_7d.start_local, datetime(2026, 4, 13, 0, 0, 0, tzinfo=IST)
        )
        rep_7d = stats.compute_stats_report(sessions, window=win_7d)
        self.assertEqual(rep_7d.overall.total_tokens, 8200)
        self.assertEqual(
            [d.date_str for d in rep_7d.daily],
            ["2026-04-14", "2026-04-18", "2026-04-19"],
        )

        # 6. --last-n-hours 6 (09:00..15:00 IST today): includes only step 1 at 10:00 IST (2700)
        win_6h = stats.resolve_time_window(
            now=now_ist, last_n_hours=6, local_tz=IST
        )
        rep_6h = stats.compute_stats_report(sessions, window=win_6h)
        self.assertEqual(rep_6h.overall.total_tokens, 2700)

    def test_resolve_time_window_rejects_invalid_arguments(self) -> None:
        now_ist = datetime(2026, 4, 19, 12, 0, 0, tzinfo=IST)
        invalid_cases = [
            {"last_n_days": 0},
            {"last_n_hours": -1},
            {"today": True, "yesterday": True},
            {"since": "2026-04-20", "until": "2026-04-19"},
        ]
        for kwargs in invalid_cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    stats.resolve_time_window(
                        now=now_ist, local_tz=IST, **kwargs
                    )

    def test_iso_timestamps_with_variable_fractional_seconds_and_non_finite_hours(
        self,
    ) -> None:
        samples = [
            ("2026-04-19T06:00:00Z", 0),
            ("2026-04-19T06:00:00.1Z", 100000),
            ("2026-04-19T06:00:00.12345Z", 123450),
            ("2026-04-19T06:00:00.123456789+05:30", 123456),
        ]
        for raw, expected_microsecond in samples:
            with self.subTest(raw=raw):
                parsed = stats.parse_iso_timestamp(raw, local_tz=IST)
                self.assertIsNotNone(parsed)
                assert parsed is not None
                dt_utc, dt_local = parsed
                self.assertEqual(dt_utc.microsecond, expected_microsecond)
                self.assertEqual(dt_local.microsecond, expected_microsecond)

        now_ist = datetime(2026, 4, 19, 12, 0, 0, tzinfo=IST)
        for bad_hours in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(bad_hours=bad_hours):
                with self.assertRaises(ValueError):
                    stats.resolve_time_window(
                        now=now_ist, last_n_hours=bad_hours, local_tz=IST
                    )


if __name__ == "__main__":
    unittest.main()
