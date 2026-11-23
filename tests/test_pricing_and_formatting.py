"""Behavior tests for Gemini token pricing, cost aggregation, and color formatting."""

from __future__ import annotations

import contextlib
from datetime import datetime, timezone
import io
import json
import os
import re
import unittest
from unittest import mock

from tests.helpers import (
    IST,
    TempBrainTestCase,
    write_jsonl,
    write_subagent_descriptor,
)

import agy_token_stats as stats

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


class PricingAndFormattingTests(TempBrainTestCase):
    """Tests for Gemini rate resolution, tree cost rollups, and ANSI color output."""

    def test_resolve_model_pricing_covers_gemini_4_3_25_and_unpriced_models(
        self,
    ) -> None:
        nov_ts = datetime(2026, 11, 23, 12, 0, 0, tzinfo=timezone.utc)
        dec_ts = datetime(2026, 12, 5, 12, 0, 0, tzinfo=timezone.utc)
        jan27_ts = datetime(2027, 1, 5, 12, 0, 0, tzinfo=timezone.utc)

        # Gemini 4 Hydrogen & Neon
        h = stats.resolve_model_pricing("Gemini 4 Hydrogen (High)", at_utc=nov_ts)
        self.assertIsNotNone(h)
        assert h is not None
        self.assertEqual((h.input_per_1m, h.cache_read_per_1m, h.output_per_1m), (1.50, 0.075, 7.50))

        n = stats.resolve_model_pricing("Gemini 4 Neon (Medium)", at_utc=nov_ts)
        self.assertIsNotNone(n)
        assert n is not None
        self.assertEqual((n.input_per_1m, n.cache_read_per_1m, n.output_per_1m), (3.00, 0.15, 15.00))

        # Gemini 4 Argon & Carbon intro (Nov 2026) vs standard (Dec 2026+)
        for name in ("Gemini 4 Argon (High)", "Gemini 4 Carbon (Medium)"):
            intro = stats.resolve_model_pricing(name, at_utc=nov_ts)
            std = stats.resolve_model_pricing(name, at_utc=dec_ts)
            assert intro is not None and std is not None
            self.assertEqual((intro.input_per_1m, intro.cache_read_per_1m, intro.output_per_1m), (2.00, 0.10, 10.00))
            self.assertEqual((std.input_per_1m, std.cache_read_per_1m, std.output_per_1m), (4.00, 0.20, 20.00))

        # Gemini 3.8 / 3.7 Flash intro (2026) vs standard (2027+)
        f38_intro = stats.resolve_model_pricing("Gemini 3.8 Flash (High)", at_utc=nov_ts)
        f38_std = stats.resolve_model_pricing("Gemini 3.8 Flash (High)", at_utc=jan27_ts)
        assert f38_intro is not None and f38_std is not None
        self.assertEqual((f38_intro.input_per_1m, f38_intro.cache_read_per_1m, f38_intro.output_per_1m), (0.75, 0.075, 3.75))
        self.assertEqual((f38_std.input_per_1m, f38_std.cache_read_per_1m, f38_std.output_per_1m), (1.50, 0.15, 7.50))

        # Gemini 3.1 Pro & API ID normalization
        p31 = stats.resolve_model_pricing("models/gemini-3.1-pro-preview", at_utc=nov_ts)
        assert p31 is not None
        self.assertEqual((p31.input_per_1m, p31.cache_read_per_1m, p31.output_per_1m), (2.00, 0.20, 12.00))
        self.assertEqual((p31.long_context_threshold, p31.long_input_per_1m, p31.long_output_per_1m), (200_000, 4.00, 18.00))

        # Gemini 3.1 Flash-Lite & 2.5 Pro
        lite = stats.resolve_model_pricing("gemini-3.1-flash-lite", at_utc=nov_ts)
        assert lite is not None
        self.assertEqual((lite.input_per_1m, lite.cache_read_per_1m, lite.output_per_1m), (0.25, 0.025, 1.50))

        p25 = stats.resolve_model_pricing("Gemini 2.5 Pro", at_utc=nov_ts)
        assert p25 is not None
        self.assertEqual((p25.input_per_1m, p25.cache_read_per_1m, p25.output_per_1m), (1.25, 0.125, 10.00))

        # Non-Gemini / Unknown return None
        for unpriced in ("Unknown", "", "Claude Opus 4.6 (Thinking)", "GPT-5"):
            self.assertIsNone(stats.resolve_model_pricing(unpriced, at_utc=nov_ts))

    def test_pro_long_context_tier_and_mixed_tree_cost_rollup(self) -> None:
        ide_brain = self.tmp_path / "antigravity/brain"
        root_id = "11111111-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
        sub_id = "22222222-bbbb-bbbb-bbbb-bbbbbbbbbbbb"

        # Root uses Gemini 3.1 Pro (High) with >200k prompt (150k uncached + 100k cache = 250k gross)
        # Long-context rates: $4.00/1M in, $0.40/1M cache, $18.00/1M out
        # In cost: 0.15 * 4.00 = $0.60; Cache cost: 0.10 * 0.40 = $0.04; Out cost: 0.01 * 18.00 = $0.18 => $0.82
        # Cache savings: 0.10 * (4.00 - 0.40) = $0.36
        # Plus a second step switched to Claude (unpriced) to verify partial coverage marker `*`.
        write_jsonl(
            ide_brain / root_id / ".system_generated/logs/transcript.jsonl",
            [
                {
                    "step_index": 0,
                    "source": "USER",
                    "type": "USER_INPUT",
                    "status": "DONE",
                    "created_at": "2026-11-23T09:00:00+05:30",
                    "content": (
                        "<USER_SETTINGS_CHANGE>The user changed setting Model Selection "
                        "from Prev to Gemini 3.1 Pro (High).</USER_SETTINGS_CHANGE>\n"
                        "<USER_REQUEST>Analyze large repo</USER_REQUEST>"
                    ),
                },
                {
                    "step_index": 1,
                    "source": "MODEL",
                    "type": "PLANNER_RESPONSE",
                    "status": "DONE",
                    "created_at": "2026-11-23T09:01:00+05:30",
                    "input_tokens": 150_000,
                    "cache_read_tokens": 100_000,
                    "output_tokens": 10_000,
                },
                {
                    "step_index": 2,
                    "source": "USER",
                    "type": "USER_INPUT",
                    "status": "DONE",
                    "created_at": "2026-11-23T09:02:00+05:30",
                    "content": (
                        "<USER_SETTINGS_CHANGE>The user changed setting Model Selection "
                        "from Gemini 3.1 Pro (High) to Claude Opus 4.6 (Thinking).</USER_SETTINGS_CHANGE>\n"
                        "<USER_REQUEST>Second step</USER_REQUEST>"
                    ),
                },
                {
                    "step_index": 3,
                    "source": "MODEL",
                    "type": "PLANNER_RESPONSE",
                    "status": "DONE",
                    "created_at": "2026-11-23T09:03:00+05:30",
                    "input_tokens": 10_000,
                    "cache_read_tokens": 0,
                    "output_tokens": 1_000,
                },
            ],
        )
        write_subagent_descriptor(
            ide_brain / root_id,
            child_id=sub_id,
            type_name="oracle",
            role="Search symbols",
            spawn_step_index=1,
        )
        # Child uses Gemini 4 Hydrogen (High) override:
        # 510k uncached @ $1.50 = $0.765; 1M cache @ $0.075 = $0.075; 20k out @ $7.50 = $0.15 => $0.99
        # Cache savings: 1.0 * (1.50 - 0.075) = $1.425
        write_jsonl(
            ide_brain / sub_id / ".system_generated/logs/transcript.jsonl",
            [
                {
                    "step_index": 0,
                    "source": "USER",
                    "type": "USER_INPUT",
                    "status": "DONE",
                    "created_at": "2026-11-23T09:01:05+05:30",
                    "content": (
                        "<USER_SETTINGS_CHANGE>The user changed setting Model Selection "
                        "from Prev to Gemini 4 Hydrogen (High).</USER_SETTINGS_CHANGE>\n"
                        "<USER_REQUEST>Explore</USER_REQUEST>"
                    ),
                },
                {
                    "step_index": 1,
                    "source": "MODEL",
                    "type": "PLANNER_RESPONSE",
                    "status": "DONE",
                    "created_at": "2026-11-23T09:01:30+05:30",
                    "input_tokens": 510_000,
                    "cache_read_tokens": 1_000_000,
                    "output_tokens": 20_000,
                },
            ],
        )

        sessions = stats.discover_sessions(
            [("ide", ide_brain)], history_path=None, local_tz=IST
        )
        report = stats.compute_stats_report(sessions, sources_scanned=["ide"])

        # Overall cost: $0.82 (Pro >200k) + $0.99 (Hydrogen) = $1.81
        self.assertAlmostEqual(report.overall.cost_usd, 1.81, places=6)
        self.assertAlmostEqual(report.overall.cache_savings_usd, 0.36 + 1.425, places=6)
        self.assertEqual(report.overall.priced_calls, 2)
        self.assertEqual(report.overall.unpriced_calls, 1)

        # Model breakdown preserves exact per-model cost and None for unpriced Claude
        self.assertAlmostEqual(
            report.by_model["Gemini 3.1 Pro (High)"].cost_usd, 0.82, places=6
        )
        self.assertAlmostEqual(
            report.by_model["Gemini 4 Hydrogen (High)"].cost_usd, 0.99, places=6
        )
        self.assertEqual(report.by_model["Claude Opus 4.6 (Thinking)"].priced_calls, 0)
        self.assertIsNone(
            report.by_model["Claude Opus 4.6 (Thinking)"].to_dict()["approx_cost_usd"]
        )

        # Tree summary rolls child cost into root without double counting
        root_sum = report.sessions[0]
        self.assertAlmostEqual(root_sum.self_metrics.cost_usd, 0.82, places=6)
        self.assertAlmostEqual(root_sum.tree_metrics.cost_usd, 1.81, places=6)

        # Human report marks partial cost with `*` and shows pricing footer
        plain = stats.format_human_report(report, show_subagents=True, color=False)
        self.assertNotIn("\x1b[", plain)
        self.assertIn("$1.81*", plain)
        self.assertIn("aistudio.google.com/docs/choosing-a-model", plain)

        # JSON output includes approx_cost_usd
        payload = json.loads(stats.format_json_report(report))
        self.assertAlmostEqual(payload["overall"]["approx_cost_usd"], 1.81, places=6)
        self.assertEqual(payload["overall"]["priced_calls"], 2)
        self.assertEqual(payload["overall"]["unpriced_calls"], 1)

    def test_color_resolution_ansi_alignment_and_cli_flags(self) -> None:
        # Environment & flag resolution
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertTrue(stats.resolve_color_enabled("always"))
            self.assertFalse(stats.resolve_color_enabled("never"))
            self.assertFalse(stats.resolve_color_enabled("auto", stream=io.StringIO()))

        with mock.patch.dict(os.environ, {"FORCE_COLOR": "1"}, clear=True):
            self.assertTrue(stats.resolve_color_enabled("auto", stream=io.StringIO()))

        with mock.patch.dict(os.environ, {"NO_COLOR": "1", "FORCE_COLOR": "1"}, clear=True):
            self.assertFalse(stats.resolve_color_enabled("auto", stream=io.StringIO()))
            self.assertTrue(stats.resolve_color_enabled("always", stream=io.StringIO()))

        with mock.patch.dict(os.environ, {"TERM": "dumb", "FORCE_COLOR": "1"}, clear=True):
            self.assertFalse(stats.resolve_color_enabled("auto", stream=io.StringIO()))

        # Build a small Gemini session and verify ANSI color output aligns tables properly
        cli_brain = self.tmp_path / "antigravity-cli/brain"
        sid = "abcdef01-2345-6789-abcd-ef0123456789"
        write_jsonl(
            cli_brain / sid / ".system_generated/logs/transcript.jsonl",
            [
                {
                    "step_index": 0,
                    "source": "USER",
                    "type": "USER_INPUT",
                    "status": "DONE",
                    "created_at": "2026-11-23T10:00:00+05:30",
                    "content": (
                        "<USER_SETTINGS_CHANGE>The user changed setting Model Selection "
                        "from Prev to Gemini 4 Carbon (High).</USER_SETTINGS_CHANGE>\n"
                        "<USER_REQUEST>Colorize CLI</USER_REQUEST>"
                    ),
                },
                {
                    "step_index": 1,
                    "source": "MODEL",
                    "type": "PLANNER_RESPONSE",
                    "status": "DONE",
                    "created_at": "2026-11-23T10:00:10+05:30",
                    "thinking": "t" * 100,
                    "content": "c" * 100,
                    "input_tokens": 100_000,
                    "cache_read_tokens": 900_000,
                    "output_tokens": 10_000,
                },
            ],
        )

        now_ist = datetime(2026, 11, 23, 12, 0, 0, tzinfo=IST)

        buf_color = io.StringIO()
        with contextlib.redirect_stdout(buf_color):
            rc = stats.main(
                ["--brain-dir", f"cli={cli_brain}", "--color", "always"],
                now=now_ist,
                local_tz=IST,
            )
        self.assertEqual(rc, 0)
        colored_out = buf_color.getvalue()
        self.assertIn("\x1b[", colored_out)

        buf_plain = io.StringIO()
        with contextlib.redirect_stdout(buf_plain):
            rc_plain = stats.main(
                ["--brain-dir", f"cli={cli_brain}", "--no-color"],
                now=now_ist,
                local_tz=IST,
            )
        self.assertEqual(rc_plain, 0)
        plain_out = buf_plain.getvalue()
        self.assertNotIn("\x1b[", plain_out)

        # Stripping ANSI codes from colored output must match plain output character-for-character
        self.assertEqual(_ANSI_RE.sub("", colored_out), plain_out)


if __name__ == "__main__":
    unittest.main()
