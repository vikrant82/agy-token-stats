"""Behavior tests for subagent tree rollups, role breakdowns, and CLI output (`aggregator`, `formatters`, `cli`)."""

from __future__ import annotations

import contextlib
from datetime import datetime
import io
import json
import unittest

from tests.helpers import (
    IST,
    TempBrainTestCase,
    write_jsonl,
    write_subagent_descriptor,
)

import agy_token_stats as stats


class AggregatorAndCliTests(TempBrainTestCase):
    """Behavior tests for tree/flat rollups, multi-level subagents, orphan subagents, and CLI output."""

    def test_subagent_tree_rollup_and_role_breakdown_without_double_counting(
        self,
    ) -> None:
        ide_brain = self.tmp_path / "antigravity/brain"
        root_id = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
        oracle_id = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
        fallback_sub_id = "cccccccc-cccc-cccc-cccc-cccccccccccc"

        # Root session: 1 LLM call (1000 uncached + 3000 cache + 200 out = 4200 total)
        # Also includes a later SYSTEM_MESSAGE at step 2 where child oracle replies to root
        # with sender=oracle_id; this must NOT invert parent-child linking or create a cycle.
        write_jsonl(
            ide_brain / root_id / ".system_generated/logs/transcript.jsonl",
            [
                {
                    "step_index": 0,
                    "source": "USER",
                    "type": "USER_INPUT",
                    "status": "DONE",
                    "created_at": "2026-04-19T06:00:00Z",
                    "content": "<USER_REQUEST>Audit architecture</USER_REQUEST>",
                },
                {
                    "step_index": 1,
                    "source": "MODEL",
                    "type": "PLANNER_RESPONSE",
                    "status": "DONE",
                    "created_at": "2026-04-19T06:00:10Z",
                    "input_tokens": 1000,
                    "cache_read_tokens": 3000,
                    "output_tokens": 200,
                    "tool_calls": [{"id": "1", "name": "invoke_subagent"}],
                },
                {
                    "step_index": 2,
                    "source": "SYSTEM",
                    "type": "SYSTEM_MESSAGE",
                    "status": "DONE",
                    "created_at": "2026-04-19T06:02:00Z",
                    "content": f"<SYSTEM_MESSAGE>sender={oracle_id} Report complete</SYSTEM_MESSAGE>",
                },
            ],
        )
        write_subagent_descriptor(
            ide_brain / root_id,
            child_id=oracle_id,
            type_name="oracle",
            role="Architecture exploration",
            spawn_step_index=1,
        )

        # Child 1 (explicit descriptor): oracle (500 uncached + 1500 cache + 100 out = 2100 total)
        write_jsonl(
            ide_brain / oracle_id / ".system_generated/logs/transcript.jsonl",
            [
                {
                    "step_index": 0,
                    "source": "SYSTEM",
                    "type": "SYSTEM_MESSAGE",
                    "status": "DONE",
                    "created_at": "2026-04-19T06:00:15Z",
                    "content": f"<SYSTEM_MESSAGE>sender={root_id} **Goal**: Explore codebase</SYSTEM_MESSAGE>",
                },
                {
                    "step_index": 1,
                    "source": "MODEL",
                    "type": "PLANNER_RESPONSE",
                    "status": "DONE",
                    "created_at": "2026-04-19T06:01:00Z",
                    "input_tokens": 500,
                    "cache_read_tokens": 1500,
                    "output_tokens": 100,
                },
            ],
        )

        # Child 2 (no descriptor file; linked via step-0 SYSTEM_MESSAGE sender=root_id):
        # (300 uncached + 700 cache + 50 out = 1050 total)
        write_jsonl(
            ide_brain
            / fallback_sub_id
            / ".system_generated/logs/transcript.jsonl",
            [
                {
                    "step_index": 0,
                    "source": "SYSTEM",
                    "type": "SYSTEM_MESSAGE",
                    "status": "DONE",
                    "created_at": "2026-04-19T06:00:20Z",
                    "content": f"<SYSTEM_MESSAGE>sender={root_id} **Goal**: Check tests</SYSTEM_MESSAGE>",
                },
                {
                    "step_index": 1,
                    "source": "MODEL",
                    "type": "PLANNER_RESPONSE",
                    "status": "DONE",
                    "created_at": "2026-04-19T06:01:30Z",
                    "input_tokens": 300,
                    "cache_read_tokens": 700,
                    "output_tokens": 50,
                },
            ],
        )

        sessions = stats.discover_sessions(
            [("ide", ide_brain)], history_path=None, local_tz=IST
        )
        self.assertIsNone(sessions[root_id].parent_id)
        self.assertEqual(sessions[oracle_id].parent_id, root_id)
        self.assertEqual(sessions[oracle_id].agent_type, "oracle")
        self.assertEqual(sessions[fallback_sub_id].parent_id, root_id)
        self.assertEqual(sessions[fallback_sub_id].agent_type, "subagent")

        # Tree mode report (default)
        tree_report = stats.compute_stats_report(
            sessions, flat=False, sources_scanned=["ide"]
        )
        self.assertEqual(tree_report.active_root_sessions, 1)
        self.assertEqual(tree_report.active_subagent_sessions, 2)
        self.assertEqual(tree_report.overall.total_tokens, 4200 + 2100 + 1050)
        self.assertEqual(
            tree_report.overall.net_tokens,
            (1000 + 200) + (500 + 100) + (300 + 50),
        )

        # Role breakdown separates root, oracle, and fallback subagent
        self.assertEqual(tree_report.by_role["root"].total_tokens, 4200)
        self.assertEqual(tree_report.by_role["oracle"].total_tokens, 2100)
        self.assertEqual(tree_report.by_role["subagent"].total_tokens, 1050)

        # Top-level sessions list has only the 1 root session, with tree_metrics rolled up
        self.assertEqual(len(tree_report.sessions), 1)
        root_sum = tree_report.sessions[0]
        self.assertEqual(root_sum.conversation_id, root_id)
        self.assertEqual(root_sum.active_subagent_count, 2)
        self.assertEqual(root_sum.self_metrics.total_tokens, 4200)
        self.assertEqual(root_sum.tree_metrics.total_tokens, 7350)
        self.assertEqual(
            [ch.conversation_id for ch in root_sum.children],
            [oracle_id, fallback_sub_id],
        )

        # Flat mode report treats all 3 conversations as independent entries
        flat_report = stats.compute_stats_report(
            sessions, flat=True, sources_scanned=["ide"]
        )
        self.assertEqual(len(flat_report.sessions), 3)
        self.assertEqual(flat_report.overall.total_tokens, 7350)
        for s in flat_report.sessions:
            self.assertEqual(
                s.tree_metrics.total_tokens, s.self_metrics.total_tokens
            )
            self.assertEqual(s.children, [])

    def test_multilevel_subagent_tree_counts_when_parent_and_intermediate_subagent_inactive_in_window(
        self,
    ) -> None:
        ide_brain = self.tmp_path / "antigravity/brain"
        root_id = "11111111-2222-3333-4444-555555555500"
        sub_a_id = "11111111-2222-3333-4444-55555555550a"
        sub_b_id = "11111111-2222-3333-4444-55555555550b"

        # Root and SubA were only active yesterday (2026-04-18 IST)
        write_jsonl(
            ide_brain / root_id / ".system_generated/logs/transcript.jsonl",
            [
                {
                    "step_index": 0,
                    "source": "MODEL",
                    "type": "PLANNER_RESPONSE",
                    "status": "DONE",
                    "created_at": "2026-04-18T18:00:00+05:30",
                    "input_tokens": 1000,
                    "cache_read_tokens": 0,
                    "output_tokens": 100,
                }
            ],
        )
        write_subagent_descriptor(
            ide_brain / root_id,
            child_id=sub_a_id,
            type_name="implementer",
            role="Intermediate worker",
        )

        write_jsonl(
            ide_brain / sub_a_id / ".system_generated/logs/transcript.jsonl",
            [
                {
                    "step_index": 0,
                    "source": "MODEL",
                    "type": "PLANNER_RESPONSE",
                    "status": "DONE",
                    "created_at": "2026-04-18T18:05:00+05:30",
                    "input_tokens": 800,
                    "cache_read_tokens": 0,
                    "output_tokens": 80,
                }
            ],
        )
        write_subagent_descriptor(
            ide_brain / sub_a_id,
            child_id=sub_b_id,
            type_name="oracle",
            role="Leaf worker active today",
        )

        # SubB is active today (2026-04-19 IST)
        write_jsonl(
            ide_brain / sub_b_id / ".system_generated/logs/transcript.jsonl",
            [
                {
                    "step_index": 0,
                    "source": "MODEL",
                    "type": "PLANNER_RESPONSE",
                    "status": "DONE",
                    "created_at": "2026-04-19T09:30:00+05:30",
                    "input_tokens": 500,
                    "cache_read_tokens": 1500,
                    "output_tokens": 250,
                }
            ],
        )

        sessions = stats.discover_sessions(
            [("ide", ide_brain)], history_path=None, local_tz=IST
        )
        now_ist = datetime(2026, 4, 19, 12, 0, 0, tzinfo=IST)
        win_today = stats.resolve_time_window(
            now=now_ist, today=True, local_tz=IST
        )

        rep_today = stats.compute_stats_report(
            sessions, window=win_today, flat=False, sources_scanned=["ide"]
        )
        self.assertEqual(rep_today.active_root_sessions, 1)
        self.assertEqual(rep_today.active_subagent_sessions, 1)
        self.assertEqual(len(rep_today.sessions), 1)

        root_sum = rep_today.sessions[0]
        self.assertEqual(root_sum.conversation_id, root_id)
        self.assertEqual(root_sum.active_subagent_count, 1)
        self.assertEqual(root_sum.total_subagent_count, 2)
        self.assertEqual(root_sum.self_metrics.total_tokens, 0)
        self.assertEqual(root_sum.tree_metrics.total_tokens, 2250)

        # Filtering directly on SubA across all time: active_root_sessions is 0,
        # and human report labels SubA's self row with "implementer (self)"
        rep_sub_a = stats.compute_stats_report(
            sessions,
            session_filter=sub_a_id,
            flat=False,
            sources_scanned=["ide"],
        )
        self.assertEqual(rep_sub_a.active_root_sessions, 0)
        self.assertEqual(rep_sub_a.active_subagent_sessions, 2)
        rendered = stats.format_human_report(rep_sub_a, show_subagents=True)
        self.assertIn("implementer (self)", rendered)

    def test_orphan_subagent_is_not_counted_as_root_session(self) -> None:
        ide_brain = self.tmp_path / "antigravity/brain"
        missing_parent_id = "99999999-9999-9999-9999-999999999999"
        orphan_sub_id = "88888888-8888-8888-8888-888888888888"

        # First step has step_index=5 (non-zero) to also verify fallback parent linking on first step
        write_jsonl(
            ide_brain
            / orphan_sub_id
            / ".system_generated/logs/transcript.jsonl",
            [
                {
                    "step_index": 5,
                    "source": "SYSTEM",
                    "type": "SYSTEM_MESSAGE",
                    "status": "DONE",
                    "created_at": "2026-04-19T10:00:00+05:30",
                    "content": f"<SYSTEM_MESSAGE>sender={missing_parent_id} **Goal**: Run check</SYSTEM_MESSAGE>",
                },
                {
                    "step_index": 6,
                    "source": "MODEL",
                    "type": "PLANNER_RESPONSE",
                    "status": "DONE",
                    "created_at": "2026-04-19T10:01:00+05:30",
                    "input_tokens": 400,
                    "cache_read_tokens": 600,
                    "output_tokens": 100,
                },
            ],
        )

        sessions = stats.discover_sessions(
            [("ide", ide_brain)], history_path=None, local_tz=IST
        )
        self.assertEqual(sessions[orphan_sub_id].parent_id, missing_parent_id)
        self.assertEqual(sessions[orphan_sub_id].agent_type, "subagent")

        report = stats.compute_stats_report(
            sessions, flat=False, sources_scanned=["ide"]
        )
        self.assertEqual(report.active_root_sessions, 0)
        self.assertEqual(report.active_subagent_sessions, 1)
        self.assertEqual(len(report.daily), 1)
        self.assertEqual(report.daily[0].active_root_sessions, 0)
        self.assertEqual(report.daily[0].active_subagent_sessions, 1)

    def test_cli_main_json_and_human_session_tree_output(self) -> None:
        ide_brain = self.tmp_path / "antigravity/brain"
        root_id = "12345678-1234-1234-1234-123456789abc"
        child_id = "87654321-4321-4321-4321-cba987654321"

        write_jsonl(
            ide_brain / root_id / ".system_generated/logs/transcript.jsonl",
            [
                {
                    "step_index": 0,
                    "source": "USER",
                    "type": "USER_INPUT",
                    "status": "DONE",
                    "created_at": "2026-04-19T09:00:00+05:30",
                    "content": "<USER_REQUEST>Build token report</USER_REQUEST>",
                },
                {
                    "step_index": 1,
                    "source": "MODEL",
                    "type": "PLANNER_RESPONSE",
                    "status": "DONE",
                    "created_at": "2026-04-19T09:01:00+05:30",
                    "input_tokens": 1000,
                    "cache_read_tokens": 2000,
                    "output_tokens": 300,
                },
            ],
        )
        write_subagent_descriptor(
            ide_brain / root_id,
            child_id=child_id,
            type_name="implementer",
            role="Write unit tests",
        )
        write_jsonl(
            ide_brain / child_id / ".system_generated/logs/transcript.jsonl",
            [
                {
                    "step_index": 0,
                    "source": "MODEL",
                    "type": "PLANNER_RESPONSE",
                    "status": "DONE",
                    "created_at": "2026-04-19T09:02:00+05:30",
                    "input_tokens": 500,
                    "cache_read_tokens": 500,
                    "output_tokens": 100,
                },
            ],
        )

        now_ist = datetime(2026, 4, 19, 12, 0, 0, tzinfo=IST)

        # 1. JSON output mode
        buf_json = io.StringIO()
        with contextlib.redirect_stdout(buf_json):
            rc_json = stats.main(
                [
                    "--brain-dir",
                    f"ide={ide_brain}",
                    "--today",
                    "--json",
                ],
                now=now_ist,
                local_tz=IST,
            )
        self.assertEqual(rc_json, 0)
        payload = json.loads(buf_json.getvalue())
        self.assertEqual(payload["window"]["label"], "today")
        self.assertEqual(payload["overall"]["total_tokens"], 4400)
        self.assertEqual(payload["active_root_sessions"], 1)
        self.assertEqual(payload["active_subagent_sessions"], 1)
        self.assertEqual(payload["by_role"]["root"]["total_tokens"], 3300)
        self.assertEqual(
            payload["by_role"]["implementer"]["total_tokens"], 1100
        )

        # 2. Human-readable output with --session prefix filter (auto-enables subagent tree rows)
        buf_human = io.StringIO()
        with contextlib.redirect_stdout(buf_human):
            rc_human = stats.main(
                [
                    "--brain-dir",
                    f"ide={ide_brain}",
                    "--session",
                    "12345678",
                ],
                now=now_ist,
                local_tz=IST,
            )
        self.assertEqual(rc_human, 0)
        human_out = buf_human.getvalue()
        self.assertIn("Antigravity (agy) Token Usage Report", human_out)
        self.assertIn("12345678", human_out)
        self.assertIn("└─ 87654321", human_out)
        self.assertIn("implementer", human_out)

    def test_mid_session_model_switch_splits_by_model_and_cli_model_filter(
        self,
    ) -> None:
        ide_brain = self.tmp_path / "antigravity/brain"
        session_id = "abcdef12-3456-7890-abcd-ef1234567890"

        write_jsonl(
            ide_brain / session_id / ".system_generated/logs/transcript.jsonl",
            [
                {
                    "step_index": 0,
                    "source": "USER",
                    "type": "USER_INPUT",
                    "status": "DONE",
                    "created_at": "2026-04-19T10:00:00+05:30",
                    "content": (
                        "<USER_SETTINGS_CHANGE>The user has changed setting `Model Selection` "
                        "from Prev to Claude Opus 4.6 (Thinking).</USER_SETTINGS_CHANGE>\n"
                        "<USER_REQUEST>Phase 1</USER_REQUEST>"
                    ),
                },
                {
                    "step_index": 1,
                    "source": "MODEL",
                    "type": "PLANNER_RESPONSE",
                    "status": "DONE",
                    "created_at": "2026-04-19T10:00:10+05:30",
                    "thinking": "t" * 50,
                    "content": "c" * 50,
                    "input_tokens": 1000,
                    "cache_read_tokens": 2000,
                    "output_tokens": 200,
                },
                {
                    "step_index": 2,
                    "source": "USER",
                    "type": "USER_INPUT",
                    "status": "DONE",
                    "created_at": "2026-04-19T10:01:00+05:30",
                    "content": (
                        "<USER_SETTINGS_CHANGE>The user has changed setting Model Selection "
                        "from Claude Opus 4.6 (Thinking) to Gemini 3.1 Pro (High).</USER_SETTINGS_CHANGE>\n"
                        "<USER_REQUEST>Phase 2</USER_REQUEST>"
                    ),
                },
                {
                    "step_index": 3,
                    "source": "MODEL",
                    "type": "PLANNER_RESPONSE",
                    "status": "DONE",
                    "created_at": "2026-04-19T10:01:10+05:30",
                    "thinking": "t" * 300,
                    "content": "c" * 100,
                    "input_tokens": 500,
                    "cache_read_tokens": 3500,
                    "output_tokens": 400,
                },
            ],
        )

        sessions = stats.discover_sessions(
            [("ide", ide_brain)], history_path=None, local_tz=IST
        )
        report = stats.compute_stats_report(sessions, sources_scanned=["ide"])

        self.assertEqual(
            list(report.by_model.keys()),
            ["Gemini 3.1 Pro (High)", "Claude Opus 4.6 (Thinking)"],
        )
        opus = report.by_model["Claude Opus 4.6 (Thinking)"]
        self.assertEqual(opus.llm_calls, 1)
        self.assertEqual(opus.total_tokens, 3200)
        self.assertEqual(opus.thinking_tokens, 100)
        self.assertEqual(opus.response_tokens, 100)

        gemini = report.by_model["Gemini 3.1 Pro (High)"]
        self.assertEqual(gemini.llm_calls, 1)
        self.assertEqual(gemini.total_tokens, 4400)
        self.assertEqual(gemini.thinking_tokens, 300)
        self.assertEqual(gemini.response_tokens, 100)

        self.assertEqual(
            report.by_model_sessions["Claude Opus 4.6 (Thinking)"], 1
        )
        self.assertEqual(report.by_model_sessions["Gemini 3.1 Pro (High)"], 1)
        self.assertEqual(
            report.sessions[0].models,
            ["Claude Opus 4.6 (Thinking)", "Gemini 3.1 Pro (High)"],
        )

        # CLI --model case-insensitive substring filter
        now_ist = datetime(2026, 4, 19, 12, 0, 0, tzinfo=IST)
        buf_json = io.StringIO()
        with contextlib.redirect_stdout(buf_json):
            rc = stats.main(
                [
                    "--brain-dir",
                    f"ide={ide_brain}",
                    "--model",
                    "gemini",
                    "--json",
                ],
                now=now_ist,
                local_tz=IST,
            )
        self.assertEqual(rc, 0)
        filtered = json.loads(buf_json.getvalue())
        self.assertEqual(filtered["model_filter"], "gemini")
        self.assertEqual(filtered["overall"]["llm_calls"], 1)
        self.assertEqual(filtered["overall"]["total_tokens"], 4400)
        self.assertEqual(filtered["overall"]["thinking_tokens"], 300)
        self.assertEqual(filtered["overall"]["response_tokens"], 100)
        self.assertEqual(
            list(filtered["by_model"].keys()), ["Gemini 3.1 Pro (High)"]
        )

        buf_human = io.StringIO()
        with contextlib.redirect_stdout(buf_human):
            rc_human = stats.main(
                [
                    "--brain-dir",
                    f"ide={ide_brain}",
                    "--model",
                    "gemini",
                ],
                now=now_ist,
                local_tz=IST,
            )
        self.assertEqual(rc_human, 0)
        human_out = buf_human.getvalue()
        self.assertIn("Model Filter   : gemini", human_out)
        self.assertIn("Thinking Tokens (est.)", human_out)
        self.assertIn("Response Tokens (est.)", human_out)
        self.assertIn("Breakdown by Model", human_out)
        self.assertIn("Gemini 3.1 Pro (High)", human_out)

    def test_show_subagents_renders_distinct_self_and_tree_models(self) -> None:
        ide_brain = self.tmp_path / "antigravity/brain"
        root_id = "abababab-abab-abab-abab-abababababab"
        child_id = "cdcdcdcd-cdcd-cdcd-cdcd-cdcdcdcdcdcd"

        write_jsonl(
            ide_brain / root_id / ".system_generated/logs/transcript.jsonl",
            [
                {
                    "step_index": 0,
                    "source": "USER",
                    "type": "USER_INPUT",
                    "status": "DONE",
                    "created_at": "2026-04-19T10:00:00+05:30",
                    "content": (
                        "<USER_SETTINGS_CHANGE>The user has changed setting Model Selection "
                        "from Flash to ParentModel.</USER_SETTINGS_CHANGE>\n"
                        "<USER_REQUEST>Orchestrate</USER_REQUEST>"
                    ),
                },
                {
                    "step_index": 1,
                    "source": "MODEL",
                    "type": "PLANNER_RESPONSE",
                    "status": "DONE",
                    "created_at": "2026-04-19T10:00:05+05:30",
                    "content": "Delegating to child.",
                    "tool_calls": [
                        {
                            "name": "invoke_subagent",
                            "args": {
                                "Subagents": [
                                    {
                                        "TypeName": "oracle",
                                        "Model": "ChildModel",
                                    }
                                ]
                            },
                        }
                    ],
                    "input_tokens": 500,
                    "cache_read_tokens": 1000,
                    "output_tokens": 100,
                },
            ],
        )
        write_subagent_descriptor(
            ide_brain / root_id,
            child_id=child_id,
            type_name="oracle",
            role="Analyze architecture",
            spawn_step_index=1,
        )
        write_jsonl(
            ide_brain / child_id / ".system_generated/logs/transcript.jsonl",
            [
                {
                    "step_index": 0,
                    "source": "SYSTEM",
                    "type": "SYSTEM_MESSAGE",
                    "status": "DONE",
                    "created_at": "2026-04-19T10:00:06+05:30",
                    "content": f"<SYSTEM_MESSAGE>sender={root_id} **Goal**: Analyze architecture</SYSTEM_MESSAGE>",
                },
                {
                    "step_index": 1,
                    "source": "MODEL",
                    "type": "PLANNER_RESPONSE",
                    "status": "DONE",
                    "created_at": "2026-04-19T10:00:10+05:30",
                    "content": "Analysis done.",
                    "input_tokens": 300,
                    "cache_read_tokens": 700,
                    "output_tokens": 50,
                },
            ],
        )

        sessions = stats.discover_sessions(
            [("ide", ide_brain)], history_path=None, local_tz=IST
        )
        report = stats.compute_stats_report(
            sessions, flat=False, sources_scanned=["ide"]
        )
        self.assertEqual(len(report.sessions), 1)
        root_sum = report.sessions[0]
        self.assertEqual(root_sum.model, "ParentModel, ChildModel")
        self.assertEqual(root_sum.self_model, "ParentModel")
        self.assertEqual(root_sum.children[0].model, "ChildModel")
        self.assertEqual(root_sum.children[0].self_model, "ChildModel")

        rendered = stats.format_human_report(report, show_subagents=True)
        root_tree_line = next(
            line for line in rendered.splitlines() if line.strip().startswith("abababab ")
        )
        root_self_line = next(
            line
            for line in rendered.splitlines()
            if "├─ abababab (self)" in line
        )
        child_line = next(
            line for line in rendered.splitlines() if "└─ cdcdcdcd" in line
        )
        self.assertIn("ParentModel, ChildModel", root_tree_line)
        self.assertIn("ParentModel", root_self_line)
        self.assertNotIn("ChildModel", root_self_line)
        self.assertIn("ChildModel", child_line)
        self.assertNotIn("ParentModel", child_line)


if __name__ == "__main__":
    unittest.main()
