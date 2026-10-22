"""Behavior tests for transcript parsing and session discovery (`agy_token_stats.parser`, `discovery`)."""

from __future__ import annotations

import os
import unittest

from tests.helpers import (
    IST,
    TempBrainTestCase,
    write_jsonl,
    write_subagent_descriptor,
)

import agy_token_stats as stats


class ParserAndDiscoveryTests(TempBrainTestCase):
    """Behavior tests for native/fallback token parsing, history enrichment, and filesystem resilience."""

    def test_exact_native_token_accounting_and_context_metrics(self) -> None:
        cli_brain = self.tmp_path / "antigravity-cli/brain"
        session_id = "11111111-1111-1111-1111-111111111111"
        transcript = (
            cli_brain / session_id / ".system_generated/logs/transcript.jsonl"
        )

        write_jsonl(
            transcript,
            [
                {
                    "step_index": 0,
                    "source": "USER",
                    "type": "USER_INPUT",
                    "status": "DONE",
                    "created_at": "2026-04-19T04:30:00Z",
                    "content": (
                        "<ADDITIONAL_METADATA>meta</ADDITIONAL_METADATA>\n"
                        "<USER_SETTINGS_CHANGE>The user has changed setting `Model Selection` "
                        "from Previous Model to Claude Opus 4.6 (Thinking).</USER_SETTINGS_CHANGE>\n"
                        "<USER_REQUEST>\nFix token stats aggregation\n</USER_REQUEST>"
                    ),
                },
                {
                    "step_index": 1,
                    "source": "MODEL",
                    "type": "PLANNER_RESPONSE",
                    "status": "DONE",
                    "created_at": "2026-04-19T04:30:05Z",
                    "content": "Inspecting files.",
                    "input_tokens": 1200,
                    "cache_read_tokens": 2800,
                    "output_tokens": 300,
                    "tool_calls": [
                        {"id": "tc1", "name": "view_file", "args": "{}"},
                        {"id": "tc2", "name": "grep_search", "args": "{}"},
                    ],
                },
                {
                    "step_index": 2,
                    "source": "USER",
                    "type": "USER_INPUT",
                    "status": "DONE",
                    "created_at": "2026-04-19T04:31:00Z",
                    "content": "<USER_REQUEST>Now run the tests</USER_REQUEST>",
                },
                {
                    "step_index": 3,
                    "source": "MODEL",
                    "type": "PLANNER_RESPONSE",
                    "status": "DONE",
                    "created_at": "2026-04-19T04:31:10Z",
                    "content": "Tests passed.",
                    "input_tokens": 800,
                    "cache_read_tokens": 7200,
                    "output_tokens": 500,
                    "tool_calls": [
                        {"id": "tc3", "name": "run_command", "args": "{}"},
                    ],
                },
            ],
        )

        sessions = stats.discover_sessions(
            [("cli", cli_brain)], history_path=None, local_tz=IST
        )
        report = stats.compute_stats_report(sessions, sources_scanned=["cli"])

        # Step 1: gross_in = 1200 + 2800 = 4000, step_ctx = 4300
        # Step 3: gross_in = 800 + 7200 = 8000, step_ctx = 8500
        # Totals: uncached_in = 2000, cache_read = 10000, gross_in = 12000,
        # output = 800, net = 2800, total = 12800, cache_hit_pct = 10000/12000*100 = 83.333...%
        ov = report.overall
        self.assertEqual(ov.llm_calls, 2)
        self.assertEqual(ov.estimated_steps, 0)
        self.assertEqual(ov.user_turns, 2)
        self.assertEqual(ov.tool_calls, 3)
        self.assertEqual(ov.uncached_input, 2000)
        self.assertEqual(ov.cache_read, 10000)
        self.assertEqual(ov.gross_input, 12000)
        self.assertEqual(ov.output_tokens, 800)
        self.assertEqual(ov.net_tokens, 2800)
        self.assertEqual(ov.total_tokens, 12800)
        self.assertEqual(ov.initial_context, 4000)
        self.assertEqual(ov.peak_context, 8500)
        self.assertEqual(round(ov.cache_hit_pct, 2), 83.33)

        self.assertEqual(len(report.sessions), 1)
        summary = report.sessions[0]
        self.assertEqual(summary.conversation_id, session_id)
        self.assertEqual(summary.title, "Fix token stats aggregation")
        self.assertEqual(summary.model, "Claude Opus 4.6 (Thinking)")

    def test_fallback_char_estimation_when_native_token_fields_missing(
        self,
    ) -> None:
        ide_brain = self.tmp_path / "antigravity/brain"
        session_id = "22222222-2222-2222-2222-222222222222"
        transcript = (
            ide_brain / session_id / ".system_generated/logs/transcript.jsonl"
        )

        # Step 0 USER_INPUT has 20 chars -> accumulated_chars = 20
        # Step 1 PLANNER_RESPONSE has no input_tokens/output_tokens, content="abcdefghijkl" (12 chars),
        # thinking="1234" (4 chars) -> step_chars = 16.
        # Expected fallback: input_tokens = ceil(20 / 4) = 5, output_tokens = ceil(16 / 4) = 4,
        # cache_read_tokens = 0, is_estimated = True.
        write_jsonl(
            transcript,
            [
                {
                    "step_index": 0,
                    "source": "USER",
                    "type": "USER_INPUT",
                    "status": "DONE",
                    "created_at": "2026-04-19T05:00:00Z",
                    "content": "12345678901234567890",
                },
                {
                    "step_index": 1,
                    "source": "MODEL",
                    "type": "PLANNER_RESPONSE",
                    "status": "DONE",
                    "created_at": "2026-04-19T05:00:05Z",
                    "content": "abcdefghijkl",
                    "thinking": "1234",
                },
            ],
        )

        sessions = stats.discover_sessions(
            [("ide", ide_brain)], history_path=None, local_tz=IST
        )
        report = stats.compute_stats_report(sessions, sources_scanned=["ide"])

        ov = report.overall
        self.assertEqual(ov.llm_calls, 1)
        self.assertEqual(ov.estimated_steps, 1)
        self.assertEqual(ov.uncached_input, 5)
        self.assertEqual(ov.cache_read, 0)
        self.assertEqual(ov.output_tokens, 4)
        self.assertEqual(ov.net_tokens, 9)
        self.assertEqual(ov.total_tokens, 9)
        self.assertEqual(ov.initial_context, 5)
        self.assertEqual(ov.peak_context, 9)

    def test_proportional_thinking_token_decomposition_and_truncation_marker_expansion(
        self,
    ) -> None:
        ide_brain = self.tmp_path / "antigravity/brain"
        session_id = "33333333-3333-3333-3333-333333333333"
        transcript = (
            ide_brain / session_id / ".system_generated/logs/transcript.jsonl"
        )

        write_jsonl(
            transcript,
            [
                # Step 1: thinking only (other_len == 0) -> all 200 output_tokens attributed to thinking
                {
                    "step_index": 1,
                    "source": "MODEL",
                    "type": "PLANNER_RESPONSE",
                    "status": "DONE",
                    "created_at": "2026-04-19T05:10:00Z",
                    "thinking": "t" * 80,
                    "content": "",
                    "input_tokens": 500,
                    "cache_read_tokens": 1000,
                    "output_tokens": 200,
                },
                # Step 2: thinking (90 chars) + content (18 chars) + serialized tool_calls (42 chars)
                # thinking_len = 90, other_len = 18 + 42 = 60 -> 90 / 150 = 60% of 100 = 60 thinking_tokens
                {
                    "step_index": 2,
                    "source": "MODEL",
                    "type": "PLANNER_RESPONSE",
                    "status": "DONE",
                    "created_at": "2026-04-19T05:11:00Z",
                    "thinking": "t" * 90,
                    "content": "c" * 18,
                    "tool_calls": [{"name": "fn", "args": "x" * 18}],
                    "input_tokens": 600,
                    "cache_read_tokens": 1400,
                    "output_tokens": 100,
                },
                # Step 3: no thinking -> 0 thinking_tokens, has_thinking=False
                {
                    "step_index": 3,
                    "source": "MODEL",
                    "type": "PLANNER_RESPONSE",
                    "status": "DONE",
                    "created_at": "2026-04-19T05:12:00Z",
                    "thinking": "",
                    "content": "Final answer.",
                    "input_tokens": 400,
                    "cache_read_tokens": 1600,
                    "output_tokens": 50,
                },
                # Step 4: thinking with <truncated 90 bytes> marker:
                # 10 prefix chars + 90 truncated bytes = 100 untruncated thinking chars;
                # content has 100 chars -> 50% of 400 = 200 thinking_tokens
                {
                    "step_index": 4,
                    "source": "MODEL",
                    "type": "PLANNER_RESPONSE",
                    "status": "DONE",
                    "created_at": "2026-04-19T05:13:00Z",
                    "thinking": "0123456789<truncated 90 bytes>",
                    "content": "r" * 100,
                    "input_tokens": 500,
                    "cache_read_tokens": 2000,
                    "output_tokens": 400,
                },
            ],
        )

        sessions = stats.discover_sessions(
            [("ide", ide_brain)], history_path=None, local_tz=IST
        )
        steps = sessions[session_id].steps
        self.assertEqual(
            [st.thinking_tokens for st in steps], [200, 60, 0, 200]
        )
        self.assertEqual(
            [st.has_thinking for st in steps], [True, True, False, True]
        )

        report = stats.compute_stats_report(sessions, sources_scanned=["ide"])
        ov = report.overall
        self.assertEqual(ov.llm_calls, 4)
        self.assertEqual(ov.thinking_steps, 3)
        self.assertEqual(ov.output_tokens, 750)
        self.assertEqual(ov.thinking_tokens, 460)
        self.assertEqual(ov.response_tokens, 290)
        self.assertEqual(ov.thinking_tokens + ov.response_tokens, ov.output_tokens)
        self.assertEqual(ov.gross_input, 8000)
        self.assertEqual(ov.total_tokens, ov.gross_input + ov.output_tokens)
        self.assertEqual(ov.net_tokens, ov.uncached_input + ov.output_tokens)

    def test_subagent_and_grandchild_inherit_parent_model_without_settings_change(
        self,
    ) -> None:
        ide_brain = self.tmp_path / "antigravity/brain"
        root_id = "44444444-4444-4444-4444-444444444400"
        child_a_id = "44444444-4444-4444-4444-44444444440a"
        grandchild_id = "44444444-4444-4444-4444-4444444444aa"
        child_b_id = "44444444-4444-4444-4444-44444444440b"

        # Root starts on Claude Opus 4.6 (Thinking) at step 0, spawns child_a at step 1,
        # switches to Gemini 3.1 Pro (High) at step 2 (without backticks around Model Selection),
        # and spawns child_b at step 3.
        write_jsonl(
            ide_brain / root_id / ".system_generated/logs/transcript.jsonl",
            [
                {
                    "step_index": 0,
                    "source": "USER",
                    "type": "USER_INPUT",
                    "status": "DONE",
                    "created_at": "2026-04-19T06:00:00Z",
                    "content": (
                        "<USER_SETTINGS_CHANGE>The user has changed setting `Model Selection` "
                        "from Unknown to Claude Opus 4.6 (Thinking).</USER_SETTINGS_CHANGE>\n"
                        "<USER_REQUEST>Start task</USER_REQUEST>"
                    ),
                },
                {
                    "step_index": 1,
                    "source": "MODEL",
                    "type": "PLANNER_RESPONSE",
                    "status": "DONE",
                    "created_at": "2026-04-19T06:00:10Z",
                    "input_tokens": 1000,
                    "cache_read_tokens": 0,
                    "output_tokens": 100,
                },
                {
                    "step_index": 2,
                    "source": "USER",
                    "type": "USER_INPUT",
                    "status": "DONE",
                    "created_at": "2026-04-19T06:01:00Z",
                    "content": (
                        "<USER_SETTINGS_CHANGE>The user has changed setting Model Selection "
                        "from Claude Opus 4.6 (Thinking) to Gemini 3.1 Pro (High). "
                        "No need to comment on this change if the user doesn't ask about it.</USER_SETTINGS_CHANGE>\n"
                        "<USER_REQUEST>Switch model and continue</USER_REQUEST>"
                    ),
                },
                {
                    "step_index": 3,
                    "source": "MODEL",
                    "type": "PLANNER_RESPONSE",
                    "status": "DONE",
                    "created_at": "2026-04-19T06:01:10Z",
                    "input_tokens": 1200,
                    "cache_read_tokens": 0,
                    "output_tokens": 150,
                },
            ],
        )
        write_subagent_descriptor(
            ide_brain / root_id,
            child_id=child_a_id,
            type_name="implementer",
            role="First worker",
            spawn_step_index=1,
        )
        write_subagent_descriptor(
            ide_brain / root_id,
            child_id=child_b_id,
            type_name="oracle",
            role="Second worker after model switch",
            spawn_step_index=3,
        )

        # child_a has no USER_SETTINGS_CHANGE and spawns grandchild_id at step 1
        write_jsonl(
            ide_brain / child_a_id / ".system_generated/logs/transcript.jsonl",
            [
                {
                    "step_index": 0,
                    "source": "SYSTEM",
                    "type": "SYSTEM_MESSAGE",
                    "status": "DONE",
                    "created_at": "2026-04-19T06:00:15Z",
                    "content": f"<SYSTEM_MESSAGE>sender={root_id} task A</SYSTEM_MESSAGE>",
                },
                {
                    "step_index": 1,
                    "source": "MODEL",
                    "type": "PLANNER_RESPONSE",
                    "status": "DONE",
                    "created_at": "2026-04-19T06:00:25Z",
                    "input_tokens": 400,
                    "cache_read_tokens": 0,
                    "output_tokens": 80,
                },
            ],
        )
        write_subagent_descriptor(
            ide_brain / child_a_id,
            child_id=grandchild_id,
            type_name="bash-executor",
            role="Grandchild runner",
            spawn_step_index=1,
        )

        # grandchild has no USER_SETTINGS_CHANGE
        write_jsonl(
            ide_brain
            / grandchild_id
            / ".system_generated/logs/transcript.jsonl",
            [
                {
                    "step_index": 0,
                    "source": "MODEL",
                    "type": "PLANNER_RESPONSE",
                    "status": "DONE",
                    "created_at": "2026-04-19T06:00:30Z",
                    "input_tokens": 200,
                    "cache_read_tokens": 0,
                    "output_tokens": 40,
                }
            ],
        )

        # child_b has no USER_SETTINGS_CHANGE; spawned at root step 3 when root was on Gemini 3.1 Pro (High)
        write_jsonl(
            ide_brain / child_b_id / ".system_generated/logs/transcript.jsonl",
            [
                {
                    "step_index": 0,
                    "source": "MODEL",
                    "type": "PLANNER_RESPONSE",
                    "status": "DONE",
                    "created_at": "2026-04-19T06:01:20Z",
                    "input_tokens": 500,
                    "cache_read_tokens": 0,
                    "output_tokens": 60,
                }
            ],
        )

        sessions = stats.discover_sessions(
            [("ide", ide_brain)], history_path=None, local_tz=IST
        )
        self.assertEqual(
            sessions[child_a_id].model, "Claude Opus 4.6 (Thinking)"
        )
        self.assertTrue(
            all(
                st.model == "Claude Opus 4.6 (Thinking)"
                for st in sessions[child_a_id].steps
            )
        )
        self.assertEqual(
            sessions[grandchild_id].model, "Claude Opus 4.6 (Thinking)"
        )
        self.assertTrue(
            all(
                st.model == "Claude Opus 4.6 (Thinking)"
                for st in sessions[grandchild_id].steps
            )
        )
        self.assertEqual(sessions[child_b_id].model, "Gemini 3.1 Pro (High)")
        self.assertTrue(
            all(
                st.model == "Gemini 3.1 Pro (High)"
                for st in sessions[child_b_id].steps
            )
        )

    def test_skips_tempmedia_missing_transcripts_and_malformed_lines_and_enriches_history(
        self,
    ) -> None:
        cli_brain = self.tmp_path / "antigravity-cli/brain"
        history_file = self.tmp_path / "antigravity-cli/history.jsonl"

        # Ignored directories
        (cli_brain / "tempmediaStorage").mkdir(parents=True)
        (cli_brain / "empty-session-dir").mkdir(parents=True)

        valid_id = "ffffffff-ffff-ffff-ffff-ffffffffffff"
        write_jsonl(
            cli_brain / valid_id / ".system_generated/logs/transcript.jsonl",
            [
                "{not valid json",
                {
                    "step_index": 0,
                    "type": "PLANNER_RESPONSE",
                    "created_at": "invalid-date",
                },
                {
                    "step_index": 1,
                    "source": "MODEL",
                    "type": "PLANNER_RESPONSE",
                    "status": "DONE",
                    "created_at": "2026-04-19T08:00:00Z",
                    "input_tokens": 400,
                    "cache_read_tokens": 600,
                    "output_tokens": 50,
                },
            ],
        )
        write_jsonl(
            history_file,
            [
                {
                    "conversationId": valid_id,
                    "workspace": "/Users/test/project",
                    "display": "Investigate memory usage",
                    "type": "prompt",
                }
            ],
        )

        sessions = stats.discover_sessions(
            [("cli", cli_brain)], history_path=history_file, local_tz=IST
        )
        self.assertEqual(list(sessions.keys()), [valid_id])
        self.assertEqual(sessions[valid_id].workspace, "/Users/test/project")
        self.assertEqual(sessions[valid_id].title, "Investigate memory usage")
        self.assertEqual(len(sessions[valid_id].steps), 1)

    def test_unreadable_session_directory_is_skipped_without_permission_error(
        self,
    ) -> None:
        if hasattr(os, "geteuid") and os.geteuid() == 0:
            self.skipTest(
                "chmod 000 permission test is not applicable when running as root"
            )

        ide_brain = self.tmp_path / "antigravity/brain"
        valid_id = "77777777-7777-7777-7777-777777777777"
        locked_id = "66666666-6666-6666-6666-666666666666"

        write_jsonl(
            ide_brain / valid_id / ".system_generated/logs/transcript.jsonl",
            [
                {
                    "step_index": 0,
                    "source": "MODEL",
                    "type": "PLANNER_RESPONSE",
                    "status": "DONE",
                    "created_at": "2026-04-19T11:00:00+05:30",
                    "input_tokens": 300,
                    "cache_read_tokens": 200,
                    "output_tokens": 50,
                }
            ],
        )
        locked_dir = ide_brain / locked_id
        write_jsonl(
            locked_dir / ".system_generated/logs/transcript.jsonl",
            [
                {
                    "step_index": 0,
                    "source": "MODEL",
                    "type": "PLANNER_RESPONSE",
                    "status": "DONE",
                    "created_at": "2026-04-19T11:00:00+05:30",
                    "input_tokens": 900,
                    "cache_read_tokens": 0,
                    "output_tokens": 90,
                }
            ],
        )

        locked_dir.chmod(0o000)
        try:
            sessions = stats.discover_sessions(
                [("ide", ide_brain)], history_path=None, local_tz=IST
            )
            self.assertEqual(list(sessions.keys()), [valid_id])
        finally:
            locked_dir.chmod(0o755)

    def test_parallel_invoke_subagent_override_does_not_clobber_inherit_sibling(
        self,
    ) -> None:
        ide_brain = self.tmp_path / "antigravity/brain"
        root_id = "12121212-1212-1212-1212-121212121212"
        oracle_id = "23232323-2323-2323-2323-232323232323"
        impl_id = "34343434-3434-3434-3434-343434343434"

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
                        "from Flash to Claude Opus 4.6 (Thinking).</USER_SETTINGS_CHANGE>\n"
                        "<USER_REQUEST>Run parallel tasks</USER_REQUEST>"
                    ),
                },
                {
                    "step_index": 1,
                    "source": "MODEL",
                    "type": "PLANNER_RESPONSE",
                    "status": "DONE",
                    "created_at": "2026-04-19T10:00:05+05:30",
                    "content": "Spawning oracle and implementer in parallel.",
                    "tool_calls": [
                        {
                            "name": "invoke_subagent",
                            "args": {
                                "Subagents": [
                                    {
                                        "TypeName": "oracle",
                                        "Task": "Inspect graph",
                                        "Model": "ProOverride",
                                    },
                                    {
                                        "TypeName": "implementer",
                                        "Task": "Apply fix",
                                        "Model": "inherit",
                                    },
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
            child_id=oracle_id,
            type_name="oracle",
            role="Inspect graph",
            spawn_step_index=1,
        )
        write_jsonl(
            ide_brain / oracle_id / ".system_generated/logs/transcript.jsonl",
            [
                {
                    "step_index": 0,
                    "source": "SYSTEM",
                    "type": "SYSTEM_MESSAGE",
                    "status": "DONE",
                    "created_at": "2026-04-19T10:00:06+05:30",
                    "content": f"<SYSTEM_MESSAGE>sender={root_id} **Goal**: Inspect graph</SYSTEM_MESSAGE>",
                },
                {
                    "step_index": 1,
                    "source": "MODEL",
                    "type": "PLANNER_RESPONSE",
                    "status": "DONE",
                    "created_at": "2026-04-19T10:00:10+05:30",
                    "content": "Graph inspected.",
                    "input_tokens": 300,
                    "cache_read_tokens": 200,
                    "output_tokens": 50,
                },
            ],
        )

        write_subagent_descriptor(
            ide_brain / root_id,
            child_id=impl_id,
            type_name="implementer",
            role="Apply fix",
            spawn_step_index=1,
        )
        write_jsonl(
            ide_brain / impl_id / ".system_generated/logs/transcript.jsonl",
            [
                {
                    "step_index": 0,
                    "source": "SYSTEM",
                    "type": "SYSTEM_MESSAGE",
                    "status": "DONE",
                    "created_at": "2026-04-19T10:00:06+05:30",
                    "content": f"<SYSTEM_MESSAGE>sender={root_id} **Goal**: Apply fix</SYSTEM_MESSAGE>",
                },
                {
                    "step_index": 1,
                    "source": "MODEL",
                    "type": "PLANNER_RESPONSE",
                    "status": "DONE",
                    "created_at": "2026-04-19T10:00:12+05:30",
                    "content": "Fix applied.",
                    "input_tokens": 400,
                    "cache_read_tokens": 300,
                    "output_tokens": 80,
                },
            ],
        )

        sessions = stats.discover_sessions(
            [("ide", ide_brain)], history_path=None, local_tz=IST
        )
        self.assertEqual(sessions[oracle_id].model, "ProOverride")
        self.assertEqual(
            [st.model for st in sessions[oracle_id].steps],
            ["ProOverride", "ProOverride"],
        )
        self.assertEqual(
            sessions[impl_id].model, "Claude Opus 4.6 (Thinking)"
        )
        self.assertEqual(
            [st.model for st in sessions[impl_id].steps],
            ["Claude Opus 4.6 (Thinking)", "Claude Opus 4.6 (Thinking)"],
        )

    def test_subagent_inherits_parent_model_for_initial_steps_before_mid_session_switch(
        self,
    ) -> None:
        ide_brain = self.tmp_path / "antigravity/brain"
        root_id = "45454545-4545-4545-4545-454545454545"
        child_id = "56565656-5656-5656-5656-565656565656"

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
                        "from Flash to Model A.</USER_SETTINGS_CHANGE>\n"
                        "<USER_REQUEST>Start work</USER_REQUEST>"
                    ),
                },
                {
                    "step_index": 1,
                    "source": "MODEL",
                    "type": "PLANNER_RESPONSE",
                    "status": "DONE",
                    "created_at": "2026-04-19T10:00:05+05:30",
                    "content": "Spawning subagent.",
                    "input_tokens": 500,
                    "cache_read_tokens": 500,
                    "output_tokens": 50,
                },
            ],
        )
        write_subagent_descriptor(
            ide_brain / root_id,
            child_id=child_id,
            type_name="implementer",
            role="Implement feature",
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
                    "content": f"<SYSTEM_MESSAGE>sender={root_id} **Goal**: Implement feature</SYSTEM_MESSAGE>",
                },
                {
                    "step_index": 1,
                    "source": "MODEL",
                    "type": "PLANNER_RESPONSE",
                    "status": "DONE",
                    "created_at": "2026-04-19T10:00:10+05:30",
                    "content": "Initial step under inherited Model A.",
                    "input_tokens": 400,
                    "cache_read_tokens": 600,
                    "output_tokens": 100,
                },
                {
                    "step_index": 2,
                    "source": "USER",
                    "type": "USER_INPUT",
                    "status": "DONE",
                    "created_at": "2026-04-19T10:01:00+05:30",
                    "content": (
                        "<USER_SETTINGS_CHANGE>The user has changed setting Model Selection "
                        "from Model A to Model B.</USER_SETTINGS_CHANGE>\n"
                        "<USER_REQUEST>Continue with Model B</USER_REQUEST>"
                    ),
                },
                {
                    "step_index": 3,
                    "source": "MODEL",
                    "type": "PLANNER_RESPONSE",
                    "status": "DONE",
                    "created_at": "2026-04-19T10:01:10+05:30",
                    "content": "Second step under Model B.",
                    "input_tokens": 300,
                    "cache_read_tokens": 700,
                    "output_tokens": 200,
                },
            ],
        )

        sessions = stats.discover_sessions(
            [("ide", ide_brain)], history_path=None, local_tz=IST
        )
        child_rec = sessions[child_id]
        self.assertEqual(
            [st.model for st in child_rec.steps],
            ["Model A", "Model A", "Model B", "Model B"],
        )
        self.assertEqual(child_rec.models_used, ["Model A", "Model B"])

        report = stats.compute_stats_report(sessions, sources_scanned=["ide"])
        self.assertEqual(report.by_model["Model A"].llm_calls, 2)
        self.assertEqual(report.by_model["Model A"].total_tokens, 2150)
        self.assertEqual(report.by_model["Model B"].llm_calls, 1)
        self.assertEqual(report.by_model["Model B"].total_tokens, 1200)


if __name__ == "__main__":
    unittest.main()
