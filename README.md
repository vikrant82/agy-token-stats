# agy-token-stats

Zero-dependency CLI and Python library for analyzing token usage, prompt cache efficiency, thinking token decomposition, model breakdowns, context growth, and subagent trees across **Antigravity (`agy`)** CLI and IDE sessions.

---

## Overview

Antigravity records structured per-step execution transcripts for every root session and spawned subagent conversation across two local brain directories:

- **CLI brain**: `~/.gemini/antigravity-cli/brain/<conversation-id>/`
- **IDE brain**: `~/.gemini/antigravity/brain/<conversation-id>/`

Within each conversation directory:

- `.system_generated/logs/transcript.jsonl` stores chronological step records (`USER_INPUT`, `PLANNER_RESPONSE`, `SYSTEM_MESSAGE`, etc.), including native token counters (`input_tokens`, `cache_read_tokens`, `output_tokens`), `thinking` text, `content`, and tool call payloads.
- `.system_generated/subagents/<child-id>.json` stores explicit parent-to-child subagent descriptors (`typeName`, `role`, `spawnStepIndex`, `workspaceUris`).
- `~/.gemini/antigravity-cli/history.jsonl` (optional) stores workspace paths and user prompt history for CLI sessions.

`agy-token-stats` discovers all sessions across CLI and IDE brains, reconstructs multi-level subagent trees (`root` -> `implementer` / `oracle` / `reviewer` / `bash-executor`), tracks and propagates model selections across steps and subagents, filters by local-time windows or `--model`, and produces human-readable tables or structured JSON reports without double counting.

---

## Token Metrics, Thinking Decomposition & Subagent Tree Rollups

### Token Metric Definitions

| Metric | Formula / Meaning |
|---|---|
| `uncached_input` | Sum of native `input_tokens` across `PLANNER_RESPONSE` steps (prompt tokens not served from cache). |
| `cache_read` | Sum of native `cache_read_tokens` across `PLANNER_RESPONSE` steps. |
| `gross_input` | `uncached_input + cache_read` — total prompt context processed by the model across all calls. |
| `cache_hit_pct` | `cache_read / gross_input * 100` (`0.0%` when `gross_input == 0`). |
| `output_tokens` | Sum of native `output_tokens` generated across `PLANNER_RESPONSE` steps (includes thinking + visible content + tool calls). |
| `thinking_tokens` | Estimated portion of `output_tokens` spent on `thinking` text (`0 <= thinking_tokens <= output_tokens`). |
| `response_tokens` | `max(0, output_tokens - thinking_tokens)` — estimated non-thinking output tokens (visible content + tool calls). |
| `thinking_pct` | `thinking_tokens / output_tokens * 100` (`0.0%` when `output_tokens == 0`). |
| `thinking_steps` | Number of `PLANNER_RESPONSE` steps with non-empty `thinking` text. |
| `net_tokens` | `uncached_input + output_tokens` — non-cached tokens processed plus output generated. (`thinking_tokens` is a decomposition of `output_tokens` and is never double-counted.) |
| `total_tokens` | `gross_input + output_tokens` — total tokens processed including prompt cache reads. |
| `initial_context` | `gross_input` of the first `PLANNER_RESPONSE` step in the session/window. |
| `peak_context` | Maximum single-step context (`input_tokens + cache_read_tokens + output_tokens`) across included `PLANNER_RESPONSE` steps. |

### Thinking Token Estimation

Native Antigravity `PLANNER_RESPONSE` entries emit a single combined `output_tokens` counter covering internal reasoning (`thinking`), visible response text (`content`), and serialized tool invocations (`tool_calls`).

For each `PLANNER_RESPONSE` step with non-empty `thinking`:
1. Untruncated text sizes (`thinking_len`, `content_len`, `tool_calls_len`) are computed, expanding any `<truncated N bytes>` markers recorded in `transcript.jsonl`.
2. When `output_tokens > 0`:
   - If `content_len + tool_calls_len == 0`, `thinking_tokens = output_tokens`.
   - Otherwise, `thinking_tokens = min(output_tokens, max(1, round(output_tokens * (thinking_len / (thinking_len + content_len + tool_calls_len)))))`.
3. `response_tokens = output_tokens - thinking_tokens`, preserving the exact invariant `thinking_tokens + response_tokens == output_tokens`.

> **Fallback character estimation**: If a legacy or truncated `PLANNER_RESPONSE` step omits both `input_tokens` and `output_tokens`, `agy-token-stats` estimates `input_tokens = ceil(accumulated_context_chars / 4)` and `output_tokens = ceil(step_text_chars / 4)`, sets `is_estimated = True`, and reports the exact count of estimated steps in the header.

### Model Attribution & Subagent Inheritance

- **Sequential step tracking**: `<USER_SETTINGS_CHANGE>` model tags (`changed setting Model Selection from ... to <Model>.`) are tracked chronologically across `USER_INPUT` steps. If a user switches models mid-session, steps before and after the switch are attributed to their respective models in `by_model`.
- **Subagent inheritance**: Subagent transcripts begin with `SYSTEM_MESSAGE` (`step_index: 0`) and omit `<USER_SETTINGS_CHANGE>`. During discovery, models are propagated top-down from parent sessions to child and grandchild subagents using the parent's active model at `spawnStepIndex` (or any explicit `invoke_subagent` `Model` override).

### Subagent Tree Rollup vs. Flat Mode

1. **Parent-Child Linking**:
   - **Primary**: Explicit descriptor files in `<parent>/.system_generated/subagents/<child>.json` provide `parent_id`, `typeName` (`oracle`, `implementer`, `reviewer`, `bash-executor`, etc.), `role`, `spawnStepIndex`, and workspace fallback.
   - **Fallback**: If a descriptor file is missing, an initial `SYSTEM_MESSAGE` step containing `sender=<parent-uuid>` links the subagent to its parent session (with cycle prevention).
2. **No Double Counting**:
   - `overall`, `by_source`, `by_role`, `by_model`, and `daily` aggregate each active step exactly once.
   - In **tree mode** (default), the `Sessions` table displays top-level root sessions with `tree_metrics` (parent `self_metrics` + all descendant subagents' `self_metrics`). Passing `--show-subagents` (or `--session <id>`) expands an indented breakdown showing `├─ <id> (self)` and each active child/grandchild subagent row.
   - In **flat mode** (`--flat`), every conversation UUID is listed independently with `tree_metrics == self_metrics`.

---

## Installation

`agy-token-stats` requires **Python 3.9+** and has **zero third-party runtime dependencies**.

```bash
# Install isolated CLI globally with uv or pipx from GitHub
uv tool install git+https://github.com/vikrant82/agy-token-stats.git
# or
pipx install git+https://github.com/vikrant82/agy-token-stats.git

# Or install from a local clone
git clone https://github.com/vikrant82/agy-token-stats.git
cd agy-token-stats
pip install -e .
```

---

## CLI Options

| Flag | Description |
|---|---|
| `--source {all,cli,ide}` | Brain source(s) to scan when `--brain-dir` is not specified (default: `all`). |
| `--brain-dir [LABEL=]PATH` | Override brain root directory to scan (repeatable, e.g. `--brain-dir cli=/path/to/brain`). |
| `--history-file PATH` | Path to CLI `history.jsonl` for workspace/title enrichment. |
| `--today` | Include activity from today `00:00:00` local time to now. |
| `--yesterday` | Include activity from yesterday `00:00:00` to `23:59:59.999999` local time. |
| `--last-n-days N`, `--days N` | Include activity from the last `N` calendar days in local time including today (`N=1` equals `--today`; `N=7` covers today + 6 preceding days). |
| `--last-n-hours H`, `--hours H` | Include activity from the rolling last `H` hours up to now. |
| `--since DATE_OR_ISO` | Lower bound in local time (`YYYY-MM-DD` at `00:00:00` or ISO-8601 timestamp). |
| `--until DATE_OR_ISO` | Upper bound in local time (`YYYY-MM-DD` at `23:59:59.999999` or ISO-8601 timestamp). |
| `--by-session` | Filter whole sessions by session start time instead of individual step timestamps. |
| `--session ID_OR_PREFIX` | Filter to a conversation UUID (or prefix) and expand its subagent tree. |
| `--model PATTERN` | Filter steps/sessions by model name using a case-insensitive substring match (e.g. `--model claude`, `--model flash`). |
| `--flat` | Treat every conversation UUID independently without rolling subagents into parent roots. |
| `--show-subagents` | Display indented child subagent rows beneath each parent session in the sessions table. |
| `--top N` | Number of sessions to show in the sessions table (`0` for all; default: `20`). |
| `--daily` / `--no-daily` | Force-enable or hide the local-date daily breakdown table. |
| `--json` | Emit structured JSON output instead of the human-readable report. |

---

## Usage Examples

```bash
# Today's token usage across both CLI and IDE brains
agy-token-stats --today

# Filter to a specific model family over the last 7 calendar days
agy-token-stats --days 7 --model claude

# Last 7 calendar days with daily breakdown and expanded subagent trees
agy-token-stats --days 7 --daily --show-subagents

# Rolling last 4 hours for IDE sessions only
agy-token-stats --source ide --hours 4

# Drill into a specific root or subagent session by UUID prefix
agy-token-stats --session 179c4f24

# Export JSON report for a custom date range
agy-token-stats --since 2026-04-01 --until 2026-04-19 --json
```

---

## Package Architecture

```text
agy-token-stats/
├── LICENSE
├── pyproject.toml
├── README.md
├── src/agy_token_stats/
│   ├── __init__.py       # Public package API and version metadata
│   ├── __main__.py       # `python3 -m agy_token_stats` entry point
│   ├── models.py         # Dataclasses: StepRecord, SessionRecord, TokenMetrics, StatsReport, etc.
│   ├── time_window.py    # Timezone resolution, ISO-8601 parsing, calendar/rolling window bounds
│   ├── parser.py         # Transcript JSONL step parsing, thinking split, model tracking, char/4 fallback
│   ├── discovery.py      # Brain directory scanning, history.jsonl enrichment, subagent model inheritance
│   ├── aggregator.py     # Step/session/model filtering and tree/flat rollup aggregation
│   ├── formatters.py     # Aligned human-readable CLI tables and JSON report serialization
│   └── cli.py            # Argument parser, source path resolution, and CLI main()
└── tests/
    ├── helpers.py
    ├── test_time_window.py
    ├── test_parser_and_discovery.py
    └── test_aggregator_and_cli.py
```

### Running Tests

```bash
python3 -m unittest discover -s tests -t . -v
```
