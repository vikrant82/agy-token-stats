"""Data models for Antigravity (agy) transcript steps, sessions, token metrics, and reports.

Threading & Mutability:
    - `SubagentLink` and `TimeWindow` are immutable (`@dataclass(frozen=True)`) and safe to
      share across threads.
    - `StepRecord` is mutated only during single-threaded session discovery when back-filling
      early step models or propagating parent models to child subagents.
    - Accumulator and container models (`TokenMetrics`, `SessionRecord`, `SessionSummary`,
      `DailyStats`, `StatsReport`) are mutated only during single-threaded discovery or
      report construction. `TokenMetrics.merge()` is pure and returns a new instance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class StepRecord:
    """Single parsed transcript step with normalized UTC/local timestamps, model, and token counts.

    Invariants:
        - `created_at_utc` has `tzinfo == timezone.utc`.
        - `created_at_local` is converted to the caller's target `local_tz`.
        - Token fields (`input_tokens`, `cache_read_tokens`, `output_tokens`,
          `thinking_tokens`, `tool_calls_count`) are non-negative integers.
        - `0 <= thinking_tokens <= output_tokens`.
        - `has_thinking` is True iff the step contained non-empty `thinking` text.
        - `is_estimated` is True only when a `PLANNER_RESPONSE` lacked native token counters
          and fell back to character-based estimation.
    """

    step_index: int
    source: str
    step_type: str
    status: str
    created_at_utc: datetime
    created_at_local: datetime
    input_tokens: int = 0
    cache_read_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    has_thinking: bool = False
    tool_calls_count: int = 0
    is_estimated: bool = False
    model: str | None = None


@dataclass(frozen=True)
class SubagentLink:
    """Metadata linking a parent session to a spawned child subagent conversation.

    Invariants:
        - `conversation_id` is lowercase.
        - `type_name` is non-empty (defaults to `"subagent"` when unknown).
    """

    conversation_id: str
    type_name: str = "subagent"
    role: str = ""
    spawn_step_index: int | None = None
    workspace_uris: tuple[str, ...] = ()
    model: str = ""


@dataclass
class SessionRecord:
    """Complete parsed representation of a single Antigravity conversation directory.

    Invariants:
        - `steps` are ordered chronologically by `(created_at_utc, step_index)`.
        - `parent_id`, if set, references another session's `conversation_id` without cycles.
        - `agent_type` is `"root"` for top-level sessions or the subagent `typeName`
          (such as `"oracle"`, `"implementer"`, `"reviewer"`, `"bash-executor"`).
        - `model` is the latest model observed in the session (or inherited from its parent
          when the session has no explicit `<USER_SETTINGS_CHANGE>`).
        - `models_used` preserves unique model names in chronological order of appearance.
    """

    conversation_id: str
    source_kind: str
    brain_dir: Path
    transcript_path: Path
    steps: list[StepRecord]
    parent_id: str | None = None
    agent_type: str = "root"
    agent_role: str = "main"
    spawn_step_index: int | None = None
    child_ids: list[str] = field(default_factory=list)
    title: str = ""
    workspace: str = ""
    model: str = ""
    models_used: list[str] = field(default_factory=list)
    spawn_model_overrides: dict[tuple[int, str], str] = field(default_factory=dict)
    started_at_local: datetime | None = None
    last_active_local: datetime | None = None


@dataclass
class TokenMetrics:
    """Aggregated token and call statistics across a set of transcript steps.

    Invariants:
        - `gross_input == uncached_input + cache_read`
        - `net_tokens == uncached_input + output_tokens`
        - `total_tokens == gross_input + output_tokens`
        - `response_tokens == max(0, output_tokens - thinking_tokens)`
        - `thinking_tokens` is a decomposition of `output_tokens` and is NOT added to
          `net_tokens` or `total_tokens`.
        - `cache_hit_pct` and `thinking_pct` are in `[0.0, 100.0]`.
        - `peak_context` is the maximum single-step context (`input + cache_read + output`)
          across all `PLANNER_RESPONSE` steps included in the aggregation.
    """

    llm_calls: int = 0
    estimated_steps: int = 0
    user_turns: int = 0
    tool_calls: int = 0
    uncached_input: int = 0
    cache_read: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    thinking_steps: int = 0
    initial_context: int = 0
    peak_context: int = 0

    @property
    def gross_input(self) -> int:
        """Total prompt tokens processed (`uncached_input + cache_read`)."""
        return self.uncached_input + self.cache_read

    @property
    def response_tokens(self) -> int:
        """Estimated non-thinking output tokens (`max(0, output_tokens - thinking_tokens)`)."""
        return max(0, self.output_tokens - self.thinking_tokens)

    @property
    def thinking_pct(self) -> float:
        """Percentage of `output_tokens` spent on thinking (`0.0` if `output_tokens == 0`)."""
        if self.output_tokens <= 0:
            return 0.0
        return (self.thinking_tokens / self.output_tokens) * 100.0

    @property
    def net_tokens(self) -> int:
        """Non-cached tokens processed plus generated output (`uncached_input + output_tokens`)."""
        return self.uncached_input + self.output_tokens

    @property
    def total_tokens(self) -> int:
        """Total tokens processed including cache reads (`gross_input + output_tokens`)."""
        return self.gross_input + self.output_tokens

    @property
    def cache_hit_pct(self) -> float:
        """Percentage of gross input satisfied from prompt cache (`0.0` if `gross_input == 0`)."""
        gross = self.gross_input
        if gross <= 0:
            return 0.0
        return (self.cache_read / gross) * 100.0

    @property
    def has_activity(self) -> bool:
        """True if at least one LLM call or user turn was recorded."""
        return self.llm_calls > 0 or self.user_turns > 0

    def add_step(self, step: StepRecord) -> None:
        """Incorporate a single `StepRecord` into this metrics accumulator."""
        if step.step_type == "USER_INPUT":
            self.user_turns += 1
            return
        if step.step_type != "PLANNER_RESPONSE":
            return

        gross_in = step.input_tokens + step.cache_read_tokens
        step_ctx = gross_in + step.output_tokens
        if self.llm_calls == 0:
            self.initial_context = gross_in
        self.llm_calls += 1
        if step.is_estimated:
            self.estimated_steps += 1
        if step.has_thinking:
            self.thinking_steps += 1
        self.tool_calls += step.tool_calls_count
        self.uncached_input += step.input_tokens
        self.cache_read += step.cache_read_tokens
        self.output_tokens += step.output_tokens
        self.thinking_tokens += step.thinking_tokens
        if step_ctx > self.peak_context:
            self.peak_context = step_ctx

    def merge(self, other: TokenMetrics) -> TokenMetrics:
        """Return a new `TokenMetrics` combining `self` and `other` without mutating either."""
        return TokenMetrics(
            llm_calls=self.llm_calls + other.llm_calls,
            estimated_steps=self.estimated_steps + other.estimated_steps,
            user_turns=self.user_turns + other.user_turns,
            tool_calls=self.tool_calls + other.tool_calls,
            uncached_input=self.uncached_input + other.uncached_input,
            cache_read=self.cache_read + other.cache_read,
            output_tokens=self.output_tokens + other.output_tokens,
            thinking_tokens=self.thinking_tokens + other.thinking_tokens,
            thinking_steps=self.thinking_steps + other.thinking_steps,
            initial_context=(
                self.initial_context if self.initial_context > 0 else other.initial_context
            ),
            peak_context=max(self.peak_context, other.peak_context),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize metrics and derived properties to a JSON-compatible dictionary."""
        return {
            "llm_calls": self.llm_calls,
            "estimated_steps": self.estimated_steps,
            "thinking_steps": self.thinking_steps,
            "user_turns": self.user_turns,
            "tool_calls": self.tool_calls,
            "uncached_input": self.uncached_input,
            "cache_read": self.cache_read,
            "gross_input": self.gross_input,
            "cache_hit_pct": round(self.cache_hit_pct, 2),
            "output_tokens": self.output_tokens,
            "thinking_tokens": self.thinking_tokens,
            "response_tokens": self.response_tokens,
            "thinking_pct": round(self.thinking_pct, 2),
            "net_tokens": self.net_tokens,
            "total_tokens": self.total_tokens,
            "initial_context": self.initial_context,
            "peak_context": self.peak_context,
        }


@dataclass(frozen=True)
class TimeWindow:
    """Inclusive local-time filtering window `[start_local, end_local]`.

    Invariants:
        - When both `start_local` and `end_local` are non-None, `start_local <= end_local`.
        - `contains(dt)` checks inclusive membership against non-None bounds.
    """

    label: str
    start_local: datetime | None = None
    end_local: datetime | None = None

    def contains(self, dt_local: datetime) -> bool:
        """Return True if `dt_local` falls within `[start_local, end_local]` inclusive."""
        if self.start_local is not None and dt_local < self.start_local:
            return False
        if self.end_local is not None and dt_local > self.end_local:
            return False
        return True


@dataclass
class SessionSummary:
    """Per-session token summary supporting both self-only and subagent-tree rollups.

    Invariants:
        - In flat mode (or when a session has no children), `tree_metrics == self_metrics`
          and `children` is empty.
        - In tree mode, `tree_metrics` equals `self_metrics` merged across all `children`.
    """

    conversation_id: str
    source_kind: str
    parent_id: str | None
    agent_type: str
    agent_role: str
    title: str
    workspace: str
    model: str
    first_active_local: datetime | None
    last_active_local: datetime | None
    self_metrics: TokenMetrics
    tree_metrics: TokenMetrics
    children: list[SessionSummary] = field(default_factory=list)
    active_subagent_count: int = 0
    total_subagent_count: int = 0
    models: list[str] = field(default_factory=list)
    self_model: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize session summary and any child subagent summaries to a dictionary."""
        return {
            "conversation_id": self.conversation_id,
            "source_kind": self.source_kind,
            "parent_id": self.parent_id,
            "agent_type": self.agent_type,
            "agent_role": self.agent_role,
            "title": self.title,
            "workspace": self.workspace,
            "model": self.model,
            "self_model": self.self_model,
            "models": list(self.models),
            "first_active_local": (
                self.first_active_local.isoformat() if self.first_active_local else None
            ),
            "last_active_local": (
                self.last_active_local.isoformat() if self.last_active_local else None
            ),
            "active_subagent_count": self.active_subagent_count,
            "total_subagent_count": self.total_subagent_count,
            "self_metrics": self.self_metrics.to_dict(),
            "tree_metrics": self.tree_metrics.to_dict(),
            "children": [ch.to_dict() for ch in self.children],
        }


@dataclass
class DailyStats:
    """Token statistics aggregated for a single local calendar date (`YYYY-MM-DD`)."""

    date_str: str
    active_root_sessions: int
    active_subagent_sessions: int
    metrics: TokenMetrics

    def to_dict(self) -> dict[str, Any]:
        """Serialize daily stats to a dictionary."""
        return {
            "date": self.date_str,
            "active_root_sessions": self.active_root_sessions,
            "active_subagent_sessions": self.active_subagent_sessions,
            "metrics": self.metrics.to_dict(),
        }


@dataclass
class StatsReport:
    """Complete token usage report across matched Antigravity sessions.

    Invariants:
        - Every matched step is counted once in `overall`, `by_source`, `by_role`, `by_model`,
          and `daily`.
        - Sum of `tree_metrics.total_tokens` across top-level `sessions` equals `overall.total_tokens`.
    """

    window: TimeWindow
    by_session_mode: bool
    flat_mode: bool
    sources_scanned: list[str]
    active_root_sessions: int
    active_subagent_sessions: int
    overall: TokenMetrics
    by_source: dict[str, TokenMetrics]
    by_role: dict[str, TokenMetrics]
    by_source_sessions: dict[str, int]
    by_role_sessions: dict[str, int]
    daily: list[DailyStats]
    sessions: list[SessionSummary]
    by_model: dict[str, TokenMetrics] = field(default_factory=dict)
    by_model_sessions: dict[str, int] = field(default_factory=dict)
    model_filter: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the complete report to a JSON-compatible dictionary."""
        return {
            "window": {
                "label": self.window.label,
                "start_local": (
                    self.window.start_local.isoformat() if self.window.start_local else None
                ),
                "end_local": (
                    self.window.end_local.isoformat() if self.window.end_local else None
                ),
                "filter_mode": "session_start" if self.by_session_mode else "step_timestamp",
                "hierarchy_mode": "flat" if self.flat_mode else "tree",
                "model_filter": self.model_filter,
            },
            "model_filter": self.model_filter,
            "sources_scanned": list(self.sources_scanned),
            "active_root_sessions": self.active_root_sessions,
            "active_subagent_sessions": self.active_subagent_sessions,
            "overall": self.overall.to_dict(),
            "by_source": {
                k: {
                    "sessions": self.by_source_sessions.get(k, 0),
                    **v.to_dict(),
                }
                for k, v in self.by_source.items()
            },
            "by_role": {
                k: {
                    "sessions": self.by_role_sessions.get(k, 0),
                    **v.to_dict(),
                }
                for k, v in self.by_role.items()
            },
            "by_model": {
                k: {
                    "sessions": self.by_model_sessions.get(k, 0),
                    **v.to_dict(),
                }
                for k, v in self.by_model.items()
            },
            "daily": [d.to_dict() for d in self.daily],
            "sessions": [s.to_dict() for s in self.sessions],
        }
