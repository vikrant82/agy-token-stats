"""Transcript JSONL step parser, thinking token decomposition, and model attribution helpers.

Invariants & Error Behavior:
    - Malformed JSON lines, non-object JSON values, and steps without valid ISO-8601
      `created_at` timestamps are skipped without aborting transcript parsing.
    - Unreadable or missing transcript files return `(None, None)` without raising `OSError`.
    - When a `PLANNER_RESPONSE` step lacks both `input_tokens` and `output_tokens`,
      fallback character estimation (`ceil(chars / 4)`) is used and `is_estimated=True`.
    - On `PLANNER_RESPONSE` steps with non-empty `thinking`, `thinking_tokens` is
      decomposed proportionally from `output_tokens` using untruncated lengths
      (accounting for `<truncated N bytes>` markers) so `0 <= thinking_tokens <= output_tokens`.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import tzinfo
import json
import math
from pathlib import Path
import re
from typing import Any

from agy_token_stats.models import SessionRecord, StepRecord
from agy_token_stats.time_window import parse_iso_timestamp

SENDER_UUID_RE = re.compile(
    r"\bsender[=:]\s*([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\b",
    re.IGNORECASE,
)
USER_REQUEST_RE = re.compile(
    r"<USER_REQUEST>\s*(.*?)\s*</USER_REQUEST>",
    re.DOTALL,
)
MODEL_SELECTION_RE = re.compile(
    r"changed setting `?Model Selection`? from [^\n]+? to ([^\n]+?)\.(?=\s|<|$)",
)
METADATA_TAG_RE = re.compile(
    r"<(?:ADDITIONAL_METADATA|USER_SETTINGS_CHANGE)>.*?</(?:ADDITIONAL_METADATA|USER_SETTINGS_CHANGE)>",
    re.DOTALL,
)
GOAL_LINE_RE = re.compile(r"\*\*Goal\*\*:\s*([^\n]+)")
PEER_HEADER_RE = re.compile(r"^\[Peer message[^\]]*\]\s*", re.MULTILINE)
TRUNCATED_MARKER_RE = re.compile(r"<truncated\s+(\d+)\s+bytes>", re.IGNORECASE)


def _safe_is_file(path: Path) -> bool:
    """Return True if `path` is an accessible regular file, suppressing `OSError`."""
    try:
        return path.is_file()
    except OSError:
        return False


def _untruncated_text_len(text: str) -> int:
    """Compute effective length of `text`, replacing `<truncated N bytes>` markers with `N`."""
    if not text:
        return 0
    total = len(text)
    for match in TRUNCATED_MARKER_RE.finditer(text):
        total = total - len(match.group(0)) + int(match.group(1))
    return max(0, total)


def _serialized_tool_calls_len(tool_calls: Any) -> int:
    """Compute untruncated serialized length of `tool_calls`."""
    if not tool_calls:
        return 0
    try:
        serialized = json.dumps(
            tool_calls, separators=(",", ":"), ensure_ascii=False
        )
    except (TypeError, ValueError):
        serialized = str(tool_calls)
    return _untruncated_text_len(serialized)


def _step_text_chars(entry: Mapping[str, Any]) -> int:
    """Count untruncated length of `content`, `thinking`, and serialized `tool_calls` in a step."""
    content_len = _untruncated_text_len(str(entry.get("content") or ""))
    thinking_len = _untruncated_text_len(str(entry.get("thinking") or ""))
    tc_len = _serialized_tool_calls_len(entry.get("tool_calls"))
    return content_len + thinking_len + tc_len


def _extract_subagent_model_overrides(
    step_index: int,
    raw_tc: Any,
    overrides: dict[tuple[int, str], str],
) -> None:
    """Extract explicit non-inherit `Model` overrides from `invoke_subagent` tool calls."""
    if not isinstance(raw_tc, list):
        return
    for tc in raw_tc:
        if not isinstance(tc, dict):
            continue
        name = str(tc.get("name") or "")
        if name != "invoke_subagent":
            continue
        raw_args = tc.get("args")
        if isinstance(raw_args, str):
            try:
                raw_args = json.loads(raw_args)
            except (TypeError, ValueError):
                raw_args = None
        if not isinstance(raw_args, dict):
            continue

        top_model = str(raw_args.get("Model") or raw_args.get("model") or "").strip()
        if top_model and top_model.lower() != "inherit":
            overrides[(step_index, "*")] = top_model

        subagents_list = raw_args.get("Subagents") or raw_args.get("subagents")
        if isinstance(subagents_list, list):
            for sub_item in subagents_list:
                if not isinstance(sub_item, dict):
                    continue
                sub_model = str(
                    sub_item.get("Model") or sub_item.get("model") or ""
                ).strip()
                if not sub_model or sub_model.lower() == "inherit":
                    continue
                t_name = (
                    str(
                        sub_item.get("TypeName")
                        or sub_item.get("typeName")
                        or sub_item.get("name")
                        or "*"
                    )
                    .strip()
                    .lower()
                    or "*"
                )
                overrides[(step_index, t_name)] = sub_model


def _clean_prompt_preview(raw_text: str, max_len: int = 80) -> str:
    """Extract a clean single-line title/prompt preview from user or system message content."""
    if not raw_text:
        return ""
    req_match = USER_REQUEST_RE.search(raw_text)
    if req_match:
        candidate = req_match.group(1).strip()
    else:
        goal_match = GOAL_LINE_RE.search(raw_text)
        if goal_match:
            candidate = f"Goal: {goal_match.group(1).strip()}"
        else:
            stripped = METADATA_TAG_RE.sub("", raw_text)
            if "content=" in stripped and "<SYSTEM_MESSAGE>" in stripped:
                stripped = stripped.split("content=", 1)[1]
            stripped = PEER_HEADER_RE.sub("", stripped)
            stripped = stripped.replace("</SYSTEM_MESSAGE>", "").strip()
            candidate = stripped

    lines = [ln.strip() for ln in candidate.splitlines() if ln.strip()]
    if not lines:
        return ""
    one_line = " ".join(lines)
    one_line = re.sub(r"\s+", " ", one_line).strip()
    if len(one_line) > max_len:
        return one_line[: max_len - 1].rstrip() + "…"
    return one_line


def parse_transcript_file(
    transcript_path: Path,
    *,
    conversation_id: str,
    source_kind: str,
    brain_dir: Path,
    local_tz: tzinfo,
) -> tuple[SessionRecord | None, str | None]:
    """Parse a single `transcript.jsonl` into a `SessionRecord` and optional step-0 sender UUID.

    Fallback token estimation rule:
        For any `PLANNER_RESPONSE` step where both `input_tokens` and `output_tokens`
        are `None`, `input_tokens` is estimated as `ceil(accumulated_context_chars / 4)`,
        `output_tokens` as `ceil(step_text_chars / 4)`, `cache_read_tokens` as `0`, and
        `is_estimated` is set to `True`.

    Thinking token decomposition rule:
        For any `PLANNER_RESPONSE` step with non-empty `thinking` (`thinking_len > 0`),
        `has_thinking` is set to `True`. When `output_tokens > 0`, `thinking_tokens` is
        `output_tokens` if `other_len == 0` (`content_len + tool_calls_len == 0`), or
        `min(output_tokens, max(1, round(output_tokens * (thinking_len / (thinking_len + other_len)))))`.

    Returns:
        `(session_record, fallback_parent_uuid)` or `(None, None)` if the transcript
        has no valid timestamped steps or cannot be read.
    """
    if not _safe_is_file(transcript_path):
        return None, None

    steps: list[StepRecord] = []
    accumulated_chars = 0
    title = ""
    current_model: str | None = None
    models_used: list[str] = []
    spawn_model_overrides: dict[tuple[int, str], str] = {}
    fallback_parent_id: str | None = None

    try:
        with transcript_path.open("r", encoding="utf-8", errors="replace") as fh:
            for line_num, raw_line in enumerate(fh):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(entry, dict):
                    continue

                ts_pair = parse_iso_timestamp(entry.get("created_at"), local_tz)
                if ts_pair is None:
                    continue
                dt_utc, dt_local = ts_pair

                raw_idx = entry.get("step_index")
                step_index = int(raw_idx) if isinstance(raw_idx, int) else line_num
                source = str(entry.get("source") or "")
                step_type = str(entry.get("type") or entry.get("event") or "")
                status = str(entry.get("status") or "")
                content_str = str(entry.get("content") or "")
                thinking_str = str(entry.get("thinking") or "")

                if step_type == "USER_INPUT" and content_str:
                    if not title:
                        title = _clean_prompt_preview(content_str)
                    for m_mod in MODEL_SELECTION_RE.finditer(content_str):
                        candidate_model = m_mod.group(1).strip()
                        if candidate_model:
                            current_model = candidate_model
                            if candidate_model not in models_used:
                                models_used.append(candidate_model)
                elif (
                    not steps
                    and source == "SYSTEM"
                    and step_type == "SYSTEM_MESSAGE"
                    and content_str
                ):
                    m_sender = SENDER_UUID_RE.search(content_str)
                    if m_sender:
                        sender_uuid = m_sender.group(1).lower()
                        if sender_uuid != conversation_id:
                            fallback_parent_id = sender_uuid
                    if not title:
                        title = _clean_prompt_preview(content_str)

                content_len = _untruncated_text_len(content_str)
                thinking_len = _untruncated_text_len(thinking_str)
                raw_tc = entry.get("tool_calls")
                tc_len = _serialized_tool_calls_len(raw_tc)
                step_chars = content_len + thinking_len + tc_len

                in_tok = 0
                cache_tok = 0
                out_tok = 0
                thinking_tok = 0
                has_thinking = False
                tc_count = 0
                is_estimated = False

                if step_type == "PLANNER_RESPONSE":
                    if isinstance(raw_tc, list):
                        tc_count = len(raw_tc)
                        _extract_subagent_model_overrides(
                            step_index, raw_tc, spawn_model_overrides
                        )

                    raw_in = entry.get("input_tokens")
                    raw_cache = entry.get("cache_read_tokens")
                    raw_out = entry.get("output_tokens")

                    has_native_in = isinstance(raw_in, (int, float)) and not isinstance(
                        raw_in, bool
                    )
                    has_native_out = isinstance(
                        raw_out, (int, float)
                    ) and not isinstance(raw_out, bool)

                    if not has_native_in and not has_native_out:
                        in_tok = (
                            math.ceil(accumulated_chars / 4)
                            if accumulated_chars > 0
                            else 0
                        )
                        out_tok = math.ceil(step_chars / 4) if step_chars > 0 else 0
                        cache_tok = 0
                        is_estimated = True
                    else:
                        in_tok = max(0, int(raw_in)) if has_native_in else 0
                        out_tok = max(0, int(raw_out)) if has_native_out else 0
                        if isinstance(raw_cache, (int, float)) and not isinstance(
                            raw_cache, bool
                        ):
                            cache_tok = max(0, int(raw_cache))

                    if thinking_len > 0:
                        has_thinking = True
                        other_len = content_len + tc_len
                        if out_tok > 0:
                            if other_len == 0:
                                thinking_tok = out_tok
                            else:
                                thinking_tok = min(
                                    out_tok,
                                    max(
                                        1,
                                        round(
                                            out_tok
                                            * (
                                                thinking_len
                                                / (thinking_len + other_len)
                                            )
                                        ),
                                    ),
                                )

                accumulated_chars += step_chars

                steps.append(
                    StepRecord(
                        step_index=step_index,
                        source=source,
                        step_type=step_type,
                        status=status,
                        created_at_utc=dt_utc,
                        created_at_local=dt_local,
                        input_tokens=in_tok,
                        cache_read_tokens=cache_tok,
                        output_tokens=out_tok,
                        thinking_tokens=thinking_tok,
                        has_thinking=has_thinking,
                        tool_calls_count=tc_count,
                        is_estimated=is_estimated,
                        model=current_model,
                    )
                )
    except OSError:
        return None, None

    if not steps:
        return None, None

    steps.sort(key=lambda s: (s.created_at_utc, s.step_index))

    # Back-fill early steps that occurred before the first model tag for root sessions.
    # Subagent transcripts (starting with SYSTEM_MESSAGE) defer leading-step backfill to
    # discovery so steps before any mid-session model switch inherit the parent/descriptor model.
    is_subagent_start = bool(
        steps and steps[0].source == "SYSTEM" and steps[0].step_type == "SYSTEM_MESSAGE"
    )
    initial_model = models_used[0] if models_used else current_model
    if initial_model and not is_subagent_start:
        for st in steps:
            if st.model is None:
                st.model = initial_model

    started_at_local = steps[0].created_at_local
    last_active_local = steps[-1].created_at_local

    record = SessionRecord(
        conversation_id=conversation_id,
        source_kind=source_kind,
        brain_dir=brain_dir,
        transcript_path=transcript_path,
        steps=steps,
        title=title,
        model=current_model or "",
        models_used=models_used,
        spawn_model_overrides=spawn_model_overrides,
        started_at_local=started_at_local,
        last_active_local=last_active_local,
    )
    return record, fallback_parent_id
