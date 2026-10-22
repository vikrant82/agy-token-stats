"""Brain directory scanning, CLI history enrichment, and cycle-safe subagent tree linking.

Invariants & Error Behavior:
    - All filesystem checks (`_safe_is_dir`, `_safe_is_file`, `iterdir`, file reads)
      catch `OSError` (`PermissionError`, broken symlinks, missing paths) and skip
      inaccessible entries without crashing discovery.
    - Explicit `.system_generated/subagents/<child>.json` descriptors take precedence
      over fallback step-0 `SYSTEM_MESSAGE` (`sender=<uuid>`) links.
    - Parent-child links are verified with `_creates_cycle` so `parent_id` chains are
      strictly acyclic.
    - Subagents without their own `<USER_SETTINGS_CHANGE>` model tags inherit the active
      model from their parent session at `spawn_step_index` (or parent start/latest model)
      top-down across multi-level subagent trees.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import tzinfo
import json
from pathlib import Path

from agy_token_stats.models import SessionRecord, SubagentLink
from agy_token_stats.parser import (
    _clean_prompt_preview,
    _safe_is_file,
    parse_transcript_file,
)
from agy_token_stats.time_window import get_default_local_tz

DEFAULT_CLI_BRAIN_DIR = Path.home() / ".gemini/antigravity-cli/brain"
DEFAULT_IDE_BRAIN_DIR = Path.home() / ".gemini/antigravity/brain"
DEFAULT_CLI_HISTORY_PATH = Path.home() / ".gemini/antigravity-cli/history.jsonl"

SKIPPED_BRAIN_DIRS = frozenset({"tempmediaStorage", ".DS_Store"})


def _safe_is_dir(path: Path) -> bool:
    """Return True if `path` is an accessible directory, suppressing `OSError`."""
    try:
        return path.is_dir()
    except OSError:
        return False


def infer_source_kind(path: Path) -> str:
    """Infer a source label (`'cli'`, `'ide'`, or directory name) from a brain directory path."""
    norm = str(path).lower()
    if (
        "antigravity-cli" in norm
        or path.name.lower() == "cli"
        or "cli" in path.name.lower()
    ):
        return "cli"
    if (
        "antigravity" in norm
        or path.name.lower() == "ide"
        or "ide" in path.name.lower()
    ):
        return "ide"
    return path.name or "custom"


def load_cli_history_metadata(history_path: Path | None) -> dict[str, dict[str, str]]:
    """Read optional `~/.gemini/antigravity-cli/history.jsonl` for workspace and prompt fallback.

    Returns a mapping `{conversation_id_lower: {"workspace": ..., "title": ...}}`.
    Gracefully returns `{}` if `history_path` is None, missing, or unreadable.
    """
    if history_path is None or not _safe_is_file(history_path):
        return {}
    meta: dict[str, dict[str, str]] = {}
    try:
        with history_path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(entry, dict):
                    continue
                cid = str(entry.get("conversationId") or "").strip().lower()
                if not cid:
                    continue
                rec = meta.setdefault(cid, {"workspace": "", "title": ""})
                ws = str(entry.get("workspace") or "").strip()
                if ws and not rec["workspace"]:
                    rec["workspace"] = ws
                if entry.get("type") != "slash_command" and not rec["title"]:
                    disp = str(entry.get("display") or "").strip()
                    if disp and not disp.startswith("/"):
                        rec["title"] = _clean_prompt_preview(disp)
    except OSError:
        return meta
    return meta


def _load_subagent_descriptors(session_dir: Path) -> list[SubagentLink]:
    """Read `<session_dir>/.system_generated/subagents/*.json` descriptors."""
    sub_dir = session_dir / ".system_generated/subagents"
    if not _safe_is_dir(sub_dir):
        return []
    links: list[SubagentLink] = []
    try:
        children = sorted(sub_dir.iterdir(), key=lambda p: p.name)
    except OSError:
        return []
    for p in children:
        if not _safe_is_file(p) or p.suffix.lower() != ".json":
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8", errors="replace"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        cid = str(data.get("conversationId") or p.stem).strip().lower()
        if not cid:
            continue
        desc = data.get("subagentDescriptor")
        if not isinstance(desc, dict):
            desc = {}
        type_name = str(desc.get("typeName") or "subagent").strip() or "subagent"
        role = str(desc.get("role") or "").strip()
        raw_model = str(desc.get("model") or data.get("model") or "").strip()
        link_model = raw_model if raw_model.lower() != "inherit" else ""
        spawn_idx_raw = data.get("spawnStepIndex")
        spawn_idx = int(spawn_idx_raw) if isinstance(spawn_idx_raw, int) else None
        raw_uris = data.get("workspaceUris")
        uris: list[str] = []
        if isinstance(raw_uris, list):
            for u in raw_uris:
                if isinstance(u, str) and u.strip():
                    uris.append(u.strip().removeprefix("file://"))
        links.append(
            SubagentLink(
                conversation_id=cid,
                type_name=type_name,
                role=role,
                spawn_step_index=spawn_idx,
                workspace_uris=tuple(uris),
                model=link_model,
            )
        )
    return links


def _resolve_parent_model_for_child(
    parent_rec: SessionRecord, child_rec: SessionRecord
) -> str:
    """Resolve the active model on `parent_rec` when `child_rec` was spawned."""
    spawn_idx = child_rec.spawn_step_index
    if spawn_idx is not None:
        override = parent_rec.spawn_model_overrides.get(
            (spawn_idx, child_rec.agent_type.lower())
        ) or parent_rec.spawn_model_overrides.get((spawn_idx, "*"))
        if override:
            return override
        at_or_before = [
            st.model
            for st in parent_rec.steps
            if st.step_index <= spawn_idx and st.model
        ]
        if at_or_before:
            return at_or_before[-1] or ""

    if child_rec.started_at_local is not None:
        by_time = [
            st.model
            for st in parent_rec.steps
            if st.created_at_local <= child_rec.started_at_local and st.model
        ]
        if by_time:
            return by_time[-1] or ""

    return parent_rec.model


def discover_sessions(
    brain_sources: Sequence[tuple[str, Path] | Path],
    *,
    history_path: Path | None = DEFAULT_CLI_HISTORY_PATH,
    local_tz: tzinfo | None = None,
) -> dict[str, SessionRecord]:
    """Discover and parse all Antigravity sessions and subagent links across `brain_sources`.

    Linking precedence:
        1. Explicit descriptor files under `<parent>/.system_generated/subagents/<child>.json`
           populate `parent_id`, `agent_type` (`subagentDescriptor.typeName`), `agent_role`
           (`subagentDescriptor.role`), `spawn_step_index`, and `workspace`.
        2. Fallback step-0 `SYSTEM_MESSAGE` (`sender=<parent-uuid>`) links child sessions
           whose parent descriptor file is absent, without creating parent-child cycles.
        3. Subagents without an explicit model inherit their parent's active model at
           `spawn_step_index` (or parent start/latest model) top-down across the tree.

    Returns:
        Dictionary mapping lowercase `conversation_id` to `SessionRecord`.
    """
    tz = local_tz or get_default_local_tz()
    history_meta = load_cli_history_metadata(history_path)

    sessions: dict[str, SessionRecord] = {}
    explicit_links: dict[str, list[SubagentLink]] = {}
    fallback_parents: dict[str, str] = {}

    for item in brain_sources:
        if isinstance(item, tuple):
            source_kind, brain_dir = item[0], Path(item[1]).expanduser()
        else:
            brain_dir = Path(item).expanduser()
            source_kind = infer_source_kind(brain_dir)

        if not _safe_is_dir(brain_dir):
            continue

        try:
            entries = sorted(brain_dir.iterdir(), key=lambda p: p.name)
        except OSError:
            continue

        for session_dir in entries:
            if session_dir.name in SKIPPED_BRAIN_DIRS or not _safe_is_dir(session_dir):
                continue

            cid = session_dir.name.strip().lower()
            if not cid:
                continue

            sub_links = _load_subagent_descriptors(session_dir)
            if sub_links:
                explicit_links.setdefault(cid, []).extend(sub_links)

            transcript_path = session_dir / ".system_generated/logs/transcript.jsonl"
            rec, fb_parent = parse_transcript_file(
                transcript_path,
                conversation_id=cid,
                source_kind=source_kind,
                brain_dir=brain_dir,
                local_tz=tz,
            )
            if rec is None:
                continue

            hist = history_meta.get(cid)
            if hist:
                if not rec.workspace and hist.get("workspace"):
                    rec.workspace = hist["workspace"]
                if not rec.title and hist.get("title"):
                    rec.title = hist["title"]

            existing = sessions.get(cid)
            if existing is None or len(rec.steps) > len(existing.steps):
                sessions[cid] = rec
            if fb_parent:
                fallback_parents[cid] = fb_parent

    def _creates_cycle(child_id: str, candidate_parent_id: str) -> bool:
        curr: str | None = candidate_parent_id
        seen: set[str] = {child_id}
        while curr is not None:
            if curr in seen:
                return True
            seen.add(curr)
            parent_rec = sessions.get(curr)
            curr = parent_rec.parent_id if parent_rec else None
        return False

    # 1. Apply explicit `.system_generated/subagents/*.json` links
    for parent_id, links in explicit_links.items():
        parent_rec = sessions.get(parent_id)
        for link in links:
            child_id = link.conversation_id
            if child_id == parent_id:
                continue
            child_rec = sessions.get(child_id)
            if child_rec is not None and not _creates_cycle(child_id, parent_id):
                child_rec.parent_id = parent_id
                child_rec.agent_type = link.type_name or "subagent"
                child_rec.agent_role = link.role or link.type_name or "subagent"
                child_rec.spawn_step_index = link.spawn_step_index
                if link.model:
                    for st in child_rec.steps:
                        if st.model is None:
                            st.model = link.model
                    if link.model not in child_rec.models_used:
                        child_rec.models_used.insert(0, link.model)
                    if not child_rec.model:
                        child_rec.model = link.model
                if link.workspace_uris and not child_rec.workspace:
                    child_rec.workspace = link.workspace_uris[0]
                if link.role and (
                    not child_rec.title or child_rec.title.startswith("Goal:")
                ):
                    child_rec.title = (
                        f"{link.role} — {child_rec.title}"
                        if child_rec.title
                        else link.role
                    )
                if (
                    parent_rec is not None
                    and not parent_rec.workspace
                    and link.workspace_uris
                ):
                    parent_rec.workspace = link.workspace_uris[0]
            if (
                parent_rec is not None
                and child_id in sessions
                and child_id not in parent_rec.child_ids
            ):
                parent_rec.child_ids.append(child_id)

    # 2. Apply fallback step-0 SYSTEM_MESSAGE `sender=<uuid>` links
    for child_id, parent_id in fallback_parents.items():
        child_rec = sessions.get(child_id)
        if child_rec is None or child_rec.parent_id is not None:
            continue
        if parent_id == child_id or _creates_cycle(child_id, parent_id):
            continue
        child_rec.parent_id = parent_id
        if child_rec.agent_type == "root":
            child_rec.agent_type = "subagent"
        if child_rec.agent_role == "main":
            child_rec.agent_role = "subagent"
        parent_rec = sessions.get(parent_id)
        if parent_rec is not None and child_id not in parent_rec.child_ids:
            parent_rec.child_ids.append(child_id)

    # Sort each parent's child_ids chronologically by start time
    for rec in sessions.values():
        rec.child_ids.sort(
            key=lambda cid: (
                sessions[cid].started_at_local.timestamp()
                if cid in sessions and sessions[cid].started_at_local is not None
                else 0.0,
                sessions[cid].spawn_step_index or 0 if cid in sessions else 0,
                cid,
            )
        )

    # 3. Propagate models top-down from root sessions to child/grandchild subagents
    visited_models: set[str] = set()

    def _propagate_model(cid: str) -> None:
        if cid in visited_models:
            return
        visited_models.add(cid)
        rec = sessions.get(cid)
        if rec is None:
            return
        if rec.parent_id and rec.parent_id in sessions:
            _propagate_model(rec.parent_id)
            if not rec.model or any(st.model is None for st in rec.steps):
                inherited = _resolve_parent_model_for_child(
                    sessions[rec.parent_id], rec
                )
                if inherited:
                    for st in rec.steps:
                        if st.model is None:
                            st.model = inherited
                    if inherited not in rec.models_used:
                        rec.models_used.insert(0, inherited)
                    if not rec.model:
                        rec.model = inherited
        fill_model = rec.models_used[0] if rec.models_used else rec.model
        if fill_model:
            for st in rec.steps:
                if st.model is None:
                    st.model = fill_model
        for ch_id in rec.child_ids:
            _propagate_model(ch_id)

    for cid in sorted(sessions.keys()):
        _propagate_model(cid)

    return sessions
