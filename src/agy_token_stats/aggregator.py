"""Token aggregation, model breakdown/filtering, and subagent tree/flat rollup computation.

Invariants:
    - Every matched step is counted once in `overall`, `by_source`, `by_role`, `by_model`,
      and `daily`.
    - Orphan subagents (whose `parent_id` is absent from `sessions`) and subagent-rooted
      queries (`--session <subagent>`) are never miscounted in `active_root_sessions` or
      `daily_roots`.
    - Multi-level subagent trees (`Root -> SubA -> SubB`) accurately count both
      `active_subagent_count` (only descendants with active `self_metrics` in the window)
      and `total_subagent_count` (all descendant subagents in `allowed_ids`).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime

from agy_token_stats.models import (
    DailyStats,
    SessionRecord,
    SessionSummary,
    StatsReport,
    StepRecord,
    TimeWindow,
    TokenMetrics,
)


def _resolve_root_id(cid: str, sessions: Mapping[str, SessionRecord]) -> str:
    """Walk `parent_id` links up to the highest present ancestor in `sessions`."""
    curr = cid
    seen: set[str] = {curr}
    while True:
        rec = sessions.get(curr)
        if rec is None or rec.parent_id is None or rec.parent_id not in sessions:
            return curr
        if rec.parent_id in seen:
            return curr
        seen.add(rec.parent_id)
        curr = rec.parent_id


_find_root_session_id = _resolve_root_id


def _collect_descendants(
    root_id: str, sessions: Mapping[str, SessionRecord]
) -> set[str]:
    """Collect `root_id` and all recursive `child_ids` present in `sessions`."""
    collected: set[str] = set()
    queue = [root_id]
    while queue:
        curr = queue.pop(0)
        if curr in collected or curr not in sessions:
            continue
        collected.add(curr)
        for ch_id in sessions[curr].child_ids:
            if ch_id not in collected:
                queue.append(ch_id)
    return collected


def _step_effective_model(st: StepRecord, rec: SessionRecord) -> str:
    """Return the effective model name for `st` in `rec`, or `'Unknown'` if none."""
    return (st.model or rec.model or "").strip() or "Unknown"


def compute_stats_report(
    sessions: Mapping[str, SessionRecord],
    *,
    window: TimeWindow | None = None,
    by_session: bool = False,
    flat: bool = False,
    session_filter: str | None = None,
    model_filter: str | None = None,
    sources_scanned: Sequence[str] | None = None,
) -> StatsReport:
    """Compute token statistics across `sessions` under the given time window and hierarchy mode.

    Filtering behavior:
        - If `session_filter` is set, matches sessions whose `conversation_id` starts with
          `session_filter` (case-insensitive). In tree mode (`flat=False`), matching a session
          also includes all of its descendant subagent sessions.
        - If `by_session=False` (default), filters individual steps by `window.contains(step.created_at_local)`.
        - If `by_session=True`, includes all steps of a session when the session's start time
          (or its root session's start time in tree mode) falls within `window`.
        - If `model_filter` is set, performs a case-insensitive substring match against the
          effective step/session model name.
    """
    active_window = window or TimeWindow(label="all-time")
    scanned = (
        list(sources_scanned)
        if sources_scanned is not None
        else sorted({s.source_kind for s in sessions.values()})
    )
    normalized_model_filter = (
        model_filter.strip() if model_filter and model_filter.strip() else None
    )
    model_needle = (
        normalized_model_filter.lower() if normalized_model_filter else None
    )

    # Filter candidate session IDs by --session prefix if requested
    if session_filter:
        prefix = session_filter.strip().lower()
        matched_roots = [cid for cid in sessions if cid.startswith(prefix)]
        if flat:
            allowed_ids = set(matched_roots)
        else:
            allowed_ids = set()
            for mid in matched_roots:
                allowed_ids |= _collect_descendants(mid, sessions)
    else:
        allowed_ids = set(sessions.keys())

    # Compute filtered steps per session
    filtered_steps_by_id: dict[str, list[StepRecord]] = {}
    self_metrics_by_id: dict[str, TokenMetrics] = {}
    first_active_by_id: dict[str, datetime | None] = {}
    last_active_by_id: dict[str, datetime | None] = {}
    active_models_by_id: dict[str, list[str]] = {}

    for cid in sorted(allowed_ids):
        rec = sessions[cid]
        if by_session:
            anchor_id = cid if flat else _resolve_root_id(cid, sessions)
            anchor_rec = sessions.get(anchor_id, rec)
            anchor_start = anchor_rec.started_at_local
            if anchor_start is not None and active_window.contains(anchor_start):
                matching_steps = list(rec.steps)
            else:
                matching_steps = []
        else:
            matching_steps = [
                st for st in rec.steps if active_window.contains(st.created_at_local)
            ]

        if model_needle is not None:
            matching_steps = [
                st
                for st in matching_steps
                if model_needle in _step_effective_model(st, rec).lower()
            ]

        metrics = TokenMetrics()
        first_dt: datetime | None = None
        last_dt: datetime | None = None
        step_models: list[str] = []
        for st in matching_steps:
            metrics.add_step(st)
            if st.step_type in ("PLANNER_RESPONSE", "USER_INPUT"):
                if first_dt is None or st.created_at_local < first_dt:
                    first_dt = st.created_at_local
                if last_dt is None or st.created_at_local > last_dt:
                    last_dt = st.created_at_local
                eff_m = (st.model or rec.model or "").strip()
                if eff_m and eff_m not in step_models:
                    step_models.append(eff_m)

        if metrics.has_activity and not step_models and rec.model:
            step_models = list(rec.models_used) if rec.models_used else [rec.model]

        filtered_steps_by_id[cid] = matching_steps
        self_metrics_by_id[cid] = metrics
        first_active_by_id[cid] = first_dt
        last_active_by_id[cid] = last_dt
        active_models_by_id[cid] = step_models

    # Aggregate overall, by_source, by_role, by_model, and daily without double counting
    overall = TokenMetrics()
    by_source: dict[str, TokenMetrics] = {}
    by_role: dict[str, TokenMetrics] = {}
    by_model: dict[str, TokenMetrics] = {}
    by_source_sessions: dict[str, int] = {}
    by_role_sessions: dict[str, int] = {}
    by_model_sessions: dict[str, int] = {}

    daily_metrics: dict[str, TokenMetrics] = {}
    daily_roots: dict[str, set[str]] = {}
    daily_subagents: dict[str, set[str]] = {}

    active_subagent_ids: set[str] = set()

    for cid in sorted(allowed_ids):
        rec = sessions[cid]
        s_metrics = self_metrics_by_id[cid]
        if not s_metrics.has_activity:
            continue

        is_subagent = rec.parent_id is not None or rec.agent_type != "root"
        if is_subagent:
            active_subagent_ids.add(cid)

        overall = overall.merge(s_metrics)

        src = rec.source_kind
        by_source[src] = by_source.get(src, TokenMetrics()).merge(s_metrics)
        by_source_sessions[src] = by_source_sessions.get(src, 0) + 1

        role_key = rec.agent_type or ("subagent" if is_subagent else "root")
        by_role[role_key] = by_role.get(role_key, TokenMetrics()).merge(s_metrics)
        by_role_sessions[role_key] = by_role_sessions.get(role_key, 0) + 1

        session_model_accs: dict[str, TokenMetrics] = {}
        root_for_day = cid if flat else _resolve_root_id(cid, sessions)
        for st in filtered_steps_by_id[cid]:
            if st.step_type not in ("PLANNER_RESPONSE", "USER_INPUT"):
                continue

            m_key = _step_effective_model(st, rec)
            session_model_accs.setdefault(m_key, TokenMetrics()).add_step(st)

            day_key = st.created_at_local.strftime("%Y-%m-%d")
            d_acc = daily_metrics.setdefault(day_key, TokenMetrics())
            d_acc.add_step(st)
            if is_subagent and not flat:
                daily_subagents.setdefault(day_key, set()).add(cid)
                if (
                    root_for_day != cid
                    and root_for_day in allowed_ids
                    and sessions[root_for_day].parent_id is None
                    and sessions[root_for_day].agent_type == "root"
                ):
                    daily_roots.setdefault(day_key, set()).add(root_for_day)
            elif is_subagent and flat:
                daily_subagents.setdefault(day_key, set()).add(cid)
            else:
                daily_roots.setdefault(day_key, set()).add(cid)

        for m_key, m_acc in session_model_accs.items():
            if m_acc.has_activity:
                by_model[m_key] = by_model.get(m_key, TokenMetrics()).merge(m_acc)
                by_model_sessions[m_key] = by_model_sessions.get(m_key, 0) + 1

    # Build SessionSummary objects (flat vs tree)
    if flat:
        summaries: list[SessionSummary] = []
        active_root_count = 0
        for cid in sorted(allowed_ids):
            rec = sessions[cid]
            s_metrics = self_metrics_by_id[cid]
            if not s_metrics.has_activity:
                continue
            is_subagent = rec.parent_id is not None or rec.agent_type != "root"
            if not is_subagent:
                active_root_count += 1
            s_models = active_models_by_id[cid]
            s_model_str = ", ".join(s_models) if s_models else rec.model
            summaries.append(
                SessionSummary(
                    conversation_id=cid,
                    source_kind=rec.source_kind,
                    parent_id=rec.parent_id,
                    agent_type=rec.agent_type,
                    agent_role=rec.agent_role,
                    title=rec.title,
                    workspace=rec.workspace,
                    model=s_model_str,
                    first_active_local=first_active_by_id[cid],
                    last_active_local=last_active_by_id[cid],
                    self_metrics=s_metrics,
                    tree_metrics=s_metrics,
                    children=[],
                    active_subagent_count=0,
                    total_subagent_count=len(rec.child_ids),
                    models=list(s_models),
                    self_model=s_model_str,
                )
            )
    else:
        # Top-level roots in allowed_ids are sessions whose parent_id is not in allowed_ids
        top_level_ids = [
            cid
            for cid in sorted(allowed_ids)
            if sessions[cid].parent_id is None
            or sessions[cid].parent_id not in allowed_ids
        ]

        visited_tree: set[str] = set()

        def _build_tree_summary(cid: str) -> SessionSummary | None:
            if cid in visited_tree or cid not in allowed_ids:
                return None
            visited_tree.add(cid)
            rec = sessions[cid]
            self_m = self_metrics_by_id[cid]
            tree_m = self_m
            child_summaries: list[SessionSummary] = []
            first_dt = first_active_by_id[cid]
            last_dt = last_active_by_id[cid]
            active_subs = 0
            total_subs = len(_collect_descendants(cid, sessions) & allowed_ids) - 1
            self_models = list(active_models_by_id[cid]) if self_m.has_activity else []
            self_model_str = ", ".join(self_models) if self_models else rec.model
            combined_models: list[str] = list(self_models)

            for ch_id in rec.child_ids:
                if ch_id not in allowed_ids:
                    continue
                ch_sum = _build_tree_summary(ch_id)
                if ch_sum is None:
                    continue
                if ch_sum.tree_metrics.has_activity:
                    child_summaries.append(ch_sum)
                    tree_m = tree_m.merge(ch_sum.tree_metrics)
                    active_subs += (
                        1 if ch_sum.self_metrics.has_activity else 0
                    ) + ch_sum.active_subagent_count
                    for cm in ch_sum.models:
                        if cm and cm not in combined_models:
                            combined_models.append(cm)
                    if ch_sum.first_active_local is not None and (
                        first_dt is None or ch_sum.first_active_local < first_dt
                    ):
                        first_dt = ch_sum.first_active_local
                    if ch_sum.last_active_local is not None and (
                        last_dt is None or ch_sum.last_active_local > last_dt
                    ):
                        last_dt = ch_sum.last_active_local

            if not tree_m.has_activity:
                return None

            if not combined_models:
                combined_models = list(active_models_by_id[cid])
            model_str = ", ".join(combined_models) if combined_models else rec.model

            return SessionSummary(
                conversation_id=cid,
                source_kind=rec.source_kind,
                parent_id=rec.parent_id,
                agent_type=rec.agent_type,
                agent_role=rec.agent_role,
                title=rec.title,
                workspace=rec.workspace,
                model=model_str,
                first_active_local=first_dt,
                last_active_local=last_dt,
                self_metrics=self_m,
                tree_metrics=tree_m,
                children=child_summaries,
                active_subagent_count=active_subs,
                total_subagent_count=total_subs,
                models=combined_models,
                self_model=self_model_str,
            )

        summaries = []
        for top_id in top_level_ids:
            s_sum = _build_tree_summary(top_id)
            if s_sum is not None:
                summaries.append(s_sum)
        active_root_count = sum(
            1
            for s in summaries
            if s.parent_id is None and s.agent_type == "root"
        )

    summaries.sort(
        key=lambda s: (
            s.last_active_local.timestamp() if s.last_active_local is not None else 0.0,
            s.tree_metrics.total_tokens,
            s.conversation_id,
        ),
        reverse=True,
    )

    daily_list = [
        DailyStats(
            date_str=day_str,
            active_root_sessions=len(daily_roots.get(day_str, set())),
            active_subagent_sessions=len(daily_subagents.get(day_str, set())),
            metrics=daily_metrics[day_str],
        )
        for day_str in sorted(daily_metrics.keys())
    ]

    sorted_by_source = dict(
        sorted(by_source.items(), key=lambda kv: (-kv[1].total_tokens, kv[0]))
    )
    sorted_by_role = dict(
        sorted(by_role.items(), key=lambda kv: (-kv[1].total_tokens, kv[0]))
    )
    sorted_by_model = dict(
        sorted(
            by_model.items(),
            key=lambda kv: (-kv[1].total_tokens, -kv[1].llm_calls, kv[0]),
        )
    )

    return StatsReport(
        window=active_window,
        by_session_mode=by_session,
        flat_mode=flat,
        sources_scanned=scanned,
        active_root_sessions=active_root_count,
        active_subagent_sessions=len(active_subagent_ids),
        overall=overall,
        by_source=sorted_by_source,
        by_role=sorted_by_role,
        by_source_sessions=by_source_sessions,
        by_role_sessions=by_role_sessions,
        daily=daily_list,
        sessions=summaries,
        by_model=sorted_by_model,
        by_model_sessions=by_model_sessions,
        model_filter=normalized_model_filter,
    )


analyze_sessions = compute_stats_report
