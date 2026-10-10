"""Sprint state at in-sprint checkpoints, rebuilt from the TAWOS Change_Log.

For every sprint of the Phase 2 cohort and every checkpoint (25%, 50%, 75% and
100% of the planned duration) this module rebuilds what a team could have seen
at that moment, using ONLY information time-stamped at or before the checkpoint:

  * which of the issues committed at the start are already resolved
  * which issues were added to / removed from the sprint since the start
  * how the story points of the committed issues changed
  * how much time is left and how fast work would have to be finished

Membership and story points at any time T use the same Change_Log rule as
Phase 1's planning snapshot (reverse every event after T), so the
"committed at start" set reproduces Phase 1's scope exactly (checked in tests).

A sprint that was already closed at the checkpoint is not part of that
checkpoint's population: its outcome is known, so there is nothing to predict.
At 100% of the planned duration the spillover label is fully determined by the
state (any committed issue still unresolved), so spillover is not a prediction
task there; late closure still is.
"""
from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict

import numpy as np
import pandas as pd

import phase3  # noqa: F401
from phase2.data import STRICT_NUMERIC, load_phase2_frame, population, time_ordered_split
from src.data.load_tawos import load_change_log, load_issues, load_sprints
from src.data.planning_snapshot import (
    _as_number,
    _field_kind,
    _parse_change_sprint_set,
    _project_maps,
    _story_after,
    _story_before,
)

CHECKPOINTS = (0.25, 0.5, 0.75, 1.0)
MIN_DAYS = 0.5     # floor for time-based denominators
RATIO_CAP = 5.0
RATE_CAP = 50.0

PROGRESS_FEATURES = [
    "elapsed_frac",
    "remaining_days",
    "done_share",
    "done_n",
    "remaining_n",
    "remaining_sp",
    "burn_gap",
    "required_rate",
    "observed_rate",
    "added_ratio",
    "added_done_n",
    "removed_ratio",
    "sp_change_ratio",
]
CHECKPOINT_NUMERIC = STRICT_NUMERIC + PROGRESS_FEATURES

GROUPS = {
    "scope": ["committed_issue_count", "committed_story_points"],
    "sprint length": ["sprint_length_days"],
    "parallel sprints": ["n_concurrent_sprints"],
    "recent history": ["hist_avg_completion_ratio_3", "hist_last_carryover", "hist_avg_delivered_sp_3",
                       "hist_n_available", "hist_avg_completed_issues_3", "hist_avg_committed_issues_3",
                       "scope_to_capacity"],
    "past lateness": ["hist_avg_late_days_3", "hist_late_rate_3"],
    "progress so far": ["done_share", "done_n", "remaining_n", "remaining_sp", "burn_gap", "observed_rate",
                        "added_done_n"],
    "time pressure": ["elapsed_frac", "remaining_days", "required_rate"],
    "scope change": ["added_ratio", "removed_ratio", "sp_change_ratio"],
    "project": ["project"],
}


def _issue_timelines(issues, change_log, maps):
    """Yield per-issue lookups for membership/story points at any time (ns ints)."""
    jira_maps, raw_maps, name_maps = maps
    rel = change_log[change_log["field"].map(_field_kind).notna()].dropna(subset=["raw_issue_id", "change_date"])
    rel = rel.sort_values(["raw_issue_id", "change_date", "change_id"])
    by_issue = defaultdict(list)
    for row in rel.itertuples(index=False):
        by_issue[row.raw_issue_id].append(row)
    return by_issue


def build_checkpoint_state(checkpoints=CHECKPOINTS, *, sprints=None, issues=None, change_log=None,
                           cohort_ids=None) -> pd.DataFrame:
    """Raw aggregates per (sprint, checkpoint): one row each, for the Phase 2 cohort.

    The optional arguments let tests pass small synthetic tables; by default the
    TAWOS export and the Phase 2 cohort are used.
    """
    if sprints is None:
        sprints = load_sprints(phase3.RAW_DIR, modeling_only=True)
    if issues is None:
        issues = load_issues(phase3.RAW_DIR)
    if change_log is None:
        change_log = load_change_log(phase3.RAW_DIR, required=True)
    base = (load_phase2_frame()[["sprint_id"]] if cohort_ids is None
            else pd.DataFrame({"sprint_id": list(cohort_ids)}))

    cohort = sprints.merge(base, on="sprint_id", how="inner")
    maps = _project_maps(sprints)
    jira_maps, raw_maps, name_maps = maps
    events = _issue_timelines(issues, change_log, maps)

    # query times (ns) per sprint: start + each checkpoint
    start_ns, end_ns, times = {}, {}, {}
    for r in cohort.itertuples(index=False):
        s, e = r.start_date.value, r.end_date.value
        start_ns[r.sprint_id], end_ns[r.sprint_id] = s, e
        times[r.sprint_id] = [s + f * (e - s) for f in checkpoints]
    cohort_ids = set(cohort["sprint_id"])
    K = len(checkpoints)

    acc = {sid: {k: defaultdict(float) for k in range(K)} for sid in cohort_ids}

    for issue in issues.itertuples(index=False):
        project = issue.project
        if project not in jira_maps:
            continue
        ev = events.get(issue.raw_issue_id, [])
        jira_map, raw_map, name_map = jira_maps[project], raw_maps[project], name_maps[project]
        sp_ev = [e for e in ev if _field_kind(e.field) == "sprint"]
        st_ev = [e for e in ev if _field_kind(e.field) == "story_point"]

        sp_times = [e.change_date.value for e in sp_ev]
        sp_from = [_parse_change_sprint_set(e.from_value, e.from_string, jira_map=jira_map, name_map=name_map)[0]
                   for e in sp_ev]
        if sp_ev:
            terminal = _parse_change_sprint_set(sp_ev[-1].to_value, sp_ev[-1].to_string,
                                                jira_map=jira_map, name_map=name_map)[0]
            candidates = set(terminal)
            for e in sp_ev:
                for val, sv in ((e.from_value, e.from_string), (e.to_value, e.to_string)):
                    candidates |= _parse_change_sprint_set(val, sv, jira_map=jira_map, name_map=name_map)[0]
        else:
            terminal = set()
            if not pd.isna(issue.raw_sprint_id):
                key = str(int(float(issue.raw_sprint_id)))
                if key in raw_map:
                    terminal = {raw_map[key]}
            candidates = set(terminal)
        candidates &= cohort_ids
        if not candidates:
            continue

        st_times = [e.change_date.value for e in st_ev]
        static_sp = None if pd.isna(issue.story_point) else float(issue.story_point)
        st_terminal = _story_after(st_ev[-1]) if st_ev else static_sp

        def members(t):
            i = bisect_right(sp_times, t)
            return sp_from[i] if i < len(sp_times) else terminal

        def story(t):
            i = bisect_right(st_times, t)
            v = _story_before(st_ev[i]) if i < len(st_times) else st_terminal
            return 0.0 if v is None else float(v)

        created = None if pd.isna(issue.creation_date) else issue.creation_date.value
        resolved = None if pd.isna(issue.resolution_date) else issue.resolution_date.value

        for sid in candidates:
            t0 = start_ns[sid]
            in0 = (sid in members(t0) and (created is None or created <= t0)
                   and not (resolved is not None and resolved < t0))
            sp0 = story(t0) if in0 else 0.0
            for k, tk in enumerate(times[sid]):
                a = acc[sid][k]
                in_k = sid in members(tk) and (created is None or created <= tk)
                done_k = resolved is not None and resolved <= tk
                if in0:
                    a["committed_n"] += 1
                    a["committed_sp"] += sp0
                    a["sp_now"] += story(tk)
                    if done_k:
                        a["done_n"] += 1
                        a["done_sp"] += sp0
                    if not in_k:
                        a["removed_n"] += 1
                elif in_k and not (resolved is not None and resolved < t0):
                    a["added_n"] += 1
                    if done_k:
                        a["added_done_n"] += 1

    rows = []
    for r in cohort.itertuples(index=False):
        for k, f in enumerate(checkpoints):
            a = acc[r.sprint_id][k]
            rows.append({
                "sprint_id": r.sprint_id, "checkpoint": f,
                "end_date": r.end_date, "complete_date": r.complete_date,
                "checkpoint_time": pd.Timestamp(int(times[r.sprint_id][k])),
                "committed_n": a["committed_n"], "committed_sp": a["committed_sp"],
                "done_n": a["done_n"], "done_sp": a["done_sp"], "removed_n": a["removed_n"],
                "added_n": a["added_n"], "added_done_n": a["added_done_n"],
                "sp_now": a["sp_now"],
            })
    return pd.DataFrame(rows)


def add_progress_features(state: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    """Merge the Phase 2 cohort frame and derive the in-sprint features."""
    d = state.merge(base, on="sprint_id", how="inner")
    elapsed_days = (d["checkpoint_time"] - d["start_date"]).dt.total_seconds() / 86400.0
    remaining_days = ((d["end_date"] - d["checkpoint_time"]).dt.total_seconds() / 86400.0).clip(lower=0.0)
    c = d["committed_n"]
    ne = c > 0
    d["elapsed_frac"] = d["checkpoint"]
    d["remaining_days"] = remaining_days
    d["done_share"] = np.where(ne, d["done_n"] / c.where(ne, 1), 0.0)
    d["remaining_n"] = c - d["done_n"]
    d["remaining_sp"] = d["committed_sp"] - d["done_sp"]
    d["burn_gap"] = np.where(ne, d["done_share"] - d["checkpoint"], 0.0)
    d["required_rate"] = d["remaining_n"] / remaining_days.clip(lower=MIN_DAYS)
    d["observed_rate"] = d["done_n"] / elapsed_days.clip(lower=MIN_DAYS)
    # ratios are capped so a handful of tiny sprints that grew enormously cannot dominate linear models
    d["added_ratio"] = (d["added_n"] / c.clip(lower=1)).clip(upper=RATIO_CAP)
    d["removed_ratio"] = (d["removed_n"] / c.clip(lower=1)).clip(upper=RATIO_CAP)
    d["sp_change_ratio"] = ((d["sp_now"] - d["committed_sp"]) / d["committed_sp"].clip(lower=1.0)).clip(-1.0, RATIO_CAP)
    d["required_rate"] = d["required_rate"].clip(upper=RATE_CAP)
    d["observed_rate"] = d["observed_rate"].clip(upper=RATE_CAP)
    d["open_at_checkpoint"] = d["complete_date"] > d["checkpoint_time"]
    return d


def load_checkpoint_frame() -> pd.DataFrame:
    """Long frame: one row per (sprint, checkpoint) with Phase 2 features, progress features, targets.

    ``split_full`` / ``split_nonempty`` give each sprint's train/test side of Phase 2's
    time-ordered split (fixed per sprint, so checkpoints are comparable).
    """
    base = load_phase2_frame()
    state = build_checkpoint_state()
    d = add_progress_features(state, base.drop(columns=[c for c in ("end_date", "complete_date") if c in base]))
    for name in ("full", "nonempty"):
        tr, te = time_ordered_split(population(base, name))
        side = pd.concat([pd.Series("train", index=tr["sprint_id"]), pd.Series("test", index=te["sprint_id"])])
        d[f"split_{name}"] = d["sprint_id"].map(side)
    return d.sort_values(["checkpoint", "project", "start_date", "sprint_id"]).reset_index(drop=True)
