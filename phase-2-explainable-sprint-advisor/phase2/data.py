"""Phase 2 modeling frame: both outcomes, and history that is really known.

Starts from the Phase 1 pipeline (Change_Log-reconstructed planning-time scope,
CLOSED sprints only, explicit cold-start removal => the 2,602-sprint cohort and
its 1,691 nonempty sprints) and adds:

* both outcome families
    - spillover side: ``carry_ratio`` (share of committed issues not done by the
      planned end), ``completed_issue_count``, ``label_spillover``
    - late-closure side: ``late_days_capped`` (days closed after the planned
      end, 0 if on time or early, capped at 30) and ``label_delay``
      (closed more than 1 day late, the Phase 1 label)
* STRICT history features. Projects run several sprints at the same time
  (several boards/teams), so the "previous sprint" in start-date order has often
  not finished when the next one starts. Phase 1's history features ignore that.
  The strict features use only sprints of the same project whose outcome was
  already known when the sprint starts, i.e. max(planned end, closure date) is
  not later than this sprint's start.
* ``n_concurrent_sprints``: how many other sprints of the project were already
  running at planning time (planned end dates are known in advance).

Phase 1's own feature set is kept as the ``legacy`` feature set so the effect of
the timing issue can be measured (see experiments/run_models.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import phase2  # noqa: F401  (makes the Phase 1 ``src`` package importable)
from src.data.load_tawos import load_reconstructed_sprints
from src.features import (
    HISTORICAL_FEATURES,
    PLANNING_TIME_FEATURES,
    add_features,
    build_model_frame,
)
from src.labels import add_labels

LATE_CAP_DAYS = 30.0
HISTORY_WINDOW = 3

# ---- strict feature set ----------------------------------------------------
PLAN_FEATURES = [
    "committed_issue_count",
    "committed_story_points",
    "sprint_length_days",
    "n_concurrent_sprints",
]
HIST_FEATURES = [
    "hist_avg_completion_ratio_3",
    "hist_last_carryover",
    "hist_avg_delivered_sp_3",
    "hist_n_available",
]
CAPACITY_FEATURES = [
    "hist_avg_completed_issues_3",
    "hist_avg_committed_issues_3",
    "scope_to_capacity",
]
LATE_HIST_FEATURES = ["hist_avg_late_days_3", "hist_late_rate_3"]
STRICT_NUMERIC = PLAN_FEATURES + HIST_FEATURES + CAPACITY_FEATURES + LATE_HIST_FEATURES

# ---- Phase 1's feature set (for the sensitivity comparison) -----------------
LEGACY_NUMERIC = list(PLANNING_TIME_FEATURES) + list(HISTORICAL_FEATURES)

FEATURE_SETS = {"strict": STRICT_NUMERIC, "legacy": LEGACY_NUMERIC}

GROUPS = {
    "strict": {
        "scope": ["committed_issue_count", "committed_story_points"],
        "sprint length": ["sprint_length_days"],
        "parallel sprints": ["n_concurrent_sprints"],
        "recent history": HIST_FEATURES,
        "capacity": CAPACITY_FEATURES,
        "past lateness": LATE_HIST_FEATURES,
        "project": ["project"],
    },
    "legacy": {
        "scope": ["committed_issue_count", "committed_story_points"],
        "sprint length": ["sprint_length_days"],
        "recent history": list(HISTORICAL_FEATURES),
        "project": ["project"],
    },
}

TARGET_COLUMNS = [
    "carry_ratio",
    "completed_issue_count",
    "label_spillover",
    "late_days_capped",
    "label_delay",
]


def add_strict_history(df: pd.DataFrame) -> pd.DataFrame:
    """Add strict history + concurrency columns (see module docstring).

    ``df`` must contain every closed sprint of each project (not just the
    modeled ones) so that earlier sprints can serve as history.
    """
    df = df.sort_values(["project", "start_date", "sprint_id"]).reset_index(drop=True)
    avail = df[["end_date", "complete_date"]].max(axis=1).to_numpy()
    start = df["start_date"].to_numpy()
    end = df["end_date"].to_numpy()
    committed = df["committed_issue_count"].to_numpy(float)
    completed = df["completed_issue_count"].to_numpy(float)
    carry = df["not_completed_count"].to_numpy(float)
    delivered = df["delivered_story_points"].to_numpy(float)
    late = df["late_days_capped"].to_numpy(float)
    late_flag = df["label_delay"].to_numpy(float)

    n = len(df)
    cols = {c: np.full(n, np.nan) for c in (
        "hist_avg_completion_ratio_3", "hist_last_carryover", "hist_avg_delivered_sp_3",
        "hist_avg_completed_issues_3", "hist_avg_committed_issues_3",
        "hist_avg_late_days_3", "hist_late_rate_3",
    )}
    n_avail = np.zeros(n)
    n_conc = np.zeros(n)

    for _, idx in df.groupby("project").indices.items():
        idx = np.asarray(idx)
        for pos in idx:
            others = idx[idx != pos]
            # sprints running at planning time (planned end is known in advance)
            n_conc[pos] = float(((start[others] < start[pos]) & (end[others] > start[pos])).sum())
            # sprints whose outcome was already known when this one starts
            known = others[avail[others] <= start[pos]]
            if len(known) == 0:
                continue
            known = known[np.argsort(avail[known])[::-1]]          # most recent first
            recent_late = known[:HISTORY_WINDOW]
            cols["hist_avg_late_days_3"][pos] = late[recent_late].mean()
            cols["hist_late_rate_3"][pos] = late_flag[recent_late].mean()
            nonempty = known[committed[known] > 0][:HISTORY_WINDOW]
            n_avail[pos] = len(nonempty)
            if len(nonempty):
                cols["hist_avg_completion_ratio_3"][pos] = (completed[nonempty] / committed[nonempty]).mean()
                cols["hist_last_carryover"][pos] = carry[nonempty[0]]
                cols["hist_avg_delivered_sp_3"][pos] = delivered[nonempty].mean()
                cols["hist_avg_completed_issues_3"][pos] = completed[nonempty].mean()
                cols["hist_avg_committed_issues_3"][pos] = committed[nonempty].mean()

    for name, arr in cols.items():
        df[name] = arr
    df["hist_n_available"] = n_avail
    df["n_concurrent_sprints"] = n_conc
    df["scope_to_capacity"] = df["committed_issue_count"] / df["hist_avg_completed_issues_3"].clip(lower=1.0)
    return df


def load_phase2_frame() -> pd.DataFrame:
    """The 2,602-sprint Phase 1 cohort with both outcomes and both feature sets."""
    sprints = load_reconstructed_sprints()
    f = add_features(add_labels(sprints))
    late_days = (f["complete_date"] - f["end_date"]).dt.total_seconds() / 86400.0
    f["late_days"] = late_days
    f["late_days_capped"] = late_days.clip(lower=0.0, upper=LATE_CAP_DAYS)
    f = add_strict_history(f)

    cohort = build_model_frame(f, "label_spillover")[["sprint_id"]]
    d = f.merge(cohort, on="sprint_id", how="inner")
    d["nonempty"] = d["committed_issue_count"] > 0
    d["carry_ratio"] = np.where(
        d["nonempty"], d["not_completed_count"] / d["committed_issue_count"].where(d["nonempty"], 1), np.nan
    )

    keep = (
        ["sprint_id", "start_date", "project", "nonempty", "late_days", "not_completed_count"]
        + TARGET_COLUMNS
        + STRICT_NUMERIC
        + LEGACY_NUMERIC
    )
    d = d[list(dict.fromkeys(keep))].reset_index(drop=True)
    legacy_na = d[LEGACY_NUMERIC].isna().any(axis=1)
    if legacy_na.any():
        raise ValueError(f"Unexpected missing legacy features ({int(legacy_na.sum())} rows)")
    return d


def population(d: pd.DataFrame, name: str) -> pd.DataFrame:
    """'full' = all 2,602 cohort sprints; 'nonempty' = the 1,691 with committed scope."""
    if name == "full":
        return d.reset_index(drop=True)
    if name == "nonempty":
        return d[d["nonempty"]].reset_index(drop=True)
    raise ValueError(name)


def time_ordered_split(frame: pd.DataFrame, train_fraction: float = 0.7):
    """Same per-project time-ordered split as Phase 1 (first 70% of each project)."""
    train_parts, test_parts = [], []
    for _, g in frame.groupby("project"):
        g = g.sort_values("start_date")
        cut = max(1, int(len(g) * train_fraction))
        train_parts.append(g.iloc[:cut])
        test_parts.append(g.iloc[cut:])
    return pd.concat(train_parts), pd.concat(test_parts)
