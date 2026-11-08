"""Phase 4, fix 2: two corrected labels (added 8 Nov 2026).

Why. The sanity checks in experiment 4 showed that both Phase 1 labels measure
something other than what their names say:

  * Spillover ("at least one committed issue not resolved by the planned end")
    is positive for 93% of sprints with committed work and 99.7% of sprints
    with more than 10 issues. The raw issue count alone scores AUC 0.88. It is
    a sprint-size detector.
  * Late closure ("closed more than 1 day after the planned end") counts a
    sprint that ends on Friday and is closed on Monday as late. A third of all
    late sprints are like that. The Titanium SDK drops from 60% late to 19%.
  * In Atlassian Confluence Cloud, 68% of Closed issues have no resolution
    date, and the export has no status history to recover when they closed.
    Every "finished by the planned end" count is wrong there.

The corrected labels, fixed before any model was run on them:

  label_late         closed after the end of the next working day (Mon-Fri)
                     after the planned end. Dates are UTC; TAWOS has no time
                     zones. A flat 3-day grace is kept as a sensitivity check.
  label_heavy_carry  half or more of the committed issues not resolved by the
                     planned end (carry share >= 0.5). Nonempty sprints only,
                     Confluence Cloud excluded. 0.2 and 0.3 are sensitivity
                     checks.

History features are rebuilt with the new late label, so "past lateness"
means the same thing as the target. In Confluence Cloud the history features
that count finished issues are set to missing (imputed like any other gap),
because they read the broken field.

Phases 1-3 are not changed. This module only calls their functions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

import phase4  # noqa: F401  (puts Phases 1-3 on the path)
import phase2.data as p2d
from src.data.load_tawos import load_reconstructed_sprints
from src.features import add_features, build_model_frame
from src.labels import add_labels

CARRY_THRESHOLD = 0.5
CARRY_SENSITIVITY = (0.2, 0.3)
FLAT_GRACE_DAYS = 3.0
BROKEN_RESOLUTION = {"Atlassian Confluence Cloud"}
COMPLETION_HISTORY = [
    "hist_avg_completion_ratio_3", "hist_last_carryover", "hist_avg_delivered_sp_3",
    "hist_avg_completed_issues_3", "scope_to_capacity",
]


def next_working_day_deadline(end: pd.Series) -> pd.Series:
    """End of the first Mon-Fri day after the planned end date."""
    return end.dt.normalize() + BDay(1) + pd.Timedelta(days=1)


def late_next_working_day(end: pd.Series, complete: pd.Series) -> pd.Series:
    return (complete > next_working_day_deadline(end)).astype(int)


def late_flat(end: pd.Series, complete: pd.Series, days: float = FLAT_GRACE_DAYS) -> pd.Series:
    return ((complete - end).dt.total_seconds() / 86400.0 > days).astype(int)


def load_relabelled_frame(raw_dir=None) -> pd.DataFrame:
    """Same 2,602-sprint cohort and strict features as Phase 2, with the corrected labels."""
    sprints = load_reconstructed_sprints(raw_dir)            # None = the TAWOS export of Phase 1
    f = add_features(add_labels(sprints))
    f["label_late_1d"] = f["label_delay"]                      # Phase 1 label, kept for comparison
    f["label_late_3d"] = late_flat(f["end_date"], f["complete_date"])
    f["label_late"] = late_next_working_day(f["end_date"], f["complete_date"])
    f["label_delay"] = f["label_late"]                         # so past-lateness history uses the new label
    late_days = (f["complete_date"] - f["end_date"]).dt.total_seconds() / 86400.0
    f["late_days"] = late_days
    f["late_days_capped"] = late_days.clip(lower=0.0, upper=p2d.LATE_CAP_DAYS)
    f = p2d.add_strict_history(f)

    cohort = build_model_frame(f, "label_spillover")[["sprint_id"]]
    d = f.merge(cohort, on="sprint_id", how="inner")
    d["nonempty"] = d["committed_issue_count"] > 0
    d["carry_ratio"] = np.where(d["nonempty"],
                                d["not_completed_count"] / d["committed_issue_count"].where(d["nonempty"], 1), np.nan)
    broken = d["project"].isin(BROKEN_RESOLUTION)
    d["resolution_ok"] = ~broken
    d["carry_valid"] = d["nonempty"] & ~broken
    d["label_heavy_carry"] = (d["carry_ratio"] >= CARRY_THRESHOLD).astype(int)
    for t in CARRY_SENSITIVITY:
        d[f"label_carry_{int(t * 100)}"] = (d["carry_ratio"] >= t).astype(int)
    d.loc[broken, COMPLETION_HISTORY] = np.nan

    keep = (["sprint_id", "start_date", "end_date", "complete_date", "project", "nonempty", "resolution_ok",
             "carry_valid", "late_days", "carry_ratio", "committed_issue_count", "not_completed_count",
             "label_late", "label_late_3d", "label_late_1d", "label_spillover", "label_heavy_carry"]
            + [f"label_carry_{int(t * 100)}" for t in CARRY_SENSITIVITY] + p2d.STRICT_NUMERIC)
    return d[list(dict.fromkeys(keep))].sort_values(["project", "start_date", "sprint_id"]).reset_index(drop=True)


def population(d: pd.DataFrame, task: str) -> pd.DataFrame:
    """'late' = all cohort sprints; 'carry' = nonempty sprints with usable resolution dates."""
    if task == "late":
        return d.reset_index(drop=True)
    if task == "carry":
        return d[d["carry_valid"]].reset_index(drop=True)
    raise ValueError(task)
