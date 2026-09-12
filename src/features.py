"""
Feature engineering for sprint-risk prediction.

All features are computed to be available at (or before) each sprint's
own start_date -- i.e. no feature may depend on that sprint's own outcome
(completed_issue_count, not_completed_count, delivered_story_points,
completion_ratio, or the label columns). This is checked by
tests/test_no_leakage.py.

Two kinds of features:
  - "current-sprint planning" features: things decided at sprint planning
    time for the sprint about to run (committed scope, team size, length).
  - "historical" features: rolling statistics computed ONLY from sprints
    strictly before the current one, within the same project.
"""
from __future__ import annotations

import pandas as pd

PLANNING_TIME_FEATURES = [
    "committed_issue_count",
    "committed_story_points",
    "num_developers",
    "sprint_length_days",
]

HISTORICAL_FEATURES = [
    "prior_avg_delivered_sp_3",
    "prior_avg_completion_ratio_3",
    "prior_sprint_carryover",
    "prior_sprint_completion_ratio",
]

CATEGORICAL_FEATURES = ["project"]

ALL_FEATURES = PLANNING_TIME_FEATURES + HISTORICAL_FEATURES + CATEGORICAL_FEATURES


def add_features(labeled_sprints: pd.DataFrame) -> pd.DataFrame:
    df = labeled_sprints.sort_values(["project", "start_date"]).copy()

    grp = df.groupby("project", group_keys=False)

    # Rolling stats over the PRIOR 3 sprints only: shift(1) first so the
    # current row's own outcome never enters its own window.
    prior_delivered = grp["delivered_story_points"].shift(1)
    prior_ratio = grp["completion_ratio"].shift(1)

    df["prior_avg_delivered_sp_3"] = (
        prior_delivered.groupby(df["project"]).rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
    )
    df["prior_avg_completion_ratio_3"] = (
        prior_ratio.groupby(df["project"]).rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
    )
    df["prior_sprint_carryover"] = grp["not_completed_count"].shift(1)
    df["prior_sprint_completion_ratio"] = prior_ratio

    return df


def build_model_frame(featured_sprints: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """Return (features..., label) with rows lacking sufficient history dropped."""
    cols = ALL_FEATURES + [label_col, "project", "sprint_id", "start_date"]
    df = featured_sprints[list(dict.fromkeys(cols))].copy()
    # Require at least one prior sprint's worth of history.
    df = df.dropna(subset=["prior_sprint_completion_ratio"])
    return df.reset_index(drop=True)


if __name__ == "__main__":
    from src.data.load_koralage import load_dataset
    from src.labels import add_labels

    sprints, _ = load_dataset()
    labeled = add_labels(sprints)
    featured = add_features(labeled)
    frame = build_model_frame(featured, "label_delay")
    print(frame.head())
    print("\nRows available for modeling after dropping cold-start sprints:", len(frame))
