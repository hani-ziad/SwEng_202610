"""
Label construction for "sprint at risk" prediction.

Implements the two label variants defined in the Week 8 proposal
(docs reference: Sprint Risk Prediction - Week 8 Proposal):

  - delay:      sprint closed more than 1 day after its planned end date
  - spillover:  fewer than 80% of committed issues were completed
                (equivalently: at least one issue was carried over)

`at_risk` is the OR of both, matching the proposal's definition.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DELAY_THRESHOLD_DAYS = 1
SPILLOVER_COMPLETION_THRESHOLD = 0.8


def add_labels(sprints: pd.DataFrame) -> pd.DataFrame:
    df = sprints.copy()

    delay_days = (df["complete_date"] - df["end_date"]).dt.days
    df["label_delay"] = (delay_days > DELAY_THRESHOLD_DAYS).astype(int)

    safe_denominator = df["committed_issue_count"].replace(0, np.nan)
    completion_ratio = df["completed_issue_count"] / safe_denominator
    df["completion_ratio"] = completion_ratio
    df["label_spillover"] = (
        (df["not_completed_count"] > 0)
        | (completion_ratio < SPILLOVER_COMPLETION_THRESHOLD).fillna(False)
    ).astype(int)

    df["at_risk"] = ((df["label_delay"] == 1) | (df["label_spillover"] == 1)).astype(int)
    return df


if __name__ == "__main__":
    from src.data.load_koralage import load_dataset

    sprints, _ = load_dataset()
    labeled = add_labels(sprints)
    print(labeled[["project", "at_risk", "label_delay", "label_spillover"]].groupby("project").mean())
    print("\nOverall at-risk rate:", labeled["at_risk"].mean().round(3))
