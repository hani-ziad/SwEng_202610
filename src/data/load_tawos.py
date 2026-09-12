"""
Loader for the *real* TAWOS dataset (Tawosi et al., MSR 2022), the primary
dataset named in the Week 8 proposal. Not exercised by this checkpoint's
results -- the sandbox this repo was built in could not reach the host
that serves TAWOS (see docs/data_access_and_versioning.md for the exact
failure and how to obtain the data yourself). This loader is provided so
that dropping the exported CSVs into data/raw/tawos/ is the only step
needed to switch the whole pipeline (labels.py, features.py,
baselines.py, scripts/run_baseline.py) over to the full-scale dataset --
they only depend on the canonical columns this loader produces, which
are the same ones load_koralage.py produces.

Expects two CSVs in `raw_dir`, produced by the SQL in
scripts/export_tawos_subset.sql against the official MySQL dump:
  - tawos_sprint.csv
  - tawos_issue.csv
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_sprints(raw_dir: Path) -> pd.DataFrame:
    path = Path(raw_dir) / "tawos_sprint.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. See docs/data_access_and_versioning.md for how to "
            "obtain TAWOS and export this file with scripts/export_tawos_subset.sql."
        )
    df = pd.read_csv(path)
    df = df.rename(
        columns={
            "ID": "raw_sprint_id",
            "Name": "sprint_name",
            "State": "sprint_state",
            "Start_Date": "start_date",
            "End_Date": "end_date",
            "Complete_Date": "complete_date",
            "Project_Name": "project",
        }
    )
    df["sprint_id"] = df["project"].astype(str) + "__" + df["raw_sprint_id"].astype(str)
    for c in ["start_date", "end_date", "complete_date"]:
        df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def load_issues(raw_dir: Path) -> pd.DataFrame:
    path = Path(raw_dir) / "tawos_issue.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. See docs/data_access_and_versioning.md for how to "
            "obtain TAWOS and export this file with scripts/export_tawos_subset.sql."
        )
    df = pd.read_csv(path)
    df = df.rename(
        columns={
            "Issue_Key": "issue_key",
            "Type": "issue_type",
            "Status": "status",
            "Story_Point": "story_point",
            "Priority": "priority",
            "Sprint_ID": "raw_sprint_id",
            "Project_Name": "project",
            "Resolution_Date": "resolution_date",
            "Assignee_ID": "assignee",
        }
    )
    df["sprint_id"] = df["project"].astype(str) + "__" + df["raw_sprint_id"].astype(str)
    df["resolution_date"] = pd.to_datetime(df.get("resolution_date"), errors="coerce")
    return df


def derive_sprint_aggregates(sprints: pd.DataFrame, issues: pd.DataFrame) -> pd.DataFrame:
    """
    TAWOS's Issue table gives per-issue Resolution_Date and Story_Point, so
    (unlike the Koralage CSVs, which only ship pre-aggregated outcome
    buckets) the same canonical sprint-level columns can be computed
    directly and cleanly here: "completed" is simply "Resolved by the
    sprint's End_Date", so committed scope and delivered scope are both
    unambiguous, planning-time-safe quantities. Run this once after
    load_sprints/load_issues, before add_labels/add_features.
    """
    merged = issues.merge(sprints[["sprint_id", "end_date"]], on="sprint_id", how="inner")
    merged["is_completed"] = (
        merged["resolution_date"].notna() & (merged["resolution_date"] <= merged["end_date"])
    )

    agg = merged.groupby("sprint_id").agg(
        committed_issue_count=("issue_key", "count"),
        committed_story_points=("story_point", "sum"),
        completed_issue_count=("is_completed", "sum"),
        num_developers=("assignee", "nunique") if "assignee" in merged.columns else ("issue_key", "count"),
    )
    agg["not_completed_count"] = agg["committed_issue_count"] - agg["completed_issue_count"]
    agg["delivered_story_points"] = (
        merged[merged["is_completed"]].groupby("sprint_id")["story_point"].sum()
    ).reindex(agg.index).fillna(0)

    return sprints.merge(agg.reset_index(), on="sprint_id", how="left")


def load_dataset(raw_dir: Path | str = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    if raw_dir is None:
        raw_dir = Path(__file__).resolve().parents[2] / "data" / "raw" / "tawos"
    raw_dir = Path(raw_dir)
    return load_sprints(raw_dir), load_issues(raw_dir)


if __name__ == "__main__":
    sprints, issues = load_dataset()
    sprints = derive_sprint_aggregates(sprints, issues)
    print(f"Loaded {len(sprints)} TAWOS sprints across {sprints['project'].nunique()} projects.")

