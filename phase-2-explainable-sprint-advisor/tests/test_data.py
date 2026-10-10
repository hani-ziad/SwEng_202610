import numpy as np
import pandas as pd

from phase2.data import (
    FEATURE_SETS,
    STRICT_NUMERIC,
    TARGET_COLUMNS,
    add_strict_history,
    load_phase2_frame,
    population,
    time_ordered_split,
)

_FRAME = None


def _frame():
    global _FRAME
    if _FRAME is None:
        _FRAME = load_phase2_frame()
    return _FRAME


def _synthetic():
    """One project, three sprints. S1 and S2 overlap; S3 starts after S1 closed but before S2 closed."""
    t = pd.Timestamp
    rows = [
        # id, start, planned end, completed at, committed, completed issues
        ("S1", t("2020-01-01"), t("2020-01-10"), t("2020-01-10"), 10, 8),
        ("S2", t("2020-01-05"), t("2020-01-20"), t("2020-01-25"), 20, 10),
        ("S3", t("2020-01-15"), t("2020-01-30"), t("2020-01-30"), 5, 5),
        ("S4", t("2020-02-10"), t("2020-02-20"), t("2020-02-20"), 6, 3),
    ]
    df = pd.DataFrame(rows, columns=["sprint_id", "start_date", "end_date", "complete_date",
                                     "committed_issue_count", "completed_issue_count"])
    df["project"] = "P"
    df["not_completed_count"] = df["committed_issue_count"] - df["completed_issue_count"]
    df["delivered_story_points"] = df["completed_issue_count"] * 2.0
    late = (df["complete_date"] - df["end_date"]).dt.total_seconds() / 86400
    df["late_days_capped"] = late.clip(0, 30)
    df["label_delay"] = (late > 1).astype(int)
    return add_strict_history(df).set_index("sprint_id")


def test_strict_history_uses_only_sprints_already_closed_at_start():
    f = _synthetic()
    # S3 starts 2020-01-15: S1 closed (01-10) but S2 closes 01-25 -> only S1 is known.
    assert f.loc["S3", "hist_avg_completed_issues_3"] == 8.0
    assert f.loc["S3", "hist_n_available"] == 1
    # S1 has no earlier sprint; S2 starts 01-05 before S1 closed -> nothing known either.
    assert pd.isna(f.loc["S1", "hist_avg_completed_issues_3"])
    assert pd.isna(f.loc["S2", "hist_avg_completed_issues_3"])
    # S4 sees S1, S2, S3 (all closed): average of 8, 10, 5
    assert abs(f.loc["S4", "hist_avg_completed_issues_3"] - (8 + 10 + 5) / 3) < 1e-12


def test_strict_history_never_uses_the_sprint_itself_or_the_future():
    f = _synthetic()
    assert f.loc["S4", "hist_last_carryover"] == 0.0           # most recently closed = S3 (5/5 done)
    assert f.loc["S3", "hist_late_rate_3"] == 0.0              # only S1 known, closed on time


def test_concurrent_sprint_count_uses_planned_end_dates():
    f = _synthetic()
    assert f.loc["S1", "n_concurrent_sprints"] == 0
    assert f.loc["S2", "n_concurrent_sprints"] == 1            # S1 still running at 01-05
    assert f.loc["S3", "n_concurrent_sprints"] == 1            # S2 running, S1 already ended
    assert f.loc["S4", "n_concurrent_sprints"] == 0


def test_cohort_matches_phase1():
    d = _frame()
    assert len(d) == 2602
    assert int(d["nonempty"].sum()) == 1691
    assert d["project"].nunique() == 8


def test_no_target_or_outcome_in_features():
    forbidden = set(TARGET_COLUMNS) | {
        "not_completed_count", "completed_issue_count", "late_days",
        "completion_ratio", "delivered_story_points", "complete_date",
    }
    for numeric in FEATURE_SETS.values():
        assert not forbidden.intersection(numeric)


def test_targets_are_consistent():
    d = _frame()
    ne = population(d, "nonempty")
    assert ne["carry_ratio"].between(0, 1).all()
    assert np.allclose(ne["not_completed_count"] / ne["committed_issue_count"], ne["carry_ratio"])
    assert (ne["completed_issue_count"] + ne["not_completed_count"] == ne["committed_issue_count"]).all()
    assert d["late_days_capped"].between(0, 30).all()
    assert ((d["late_days"] > 1).astype(int) == d["label_delay"]).all()


def test_time_split_sizes_match_phase1():
    full, ne = population(_frame(), "full"), population(_frame(), "nonempty")
    assert tuple(map(len, time_ordered_split(full))) == (1816, 786)
    assert tuple(map(len, time_ordered_split(ne))) == (1180, 511)
    tr, te = time_ordered_split(ne)
    for _, g in pd.concat([tr.assign(s="tr"), te.assign(s="te")]).groupby("project"):
        assert g[g.s == "tr"]["start_date"].max() <= g[g.s == "te"]["start_date"].min()


def test_legacy_features_complete_and_strict_mostly_present():
    d = _frame()
    assert not d[FEATURE_SETS["legacy"]].isna().any().any()
    assert d[STRICT_NUMERIC].isna().mean().max() < 0.05
