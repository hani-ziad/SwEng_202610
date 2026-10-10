import numpy as np
import pandas as pd

from phase3.data import CHECKPOINT_NUMERIC, CHECKPOINTS, PROGRESS_FEATURES, load_checkpoint_frame

_FRAME = None


def _frame():
    global _FRAME
    if _FRAME is None:
        _FRAME = load_checkpoint_frame()
    return _FRAME


def test_one_row_per_sprint_and_checkpoint():
    d = _frame()
    assert len(d) == 2602 * len(CHECKPOINTS)
    assert not d.duplicated(["sprint_id", "checkpoint"]).any()
    assert set(d["checkpoint"]) == set(CHECKPOINTS)


def test_planned_end_state_reproduces_phase1_scope_and_outcome():
    """At 100% the committed/done counts must equal Phase 1's reconstructed aggregates."""
    from src.data.load_tawos import load_reconstructed_sprints

    r = load_reconstructed_sprints().set_index("sprint_id")
    e = _frame().query("checkpoint == 1.0").set_index("sprint_id")
    assert (e["committed_n"] == r.loc[e.index, "committed_issue_count"]).all()
    assert (e["done_n"] == r.loc[e.index, "completed_issue_count"]).all()
    assert np.allclose(e["committed_sp"], r.loc[e.index, "committed_story_points"])


def test_spillover_label_is_determined_at_planned_end():
    e = _frame().query("checkpoint == 1.0 and nonempty")
    assert ((e["remaining_n"] > 0).astype(int) == e["label_spillover"]).all()
    assert np.allclose(e["done_share"], 1 - e["carry_ratio"])


def test_done_work_never_decreases_over_checkpoints():
    d = _frame().sort_values("checkpoint")
    for _, g in d.groupby("sprint_id"):
        assert (np.diff(g["done_n"].to_numpy()) >= 0).all()


def test_checkpoint_times_lie_inside_the_planned_window():
    d = _frame()
    assert (d["checkpoint_time"] >= d["start_date"]).all()
    assert (d["checkpoint_time"] <= d["end_date"]).all()
    assert (d["remaining_days"] >= 0).all()


def test_open_flag_matches_completion_date():
    d = _frame()
    assert (d["open_at_checkpoint"] == (d["complete_date"] > d["checkpoint_time"])).all()
    assert d.query("checkpoint == 0.25")["open_at_checkpoint"].mean() > 0.95


def test_no_outcome_in_features():
    forbidden = {"carry_ratio", "completed_issue_count", "label_spillover", "late_days", "late_days_capped",
                 "label_delay", "not_completed_count", "complete_date", "open_at_checkpoint"}
    assert not forbidden.intersection(CHECKPOINT_NUMERIC)
    assert set(PROGRESS_FEATURES) <= set(CHECKPOINT_NUMERIC)


def test_progress_features_are_finite_and_capped():
    d = _frame()
    assert np.isfinite(d[PROGRESS_FEATURES].to_numpy(float)).all()
    assert d["added_ratio"].max() <= 5.0 and d["removed_ratio"].max() <= 5.0
    assert d["done_share"].between(0, 1).all()


def test_split_columns_match_phase2_sizes():
    one = _frame().query("checkpoint == 0.25")
    assert one["split_full"].value_counts().to_dict() == {"train": 1816, "test": 786}
    assert one["split_nonempty"].value_counts().to_dict() == {"train": 1180, "test": 511}
