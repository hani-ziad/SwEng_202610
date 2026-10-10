"""Synthetic check of the checkpoint reconstruction (membership, additions, removals, time travel)."""
import pandas as pd

import phase3.data as pd3


def _run(sprints, issues, changes):
    return pd3.build_checkpoint_state(
        checkpoints=(0.5, 1.0), sprints=sprints, issues=issues, change_log=changes,
        cohort_ids=sprints["sprint_id"]).set_index("checkpoint")


def _sprints():
    t = pd.Timestamp
    return pd.DataFrame([{
        "raw_sprint_id": 1, "jira_sprint_id": 101, "sprint_id": "P__1", "sprint_name": "S1", "project": "P",
        "start_date": t("2020-01-01"), "end_date": t("2020-01-11"), "complete_date": t("2020-01-12"),
    }])


def _issue(i, resolution, sprint=1, sp=2.0):
    t = pd.Timestamp
    return {"raw_issue_id": i, "issue_key": f"P-{i}", "project": "P", "raw_sprint_id": sprint,
            "creation_date": t("2019-12-01"), "resolution_date": resolution, "story_point": sp}


_COUNTER = [0]


def _ch(i, field, frm, to, when):
    _COUNTER[0] += 1
    return {"change_id": _COUNTER[0], "raw_issue_id": i, "field": field, "from_value": frm,
            "to_value": to, "from_string": None, "to_string": None, "change_date": pd.Timestamp(when)}


def test_done_added_and_removed_at_halfway():
    t = pd.Timestamp
    issues = pd.DataFrame([
        _issue(1, t("2020-01-03")),                      # committed, done before the halfway point (01-06)
        _issue(2, t("2020-01-09")),                      # committed, done later
        _issue(3, pd.NaT, sprint=None),                  # committed, removed on 01-04
        _issue(4, t("2020-01-05")),                      # added on 01-03, done before halfway
    ])
    changes = pd.DataFrame([
        _ch(3, "Sprint", "101", None, "2020-01-04"),
        _ch(4, "Sprint", None, "101", "2020-01-03"),
    ])
    s = _run(_sprints(), issues, changes)
    half, end = s.loc[0.5], s.loc[1.0]
    assert half["committed_n"] == 3 and end["committed_n"] == 3        # issue 4 was not committed at start
    assert half["done_n"] == 1 and end["done_n"] == 2
    assert half["removed_n"] == 1
    assert half["added_n"] == 1 and half["added_done_n"] == 1


def test_story_point_change_is_visible_only_after_it_happens():
    t = pd.Timestamp
    issues = pd.DataFrame([_issue(1, pd.NaT, sp=8.0)])
    changes = pd.DataFrame([_ch(1, "Story Points", "3", "8", "2020-01-08")])   # after halfway (01-06)
    s = _run(_sprints(), issues, changes)
    assert s.loc[0.5, "committed_sp"] == 3.0 and s.loc[0.5, "sp_now"] == 3.0   # not yet visible at halfway
    assert s.loc[1.0, "committed_sp"] == 3.0 and s.loc[1.0, "sp_now"] == 8.0   # visible at planned end
