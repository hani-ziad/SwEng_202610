import numpy as np
import pandas as pd

from phase2.advisor import SprintAdvisor
from phase2.data import load_phase2_frame, time_ordered_split

_CACHE = {}


def _fitted():
    if not _CACHE:
        d = load_phase2_frame()
        train, test = time_ordered_split(d)
        _CACHE["adv"] = SprintAdvisor().fit(train)
        _CACHE["test"] = test
    return _CACHE["adv"], _CACHE["test"]


def test_done_forecast_is_ordered_and_within_scope():
    adv, test = _fitted()
    ne = test[test["nonempty"]]
    fc = adv.forecast_completed(ne)
    assert (fc["q10"] <= fc["q50"]).all() and (fc["q50"] <= fc["q90"]).all()
    assert (fc["q10"] >= 0).all()
    assert (fc["q90"] <= ne["committed_issue_count"] + 1e-9).all()


def test_late_forecast_is_valid():
    adv, test = _fitted()
    fc = adv.forecast_late(test)
    assert fc["p_late"].between(0, 1).all()
    assert (fc["late_q50"] <= fc["late_q90"]).all()
    assert fc["late_q50"].between(0, 30).all() and fc["late_q90"].between(0, 30).all()


def test_what_if_does_not_mutate_input_and_has_both_outcomes():
    adv, test = _fitted()
    sprint = test[test["nonempty"]].iloc[0]
    before = sprint.copy()
    table = adv.what_if(sprint, [1, 3, 6])
    pd.testing.assert_series_equal(sprint, before)
    assert list(table["planned_issues"]) == [1.0, 3.0, 6.0]
    for col in ("expected_done_q50", "p_any_spillover", "p_late_closure", "late_days_q90"):
        assert col in table.columns and table[col].notna().all()
    assert (table["expected_done_q50"] <= table["planned_issues"] + 1e-9).all()


def test_what_if_at_actual_scope_matches_direct_forecast():
    adv, test = _fitted()
    ne = test[test["nonempty"]]
    sprint = ne.iloc[3]
    table = adv.what_if(sprint, [float(sprint["committed_issue_count"])])
    direct = adv.forecast_completed(ne.iloc[[3]])
    assert abs(table.loc[0, "expected_done_q50"] - round(float(direct["q50"].iloc[0]), 1)) < 1e-9
