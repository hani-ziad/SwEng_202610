"""Sprint advisor: forecasts for a planned sprint, for both outcomes, plus what-if.

For a planned sprint (features known at planning time) it forecasts

  spillover side     issues completed by planned end: median and 80% interval
                     (10th-90th percentile), expected carryover, P(any spillover)
  late-closure side  P(closed more than 1 day late), median and 90th-percentile
                     days late (capped at 30)

and re-forecasts all of them for any candidate scope (``what_if``).

IMPORTANT (see the Phase 2 write-up): the what-if table is a statement about
what the *fitted model* expects, learned from observational history. It is not
a causal claim that cutting scope will make a real team finish more or close
on time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from phase2.data import STRICT_NUMERIC
from phase2.models import SEED

DONE_QUANTILES = (0.1, 0.5, 0.9)
LATE_QUANTILES = (0.5, 0.9)


def _pipeline(estimator, with_project: bool) -> Pipeline:
    # HistGradientBoosting handles missing history natively, no imputation needed.
    transformers = [("num", "passthrough", STRICT_NUMERIC)]
    if with_project:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["project"]))
    return Pipeline([("prep", ColumnTransformer(transformers)), ("model", estimator)])


def _hgb_reg(q: float) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="quantile", quantile=q, max_depth=3, learning_rate=0.05, max_iter=200, random_state=SEED
    )


def _hgb_clf() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, random_state=SEED)


def _monotone(arrays, upper=None):
    """Sort crossing quantiles; keep forecasts in [0, upper]."""
    stack = np.sort(np.vstack(arrays), axis=0)
    hi = np.inf if upper is None else np.asarray(upper, dtype=float)
    return np.clip(stack, 0.0, hi)


class SprintAdvisor:
    def __init__(self, with_project: bool = True):
        self.with_project = with_project
        self.cols = STRICT_NUMERIC + (["project"] if with_project else [])
        self.done = {q: _pipeline(_hgb_reg(q), with_project) for q in DONE_QUANTILES}
        self.late = {q: _pipeline(_hgb_reg(q), with_project) for q in LATE_QUANTILES}
        self.p_spill = _pipeline(_hgb_clf(), with_project)
        self.p_late = _pipeline(_hgb_clf(), with_project)

    def fit(self, train: pd.DataFrame) -> "SprintAdvisor":
        """``train``: the full cohort frame; the spillover side uses its nonempty sprints."""
        ne = train[train["nonempty"]]
        for m in self.done.values():
            m.fit(ne[self.cols], ne["completed_issue_count"])
        self.p_spill.fit(ne[self.cols], ne["label_spillover"])
        for q, m in self.late.items():
            m.fit(train[self.cols], train["late_days_capped"])
        self.p_late.fit(train[self.cols], train["label_delay"])
        return self

    # -- spillover side ------------------------------------------------------
    def forecast_completed(self, frame: pd.DataFrame) -> pd.DataFrame:
        raw = [self.done[q].predict(frame[self.cols]) for q in DONE_QUANTILES]
        lo, mid, hi = _monotone(raw, upper=frame["committed_issue_count"])
        return pd.DataFrame({"q10": lo, "q50": mid, "q90": hi}, index=frame.index)

    def p_any_spillover(self, frame: pd.DataFrame) -> np.ndarray:
        return self.p_spill.predict_proba(frame[self.cols])[:, 1]

    # -- late-closure side ----------------------------------------------------
    def forecast_late(self, frame: pd.DataFrame) -> pd.DataFrame:
        raw = [self.late[q].predict(frame[self.cols]) for q in LATE_QUANTILES]
        q50, q90 = _monotone(raw, upper=30.0)
        return pd.DataFrame(
            {"p_late": self.p_late.predict_proba(frame[self.cols])[:, 1], "late_q50": q50, "late_q90": q90},
            index=frame.index,
        )

    # -- what-if --------------------------------------------------------------
    def what_if(self, sprint: pd.Series, scopes) -> pd.DataFrame:
        """Forecast the same sprint under different planned issue counts.

        Story points scale in proportion to the issue count (the sprint's own
        points-per-issue is kept); the team's history and the sprint's length
        and parallel-sprint count are held fixed.
        """
        base_scope = float(sprint["committed_issue_count"])
        sp_per_issue = float(sprint["committed_story_points"]) / base_scope
        rows = []
        for s in scopes:
            r = sprint.copy()
            r["committed_issue_count"] = float(s)
            r["committed_story_points"] = float(s) * sp_per_issue
            cap = r["hist_avg_completed_issues_3"]
            r["scope_to_capacity"] = float(s) / max(float(cap), 1.0) if pd.notna(cap) else np.nan
            rows.append(r)
        frame = pd.DataFrame(rows).reset_index(drop=True)
        done = self.forecast_completed(frame)
        late = self.forecast_late(frame)
        out = pd.DataFrame({
            "planned_issues": [float(s) for s in scopes],
            "expected_done_q50": done["q50"].round(1),
            "done_range_80pct_low": done["q10"].round(1),
            "done_range_80pct_high": done["q90"].round(1),
        })
        out["expected_carryover_q50"] = (out["planned_issues"] - out["expected_done_q50"]).round(1)
        out["expected_carryover_ratio"] = (out["expected_carryover_q50"] / out["planned_issues"]).round(3)
        out["p_any_spillover"] = self.p_any_spillover(frame).round(3)
        out["p_late_closure"] = late["p_late"].round(3)
        out["late_days_q50"] = late["late_q50"].round(1)
        out["late_days_q90"] = late["late_q90"].round(1)
        return out
