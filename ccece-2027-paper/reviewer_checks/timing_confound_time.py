"""Timing fix vs added features, on the per-project 70/30 time split (late closure).

Usage: python timing_confound_time.py <phase2_modeling_frame.csv> <out.csv>
"""
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score as auc
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SCOPE = ["committed_issue_count", "committed_story_points", "sprint_length_days"]
LEG_H = ["prior_avg_delivered_sp_3", "prior_avg_completion_ratio_3", "prior_sprint_carryover", "prior_sprint_completion_ratio"]
STR_H = ["hist_avg_completion_ratio_3", "hist_last_carryover", "hist_avg_delivered_sp_3", "hist_n_available"]
CAP = ["hist_avg_completed_issues_3", "hist_avg_committed_issues_3", "scope_to_capacity"]
LATE = ["hist_avg_late_days_3", "hist_late_rate_3"]
SETS = {
    "legacy (scope + old history)": SCOPE + LEG_H,
    "strict timing, same content": SCOPE + STR_H,
    "strict minus past lateness": SCOPE + ["n_concurrent_sprints"] + STR_H + CAP,
    "legacy + past lateness": SCOPE + LEG_H + LATE,
    "full strict": SCOPE + ["n_concurrent_sprints"] + STR_H + CAP + LATE,
}

d = pd.read_csv(sys.argv[1], parse_dates=["start_date"])
d = d.sort_values(["project", "start_date"])
d["rk"] = d.groupby("project").cumcount()
d["n"] = d.groupby("project").project.transform("size")
oh = pd.get_dummies(d.project, prefix="p").astype(float)
d = pd.concat([d, oh], axis=1)
tr, te = d[d.rk < np.floor(d.n * .7)], d[d.rk >= np.floor(d.n * .7)]
assert len(te) == 786  # same test set as the paper

rows = []
for name, cols in SETS.items():
    c = cols + list(oh.columns)
    for m in ["LR", "GB"]:
        if m == "LR":
            est = Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler()),
                            ("m", LogisticRegression(max_iter=1000, class_weight="balanced"))])
        else:
            est = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, random_state=42)
        p = est.fit(tr[c], tr.label_delay).predict_proba(te[c])[:, 1]
        rows.append({"feature_set": name, "model": m, "time_split_auc_late": round(auc(te.label_delay, p), 3)})
out = pd.DataFrame(rows)
print(out.to_string(index=False))
out.to_csv(sys.argv[2], index=False)
