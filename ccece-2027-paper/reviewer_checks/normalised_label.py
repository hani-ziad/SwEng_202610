"""Does the 'count wins' result survive a size-normalised spillover label?
Label: carry share (fraction of committed issues unfinished at planned end) >= threshold.
Time split, nonempty sprints. Models as in the paper (strict features + project one-hot).
Usage: python normalised_label.py <phase2_modeling_frame.csv> <out.csv>
"""
import sys
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score as auc
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

STRICT = ["committed_issue_count", "committed_story_points", "sprint_length_days", "n_concurrent_sprints",
          "hist_avg_completion_ratio_3", "hist_last_carryover", "hist_avg_delivered_sp_3", "hist_n_available",
          "hist_avg_completed_issues_3", "hist_avg_committed_issues_3", "scope_to_capacity",
          "hist_avg_late_days_3", "hist_late_rate_3"]
SCOPE = ["committed_issue_count", "committed_story_points", "sprint_length_days"]


def pipe(est, num, proj):
    t = [("num", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), num)]
    if proj:
        t.append(("cat", OneHotEncoder(handle_unknown="ignore"), ["project"]))
    return Pipeline([("p", ColumnTransformer(t)), ("m", est)])


d = pd.read_csv(sys.argv[1], parse_dates=["start_date"])
ne = d[d.nonempty].sort_values(["project", "start_date", "sprint_id"])
ne["rk"] = ne.groupby("project").cumcount()
ne["np_"] = ne.groupby("project").project.transform("size")
tr, te = ne[ne.rk < (ne.np_ * 0.7).astype(int)], ne[ne.rk >= (ne.np_ * 0.7).astype(int)]
assert len(te) == 511
rows = []
for th in [0.25, 0.5]:
    ytr, yte = (tr.carry_ratio >= th).astype(int), (te.carry_ratio >= th).astype(int)
    r = {"threshold": th, "prevalence_test": round(yte.mean(), 3), "neg_test": int((yte == 0).sum()),
         "auc_committed_count": round(auc(yte, te.committed_issue_count), 3),
         "auc_project_rate": round(auc(yte, te.project.map(ytr.groupby(tr.project).mean())), 3)}
    models = {
        "scope_only_lr": pipe(LogisticRegression(max_iter=1000, class_weight="balanced"), SCOPE, False),
        "lr_strict": pipe(LogisticRegression(max_iter=1000, class_weight="balanced"), STRICT, True),
        "rf_strict": pipe(RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=3,
                                                 class_weight="balanced", random_state=42, n_jobs=-1), STRICT, True),
        "gb_strict": pipe(HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200,
                                                         random_state=42), STRICT, True),
    }
    for name, m in models.items():
        cols = (SCOPE if name == "scope_only_lr" else STRICT + ["project"])
        r["auc_" + name] = round(auc(yte, m.fit(tr[cols], ytr).predict_proba(te[cols])[:, 1]), 3)
    rows.append(r)
out = pd.DataFrame(rows)
print(out.T.to_string())
out.to_csv(sys.argv[2], index=False)
