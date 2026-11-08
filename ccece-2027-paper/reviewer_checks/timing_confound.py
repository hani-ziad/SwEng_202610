"""Does the 'strict timing' gain come from the timing fix or from the extra features?"""
import sys, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score as auc
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
d = pd.read_csv(sys.argv[1])
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
def mk(name, cols):
    pre = ColumnTransformer([("n", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), cols)])
    est = (LogisticRegression(max_iter=1000, class_weight="balanced") if name == "LR" else
           RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=3, class_weight="balanced", random_state=42, n_jobs=-1))
    return Pipeline([("p", pre), ("m", est)])
rows = []
for sname, cols in SETS.items():
    for m in ["LR", "RF"]:
        ys, ps = [], []
        for p in d.project.unique():
            tr, te = d[d.project != p], d[d.project == p]
            ps.append(mk(m, cols).fit(tr[cols], tr.label_delay).predict_proba(te[cols])[:, 1]); ys.append(te.label_delay)
        rows.append({"feature_set": sname, "model": m, "lopo_pooled_auc_late": round(auc(np.concatenate(ys), np.concatenate(ps)), 3)})
out = pd.DataFrame(rows); print(out.to_string(index=False)); out.to_csv(sys.argv[2], index=False)
