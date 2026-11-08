"""Small facts used in the 8 Nov rewrite. Usage: python misc_numbers.py <phase2_frame.csv> <e3_external_per_project.csv> <out.csv>"""
import sys
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score as auc
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
d = pd.read_csv(sys.argv[1], parse_dates=["start_date"])
e = pd.read_csv(sys.argv[2])
R = []
def add(k, v): R.append({"check": k, "value": v}); print(k, v)
add("late_days_min", round(d.late_days.min(), 1)); add("late_days_max", round(d.late_days.max(), 1))
add("sprint_length_min_days", round(d.sprint_length_days.min(), 3)); add("sprint_length_max_days", round(d.sprint_length_days.max(), 1))
add("late_days_capped_min", d.late_days_capped.min())
def split(df):
    df = df.sort_values(["project", "start_date", "sprint_id"]).copy()
    df["rk"] = df.groupby("project").cumcount(); df["np_"] = df.groupby("project").project.transform("size")
    cut = (df.np_ * 0.7).astype(int); return df[df.rk < cut], df[df.rk >= cut]
tr, te = split(d)
add("train_late_rate", round(tr.label_delay.mean(), 3)); add("test_late_rate", round(te.label_delay.mean(), 3))
trn, ten = split(d[d.nonempty])
neg = ten[ten.label_spillover == 0].groupby("project").size()
add("spill_test_negatives_by_project", "; ".join(f"{p}={v}" for p, v in neg.items()))
STRICT = ["committed_issue_count", "committed_story_points", "sprint_length_days", "n_concurrent_sprints",
          "hist_avg_completion_ratio_3", "hist_last_carryover", "hist_avg_delivered_sp_3", "hist_n_available",
          "hist_avg_completed_issues_3", "hist_avg_committed_issues_3", "scope_to_capacity", "hist_avg_late_days_3", "hist_late_rate_3"]
SC = ["committed_issue_count", "committed_story_points", "sprint_length_days"]
def lr(num, proj):
    t = [("num", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), num)]
    if proj: t.append(("cat", OneHotEncoder(handle_unknown="ignore"), ["project"]))
    return Pipeline([("p", ColumnTransformer(t)), ("m", LogisticRegression(max_iter=1000, class_weight="balanced"))])
te = te.copy()
te["lr"] = lr(STRICT, True).fit(tr[STRICT + ["project"]], tr.label_delay).predict_proba(te[STRICT + ["project"]])[:, 1]
te["sc"] = lr(SC, False).fit(tr[SC], tr.label_delay).predict_proba(te[SC])[:, 1]
for s in ["lr", "sc"]:
    v = [auc(g.label_delay, g[s]) for _, g in te.groupby("project")]
    add(f"late_within_mean_auc_{s}", round(np.mean(v), 3))
add("late_pooled_auc_lr", round(auc(te.label_delay, te.lr), 3))
x = e[(e.task == "Late closure") & (e.model == "Scope-only LR (baseline)") & e.interpretable]
add("external_scope_only_below_half", f"{int((x.auc < .5).sum())} of {len(x)}")
pd.DataFrame(R).to_csv(sys.argv[3], index=False)
