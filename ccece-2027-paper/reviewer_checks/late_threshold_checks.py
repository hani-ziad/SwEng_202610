"""Second blind review checks: late-closure threshold sensitivity, weekday of planned end,
within-project interval, repeat-window late rates, label timing inside the training split.
Usage: python late_threshold_checks.py <phase2_frame.csv> <tawos_sprint.csv> <e1_repeats.csv> <out.csv>
"""
import sys
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score as auc
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

STRICT = ["committed_issue_count", "committed_story_points", "sprint_length_days", "n_concurrent_sprints",
          "hist_avg_completion_ratio_3", "hist_last_carryover", "hist_avg_delivered_sp_3", "hist_n_available",
          "hist_avg_completed_issues_3", "hist_avg_committed_issues_3", "scope_to_capacity", "hist_avg_late_days_3", "hist_late_rate_3"]
def gb():
    pre = ColumnTransformer([("num", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), STRICT),
                             ("cat", OneHotEncoder(handle_unknown="ignore"), ["project"])])
    return Pipeline([("p", pre), ("m", HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, random_state=42))])
def split(df):
    df = df.sort_values(["project", "start_date", "sprint_id"]).copy()
    df["rk"] = df.groupby("project").cumcount(); df["np_"] = df.groupby("project").project.transform("size")
    cut = (df.np_ * 0.7).astype(int); return df[df.rk < cut], df[df.rk >= cut]

d = pd.read_csv(sys.argv[1], parse_dates=["start_date"])
s = pd.read_csv(sys.argv[2], na_values=["NULL"], parse_dates=["Start_Date", "End_Date", "Complete_Date"])
R = []
def add(k, v, note=""): R.append({"check": k, "value": v, "note": note}); print(k, v, note)
tr, te = split(d)
for th in [1, 3, 7]:
    ytr, yte = (tr.late_days > th).astype(int), (te.late_days > th).astype(int)
    p = gb().fit(tr[STRICT + ["project"]], ytr).predict_proba(te[STRICT + ["project"]])[:, 1]
    rate = te.project.map(ytr.groupby(tr.project).mean())
    t2 = te.assign(y=yte.values, p=p)
    w = [auc(g.y, g.p) for _, g in t2.groupby("project") if g.y.nunique() == 2]
    add(f"th{th}_late_rate_all", round((d.late_days > th).mean(), 3)); add(f"th{th}_test_pos", int(yte.sum()))
    add(f"th{th}_gb_pooled", round(auc(yte, p), 3)); add(f"th{th}_project_rate", round(auc(yte, rate), 3))
    add(f"th{th}_gb_within_mean", round(np.mean(w), 3), f"{len(w)} projects")
    if th == 1:
        # within-project interval: resample sprints inside each project (2,000 times)
        rng = np.random.default_rng(42); bs = []
        groups = {pp: g for pp, g in t2.groupby("project")}
        for _ in range(2000):
            vals = []
            for pp, g in groups.items():
                gi = g.iloc[rng.integers(0, len(g), len(g))]
                if gi.y.nunique() == 2: vals.append(auc(gi.y, gi.p))
            bs.append(np.mean(vals))
        lo, hi = np.percentile(bs, [2.5, 97.5]); add("th1_gb_within_ci", f"{lo:.3f},{hi:.3f}", "sprint bootstrap within projects")
        add("test_late_positives_by_project", "; ".join(f"{pp}={int(g.y.sum())}" for pp, g in groups.items()))
late = d[d.label_delay == 1]
add("late_sprints_1_to_3_days_share", round(((late.late_days > 1) & (late.late_days <= 3)).mean(), 3))
add("late_days_median_among_late", round(late.late_days.median(), 1))
# weekday of planned end
d["ID"] = d.sprint_id.str.split("__").str[1].astype(int)
m = d.merge(s[["ID", "End_Date", "Complete_Date"]], on="ID")
wd = m.End_Date.dt.dayofweek
add("planned_end_friday_share", round((wd == 4).mean(), 3)); add("planned_end_weekend_share", round((wd >= 5).mean(), 3))
add("late_rate_end_fri_or_weekend", round(m[wd >= 4].label_delay.mean(), 3)); add("late_rate_end_mon_to_thu", round(m[wd < 4].label_delay.mean(), 3))
# training-sprint labels known only after the first test sprint of the project starts
mm = d.merge(s[["ID", "Complete_Date"]], on="ID")
trm, tem = split(mm)
first_test = tem.groupby("project").start_date.min()
lateknown = (trm.Complete_Date > trm.project.map(first_test))
add("train_labels_known_after_test_start", int(lateknown.sum()), f"of {len(trm)}")
add("sprint_length_under_1_day", int((d.sprint_length_days < 1).sum())); add("sprint_length_over_60_days", int((d.sprint_length_days > 60).sum()))
rep = pd.read_csv(sys.argv[3])
r = rep[(rep.task == "late_closure") & rep.scheme.str.startswith("C.") & (rep.model == "Gradient boosting")].sort_values("repeat")
add("repeat_window_late_rates", "; ".join(f"{a/b:.2f}" for a, b in zip(r.n_pos, r.n)))
pd.DataFrame(R).to_csv(sys.argv[4], index=False)
