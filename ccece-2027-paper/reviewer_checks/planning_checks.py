"""Reviewer checks on planning-time claims. Args: phase2 frame, sprint csv, change log csv, external per-project csv, out csv."""
import sys, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score as auc
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
frame, sprint, clog, ext, out = sys.argv[1:6]
d = pd.read_csv(frame, parse_dates=["start_date"])
R = []
def add(k, v, note=""): R.append({"check": k, "value": v, "note": note}); print(k, v, note)
def tsplit(df):
    df = df.sort_values(["project", "start_date"]).copy()
    df["rk"] = df.groupby("project").cumcount(); df["np_"] = df.groupby("project").project.transform("size")
    return df[df.rk < np.floor(df.np_ * .7)], df[df.rk >= np.floor(df.np_ * .7)]
SC = ["committed_issue_count", "committed_story_points", "sprint_length_days"]
lr = lambda: Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler()), ("m", LogisticRegression(max_iter=1000, class_weight="balanced"))])
ne = d[d.nonempty]
tr, te = tsplit(ne)
add("spill_test_n", len(te)); add("spill_test_neg", int((te.label_spillover == 0).sum()))
add("spill_time_auc_committed_issue_count_raw", round(auc(te.label_spillover, te.committed_issue_count), 3), "no model, raw count")
add("spill_time_auc_scope_only_lr", round(auc(te.label_spillover, lr().fit(tr[SC], tr.label_spillover).predict_proba(te[SC])[:, 1]), 3))
tr2, te2 = tsplit(ne[ne.project != "Atlassian Confluence Cloud"])
add("spill_time_auc_scope_only_lr_without_confluence_cloud", round(auc(te2.label_spillover, lr().fit(tr2[SC], tr2.label_spillover).predict_proba(te2[SC])[:, 1]), 3))
add("late_rate_zero_scope", round(d[~d.nonempty].label_delay.mean(), 3)); add("late_rate_nonempty", round(ne.label_delay.mean(), 3))
# when do zero-scope sprints get their first issue?
s = pd.read_csv(sprint, na_values=["NULL"], parse_dates=["Start_Date"])
c = pd.read_csv(clog, parse_dates=["Creation_Date"], dtype={"From_Value": str, "To_Value": str})
c = c[c.Field == "Sprint"].copy()
sp = lambda v: [x.strip() for x in str(v).split(",")] if pd.notna(v) else []
c["tv"] = c.To_Value.apply(sp); c["fv"] = c.From_Value.apply(sp)
c = c.explode("tv").dropna(subset=["tv"]); c = c[[t not in f for t, f in zip(c.tv, c.fv)]]
c = c[c.tv.str.isnumeric()]; c["tv"] = c.tv.astype(int)
d["ID"] = d.sprint_id.str.split("__").str[1].astype(int)
m = d.merge(s[["ID", "JiraID", "Start_Date"]], on="ID")
m["lag_h"] = (m.JiraID.map(c.groupby("tv").Creation_Date.min()) - m.Start_Date).dt.total_seconds() / 3600
z = m[~m.nonempty]
add("zero_scope_n", len(z))
add("zero_scope_first_issue_added_within_1h_after_start", round(((z.lag_h > 0) & (z.lag_h <= 1)).mean(), 3))
add("zero_scope_first_issue_added_within_24h_after_start", round(((z.lag_h > 0) & (z.lag_h <= 24)).mean(), 3))
add("zero_scope_first_issue_added_any_time_after_start", round((z.lag_h > 0).mean(), 3))
# outside projects: how many GB late-closure intervals clear 0.5
e = pd.read_csv(ext)
g = e[(e.task == "Late closure") & (e.model == "Gradient boosting") & e.interpretable]
add("outside_late_gb_projects", len(g)); add("outside_late_gb_point_auc_above_half", int((g.auc > .5).sum()))
add("outside_late_gb_ci_low_above_half", int((g.auc_lo > .5).sum()), ", ".join(g[g.auc_lo > .5].project))
pd.DataFrame(R).to_csv(out, index=False)
