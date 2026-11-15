"""Checks added after an external referee report (15 Nov 2026): purged training labels, a
calendar-time split, a weekday baseline for late closure, intervals for the 3- and 7-day
thresholds, and a plausible-duration sensitivity run.
Usage: python elicit_review_checks.py <phase2_frame.csv> <tawos_sprint.csv> <out.csv>
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
def gb(extra=()):
    num = STRICT + list(extra)
    pre = ColumnTransformer([("num", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), num),
                             ("cat", OneHotEncoder(handle_unknown="ignore"), ["project"])])
    return Pipeline([("p", pre), ("m", HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, random_state=42))])
def split(df):
    df = df.sort_values(["project", "start_date", "sprint_id"]).copy()
    df["rk"] = df.groupby("project").cumcount(); df["np_"] = df.groupby("project").project.transform("size")
    cut = (df.np_ * 0.7).astype(int); return df[df.rk < cut], df[df.rk >= cut]
def within(t, col="p"):
    return [auc(g.y, g[col]) for _, g in t.groupby("project") if g.y.nunique() == 2]
def boot_within(t, cols, n=2000, seed=42):
    """Resample sprints inside each project; return rows of mean within-project AUC per column."""
    rng = np.random.default_rng(seed); groups = [g for _, g in t.groupby("project")]; out = []
    for _ in range(n):
        vals = {c: [] for c in cols}
        for g in groups:
            gi = g.iloc[rng.integers(0, len(g), len(g))]
            if gi.y.nunique() == 2:
                for c in cols: vals[c].append(auc(gi.y, gi[c]))
        out.append([np.mean(vals[c]) for c in cols])
    return np.array(out)
def ci(x): lo, hi = np.percentile(x, [2.5, 97.5]); return f"{lo:.3f},{hi:.3f}"

d = pd.read_csv(sys.argv[1], parse_dates=["start_date"])
s = pd.read_csv(sys.argv[2], na_values=["NULL"], parse_dates=["Start_Date", "End_Date", "Complete_Date"])
d["ID"] = d.sprint_id.str.split("__").str[1].astype(int)
d = d.merge(s[["ID", "End_Date", "Complete_Date"]], on="ID", how="left")
d["end_fri_wkd"] = (d.End_Date.dt.dayofweek >= 4).astype(int)
R = []
def add(k, v, note=""): R.append({"check": k, "value": v, "note": note}); print(k, v, note)
X = STRICT + ["project"]
tr, te = split(d)

# 1. thresholds 1, 3, 7 days: within-project intervals
for th in [1, 3, 7]:
    ytr, yte = (tr.late_days > th).astype(int), (te.late_days > th).astype(int)
    p = gb().fit(tr[X], ytr).predict_proba(te[X])[:, 1]
    t = te.assign(y=yte.values, p=p)
    b = boot_within(t, ["p"])
    add(f"th{th}_gb_within_mean", round(np.mean(within(t)), 3), f"{len(within(t))} projects"); add(f"th{th}_gb_within_ci", ci(b[:, 0]), "sprint bootstrap within projects")

# 2. purge training sprints whose label was not yet known when the project's first test sprint started
first_test = te.groupby("project").start_date.min()
unknown = tr.Complete_Date > tr.project.map(first_test)
add("purge_n_removed", int(unknown.sum()), "; ".join(f"{k}={v}" for k, v in tr[unknown].project.value_counts().items()))
y = te.label_delay.values
p0 = gb().fit(tr[X], tr.label_delay).predict_proba(te[X])[:, 1]
trp = tr[~unknown]
p1 = gb().fit(trp[X], trp.label_delay).predict_proba(te[X])[:, 1]
t = te.assign(y=y, p=p0, pp=p1)
add("orig_gb_pooled", round(auc(y, p0), 3)); add("orig_gb_within", round(np.mean(within(t)), 3))
add("purged_gb_pooled", round(auc(y, p1), 3)); add("purged_gb_within", round(np.mean(within(t, "pp")), 3))

# 3. calendar-time split: one cut date for all projects; train only on sprints closed before it
cutdate = d.start_date.quantile(0.7)
trc, tec = d[d.Complete_Date < cutdate], d[d.start_date >= cutdate]
add("calendar_cut_date", str(cutdate.date())); add("calendar_n_train", len(trc)); add("calendar_n_test", len(tec), f"{int(tec.label_delay.sum())} late")
add("calendar_dropped_straddling", int(len(d) - len(trc) - len(tec)), "started before the cut, closed after it")
add("calendar_test_projects", "; ".join(f"{k}={v}" for k, v in tec.project.value_counts().items()))
pc = gb().fit(trc[X], trc.label_delay).predict_proba(tec[X])[:, 1]
tc = tec.assign(y=tec.label_delay.values, p=pc, rate=tec.project.map(trc.groupby("project").label_delay.mean()).fillna(trc.label_delay.mean()).values)
wc = within(tc)
add("calendar_gb_pooled", round(auc(tc.y, tc.p), 3)); add("calendar_project_rate_pooled", round(auc(tc.y, tc.rate), 3))
add("calendar_gb_within", round(np.mean(wc), 3), f"{len(wc)} projects"); add("calendar_gb_within_ci", ci(boot_within(tc, ["p"])[:, 0]))

# 4. weekday baseline on the main split (planned end on Friday or weekend), training rates only
glob = tr.groupby("end_fri_wkd").label_delay.mean()
pw = tr.groupby(["project", "end_fri_wkd"]).label_delay.agg(["mean", "size"])
def pw_rate(r):
    k = (r.project, r.end_fri_wkd)
    if k in pw.index and pw.loc[k, "size"] >= 5: return pw.loc[k, "mean"]
    return tr[tr.project == r.project].label_delay.mean()
t = te.assign(y=y, p=p0, wd=te.end_fri_wkd.map(glob).values, pwd=te.apply(pw_rate, axis=1).values)
p2 = gb(["end_fri_wkd"]).fit(tr[X + ["end_fri_wkd"]], tr.label_delay).predict_proba(te[X + ["end_fri_wkd"]])[:, 1]
t["pgw"] = p2
add("weekday_rule_pooled", round(auc(y, t.wd), 3)); add("weekday_rule_within", round(np.mean(within(t, "wd")), 3), f"{len(within(t, 'wd'))} projects")
add("project_weekday_rate_pooled", round(auc(y, t.pwd), 3)); add("project_weekday_rate_within", round(np.mean(within(t, "pwd")), 3))
add("gb_plus_weekday_pooled", round(auc(y, t.pgw), 3)); add("gb_plus_weekday_within", round(np.mean(within(t, "pgw")), 3))
b = boot_within(t, ["p", "wd", "pwd", "pgw"])
add("weekday_rule_within_ci", ci(b[:, 1])); add("project_weekday_rate_within_ci", ci(b[:, 2]))
add("gb_minus_weekday_rule_within", round(np.mean(within(t)) - np.mean(within(t, "wd")), 3), "CI " + ci(b[:, 0] - b[:, 1]))
add("gb_minus_project_weekday_within", round(np.mean(within(t)) - np.mean(within(t, "pwd")), 3), "CI " + ci(b[:, 0] - b[:, 2]))
add("gb_plus_weekday_minus_gb_within", round(np.mean(within(t, "pgw")) - np.mean(within(t)), 3), "CI " + ci(b[:, 3] - b[:, 0]))
add("test_share_end_fri_wkd", round(te.end_fri_wkd.mean(), 3))
add("end_fri_wkd_share_by_project", "; ".join(f"{k}={v:.2f}" for k, v in d.groupby("project").end_fri_wkd.mean().items()))

# 5. plausible durations only (1 to 60 days)
ok = (d.sprint_length_days >= 1) & (d.sprint_length_days <= 60)
tr2, te2 = split(d[ok])
p3 = gb().fit(tr2[X], tr2.label_delay).predict_proba(te2[X])[:, 1]
t2 = te2.assign(y=te2.label_delay.values, p=p3)
add("dur_1_60_n", int(ok.sum())); add("dur_1_60_gb_pooled", round(auc(t2.y, t2.p), 3)); add("dur_1_60_gb_within", round(np.mean(within(t2)), 3))
pd.DataFrame(R).to_csv(sys.argv[3], index=False)
