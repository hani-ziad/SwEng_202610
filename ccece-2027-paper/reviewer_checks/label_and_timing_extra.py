"""Extra checks from the blind review: repeat commitments, missing resolution dates by project,
and cluster-bootstrap intervals for the LOPO timing reruns.
Usage: python label_and_timing_extra.py <commitment_provenance.csv> <tawos_issue.csv> <phase2_frame.csv> <out.csv>
"""
import sys
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score as auc
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

cp = pd.read_csv(sys.argv[1])
iss = pd.read_csv(sys.argv[2], na_values=["NULL"], usecols=["ID", "Resolution_Date"])
d = pd.read_csv(sys.argv[3])
cp = cp[cp.sprint_id.isin(d.sprint_id)]
R = []
def add(k, v, note=""): R.append({"check": k, "value": v, "note": note}); print(k, v, note)
add("commitments_in_modelled_sprints", len(cp))
add("commitments_all_sprints_incl_cold_start", len(pd.read_csv(sys.argv[1])))
per_issue = cp.groupby("raw_issue_id").size()
add("distinct_committed_issues", len(per_issue))
add("share_commitments_that_are_repeats", round(1 - len(per_issue) / len(cp), 3), "issue already committed to an earlier modelled sprint")
add("share_issues_committed_more_than_once", round((per_issue > 1).mean(), 3))
m = cp.merge(iss, left_on="raw_issue_id", right_on="ID", how="left")
nr = m.groupby("project").Resolution_Date.apply(lambda s: round(s.isna().mean(), 3))
for p, v in nr.items():
    add(f"no_resolution_date_share_{p}", v)
add("no_resolution_date_share_all", round(m.Resolution_Date.isna().mean(), 3))
# LOPO timing reruns with cluster intervals on pooled predictions
SCOPE = ["committed_issue_count", "committed_story_points", "sprint_length_days"]
LEG = ["prior_avg_delivered_sp_3", "prior_avg_completion_ratio_3", "prior_sprint_carryover", "prior_sprint_completion_ratio"]
STR = ["hist_avg_completion_ratio_3", "hist_last_carryover", "hist_avg_delivered_sp_3", "hist_n_available"]
LATE = ["hist_avg_late_days_3", "hist_late_rate_3"]
SETS = {"legacy": SCOPE + LEG, "strict_timed": SCOPE + STR, "legacy_plus_late": SCOPE + LEG + LATE}
def mk(m, cols):
    pre = ColumnTransformer([("n", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), cols)])
    est = (LogisticRegression(max_iter=1000, class_weight="balanced") if m == "LR" else
           RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=3, class_weight="balanced", random_state=42, n_jobs=-1))
    return Pipeline([("p", pre), ("m", est)])
d = d.reset_index(drop=True)
P = d.project.unique()
for m in ["LR", "RF"]:
    pred = {}
    for s, cols in SETS.items():
        out = pd.Series(index=d.index, dtype=float)
        for p in P:
            tr, te = d[d.project != p], d[d.project == p]
            out[te.index] = mk(m, cols).fit(tr[cols], tr.label_delay).predict_proba(te[cols])[:, 1]
        pred[s] = out
    rng = np.random.default_rng(42); G = {p: d.index[d.project == p].to_numpy() for p in P}
    for a, b in [("strict_timed", "legacy"), ("legacy_plus_late", "legacy")]:
        diffs = []
        for _ in range(2000):
            idx = np.concatenate([G[p] for p in rng.choice(P, len(P), replace=True)])
            y = d.label_delay[idx]
            if y.nunique() < 2: continue
            diffs.append(auc(y, pred[a][idx]) - auc(y, pred[b][idx]))
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        add(f"lopo_{m}_{a}_minus_legacy", round(auc(d.label_delay, pred[a]) - auc(d.label_delay, pred[b]), 3), f"CI {lo:.3f},{hi:.3f}")
pd.DataFrame(R).to_csv(sys.argv[4], index=False)
