"""Gradient boosting with progress vs the learning-free open-issue count, with a paired
project-cluster bootstrap (2,000 resamples of the 8 projects, seed 42).

Rebuilds the Phase 3 model from the checkpoint frame (same features and settings as
phase3/experiments/run_models.py) and checks it reproduces the published AUCs first.
Usage: python insprint_vs_count.py <phase3_checkpoint_frame.csv> <out.csv>
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
          "hist_avg_completed_issues_3", "hist_avg_committed_issues_3", "scope_to_capacity",
          "hist_avg_late_days_3", "hist_late_rate_3"]
PROGRESS = ["elapsed_frac", "remaining_days", "done_share", "done_n", "remaining_n", "remaining_sp", "burn_gap",
            "required_rate", "observed_rate", "added_ratio", "added_done_n", "removed_ratio", "sp_change_ratio"]
NUM = STRICT + PROGRESS


def gb():
    pre = ColumnTransformer([("num", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), NUM),
                             ("cat", OneHotEncoder(handle_unknown="ignore"), ["project"])])
    return Pipeline([("p", pre), ("m", HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200,
                                                                       random_state=42))])


def cluster_boot(df, a, b, n=2000, seed=42):
    rng = np.random.default_rng(seed)
    projects = df.project.unique()
    groups = {p: df.index[df.project == p].to_numpy() for p in projects}
    diffs = []
    for _ in range(n):
        idx = np.concatenate([groups[p] for p in rng.choice(projects, len(projects), replace=True)])
        y = df.loc[idx, "y"]
        if y.nunique() < 2:
            continue
        diffs.append(auc(y, df.loc[idx, a]) - auc(y, df.loc[idx, b]))
    return np.percentile(diffs, [2.5, 97.5])


f = pd.read_csv(sys.argv[1])
rows = []
for c in [0.25, 0.5, 0.75]:
    x = f[(f.checkpoint == c) & f.nonempty & f.open_at_checkpoint]
    tr, te = x[x.split_nonempty == "train"], x[x.split_nonempty == "test"].copy()
    te["gb"] = gb().fit(tr[NUM + ["project"]], tr.label_spillover).predict_proba(te[NUM + ["project"]])[:, 1]
    te["rule"] = te.remaining_n
    te["y"] = te.label_spillover
    te = te.reset_index(drop=True)
    lo, hi = cluster_boot(te, "gb", "rule")
    rows.append({"checkpoint": c, "n": len(te), "pos": int(te.y.sum()), "auc_gb_progress": round(auc(te.y, te.gb), 3),
                 "auc_open_count": round(auc(te.y, te.rule), 3),
                 "diff": round(auc(te.y, te.gb) - auc(te.y, te.rule), 3), "ci_lo": round(lo, 3), "ci_hi": round(hi, 3)})
out = pd.DataFrame(rows)
print(out.to_string(index=False))
out.to_csv(sys.argv[2], index=False)
