"""Late-closure checks asked for by the blind review.
Within-project AUC, learning-free past-lateness rules, nonempty-only analysis, RF seed variance,
and the spillover all-sprint population numbers. Time split as in the paper (per project, first 70%).
Usage: python late_closure_checks.py <phase2_modeling_frame.csv> <out.csv>
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
LATE = ["hist_avg_late_days_3", "hist_late_rate_3"]


def pipe(est, num, proj=True):
    t = [("num", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), num)]
    if proj:
        t.append(("cat", OneHotEncoder(handle_unknown="ignore"), ["project"]))
    return Pipeline([("p", ColumnTransformer(t)), ("m", est)])


def gb(seed=42):
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, random_state=seed)


def rf(seed=42):
    return RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=3, class_weight="balanced",
                                  random_state=seed, n_jobs=-1)


def split(df):
    df = df.sort_values(["project", "start_date", "sprint_id"]).copy()
    df["rk"] = df.groupby("project").cumcount()
    df["np_"] = df.groupby("project").project.transform("size")
    cut = (df.np_ * 0.7).astype(int)
    return df[df.rk < cut], df[df.rk >= cut]


def within(te, score, y="label_delay"):
    vals = {p: auc(g[y], g[score]) for p, g in te.groupby("project") if g[y].nunique() == 2 and g[score].nunique() > 1}
    return np.mean(list(vals.values())), min(vals.values()), max(vals.values()), len(vals), vals


def boot_diff(te, a, b, y="label_delay", n=2000, seed=42):
    rng = np.random.default_rng(seed)
    te = te.reset_index(drop=True)
    P = te.project.unique()
    G = {p: te.index[te.project == p].to_numpy() for p in P}
    out = []
    for _ in range(n):
        idx = np.concatenate([G[p] for p in rng.choice(P, len(P), replace=True)])
        s = te.loc[idx]
        if s[y].nunique() < 2:
            continue
        out.append(auc(s[y], s[a]) - auc(s[y], s[b]))
    return np.percentile(out, [2.5, 97.5])


d = pd.read_csv(sys.argv[1], parse_dates=["start_date"])
R = []
def add(k, v, note=""):
    R.append({"check": k, "value": v, "note": note}); print(f"{k}: {v} {note}")

for pop, df in [("all", d), ("nonempty", d[d.nonempty])]:
    tr, te = split(df)
    te = te.copy()
    te["gb"] = pipe(gb(), STRICT).fit(tr[STRICT + ["project"]], tr.label_delay).predict_proba(te[STRICT + ["project"]])[:, 1]
    te["rf"] = pipe(rf(), STRICT).fit(tr[STRICT + ["project"]], tr.label_delay).predict_proba(te[STRICT + ["project"]])[:, 1]
    te["proj_rate"] = te.project.map(tr.groupby("project").label_delay.mean())
    te["rule_late_rate"] = te.hist_late_rate_3.fillna(tr.hist_late_rate_3.median())
    te["rule_late_days"] = te.hist_avg_late_days_3.fillna(tr.hist_avg_late_days_3.median())
    te["lr_late_proj"] = pipe(LogisticRegression(max_iter=1000, class_weight="balanced"), LATE).fit(
        tr[LATE + ["project"]], tr.label_delay).predict_proba(te[LATE + ["project"]])[:, 1]
    te["count"] = te.committed_issue_count
    add(f"{pop}_test_n", len(te)); add(f"{pop}_test_late", int(te.label_delay.sum()))
    for s in ["gb", "rf", "proj_rate", "rule_late_rate", "rule_late_days", "lr_late_proj", "count"]:
        add(f"{pop}_pooled_auc_{s}", round(auc(te.label_delay, te[s]), 3))
    for s in ["gb", "rf", "rule_late_rate", "rule_late_days", "count"]:
        m, lo, hi, k, vals = within(te, s)
        add(f"{pop}_within_mean_auc_{s}", round(m, 3), f"min {lo:.3f} max {hi:.3f} over {k} projects")
        if s == "gb" and pop == "all":
            for p, v in vals.items():
                add(f"all_within_gb_{p}", round(v, 3))
    if pop == "all":
        m_gb = within(te, "gb")[4]; m_rule = within(te, "rule_late_rate")[4]
        add("all_projects_gb_beats_late_rate_rule_within", int(sum(m_gb[p] > m_rule[p] for p in m_gb)), f"of {len(m_gb)}")
        add("all_projects_gb_within_above_half", int(sum(v > .5 for v in m_gb.values())), f"of {len(m_gb)}")
        lo, hi = boot_diff(te, "gb", "rule_late_rate")
        add("all_pooled_gb_minus_late_rate_rule_ci", f"{lo:.3f},{hi:.3f}")
        # spillover population trap on the same 786 test sprints
        te_sp = te.copy()
        add("all_test_spill_pos", int(te_sp.label_spillover.sum()))
        add("all_test_spill_auc_nonempty_flag", round(auc(te_sp.label_spillover, te_sp.nonempty.astype(int)), 3))
        add("all_test_spill_auc_committed_count", round(auc(te_sp.label_spillover, te_sp.committed_issue_count), 3))

# RF seed variance on the headline tasks (GB is deterministic without subsampling or early stopping)
tr, te = split(d)
sv = [auc(te.label_delay, pipe(rf(s), STRICT).fit(tr[STRICT + ["project"]], tr.label_delay)
          .predict_proba(te[STRICT + ["project"]])[:, 1]) for s in range(10)]
add("rf_late_time_auc_seed_mean_sd", f"{np.mean(sv):.3f},{np.std(sv, ddof=1):.3f}", "10 seeds")
g2 = [auc(te.label_delay, pipe(gb(s), STRICT).fit(tr[STRICT + ["project"]], tr.label_delay)
          .predict_proba(te[STRICT + ["project"]])[:, 1]) for s in range(3)]
add("gb_late_time_auc_seed_range", f"{min(g2):.4f},{max(g2):.4f}", "3 seeds")
pd.DataFrame(R).to_csv(sys.argv[2], index=False)
