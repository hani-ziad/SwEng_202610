"""Phase 2, step 3: the sprint advisor, evaluated honestly, for both outcomes.

(a) Spillover side: 80% intervals for 'issues completed' (coverage, width,
    pinball loss) vs a no-learning baseline (training completion-rate quantiles
    x planned scope); P(any spillover) vs the training base rate.
(b) Late-closure side: P(late) calibration, Brier and AUC vs the base rate;
    median and 90th-percentile days-late forecasts vs global training quantiles
    (pinball loss, and how often the 90th-percentile forecast is exceeded).
    Both under the time-ordered split and leave-one-project-out.
(c) Does the advice have an observational basis? If the what-if table is to
    mean "plan less -> more gets done / closes on time", sprints planned heavy
    relative to their team's recent capacity should do worse. Tested directly,
    within each project, for carryover share AND for late closure.
(d) Worked what-if examples on real held-out sprints.

Usage (from the phase-2 folder):  python -m experiments.run_advisor
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import brier_score_loss, roc_auc_score

import phase2
from phase2.advisor import DONE_QUANTILES, LATE_QUANTILES, SprintAdvisor
from phase2.data import load_phase2_frame, time_ordered_split

SEED = 42
N_BOOT = 2000


def pinball(y, p, q):
    diff = np.asarray(y, float) - np.asarray(p, float)
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


# ------------------------------------------------------------------ baselines
def done_baseline(train, test):
    ne = train[train["nonempty"]]
    ratio = ne["completed_issue_count"] / ne["committed_issue_count"]
    qs = np.quantile(ratio, DONE_QUANTILES)
    scope = test["committed_issue_count"].to_numpy(float)
    return pd.DataFrame({"q10": qs[0] * scope, "q50": qs[1] * scope, "q90": qs[2] * scope}, index=test.index)


def late_baseline(train, test):
    qs = np.quantile(train["late_days_capped"], LATE_QUANTILES)
    return pd.DataFrame({"late_q50": np.full(len(test), qs[0]), "late_q90": np.full(len(test), qs[1]),
                         "p_late": np.full(len(test), train["label_delay"].mean())}, index=test.index)


# ------------------------------------------------------------------- scoring
def score_done(test, fc, name, protocol):
    t = test[test["nonempty"]]
    fc = fc.loc[t.index]
    y = t["completed_issue_count"].to_numpy(float)
    covered = (y >= fc["q10"].to_numpy()) & (y <= fc["q90"].to_numpy())
    bins = pd.cut(t["committed_issue_count"], [0, 3, 12, np.inf], labels=["1-3 issues", "4-12 issues", "13+ issues"])
    row = {"protocol": protocol, "forecaster": name, "n": len(t), "coverage_80": float(covered.mean()),
           "mean_interval_width": float((fc["q90"] - fc["q10"]).mean()),
           "mae_median": float(np.mean(np.abs(y - fc["q50"]))),
           "mean_pinball": float(np.mean([pinball(y, fc[f"q{int(q*100)}"], q) for q in DONE_QUANTILES]))}
    for b in bins.cat.categories:
        row[f"coverage_80__{b}"] = float(covered[(bins == b).to_numpy()].mean())
    return row


def score_late(test, fc, name, protocol):
    y = test["late_days_capped"].to_numpy(float)
    flag = test["label_delay"].to_numpy(int)
    p = fc["p_late"].to_numpy(float)
    row = {"protocol": protocol, "forecaster": name, "n": len(test), "late_rate": float(flag.mean()),
           "brier_p_late": float(brier_score_loss(flag, np.clip(p, 0, 1))),
           "auc_p_late": float(roc_auc_score(flag, p)) if np.std(p) > 0 else float("nan"),
           "pinball_q50": pinball(y, fc["late_q50"], 0.5), "pinball_q90": pinball(y, fc["late_q90"], 0.9),
           "share_below_q90_forecast": float(np.mean(y <= fc["late_q90"].to_numpy())),
           "mean_q90_forecast_days": float(fc["late_q90"].mean())}
    return row


def calibration(test, p, n_bins=4):
    d = pd.DataFrame({"p": p, "y": test["label_delay"].to_numpy()})
    d["bin"] = pd.qcut(d["p"].rank(method="first"), n_bins, labels=False)
    return d.groupby("bin").agg(n=("y", "size"), mean_predicted=("p", "mean"), observed_rate=("y", "mean")).reset_index()


# ------------------------------------------------------------------- overload
def overload_study(d):
    """Within-project association between relative load and bad outcomes (nonempty sprints)."""
    ne = d[d["nonempty"] & d["scope_to_capacity"].notna()].copy()
    ne["load_quartile"] = ne.groupby("project")["scope_to_capacity"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 4, labels=False))
    outcomes = {"carry_ratio": "carryover share", "label_spillover": "spillover rate",
                "label_delay": "late-closure rate", "late_days_capped": "mean days late (cap 30)"}
    quart = ne.groupby("load_quartile").agg(
        n=("carry_ratio", "size"), median_planned_issues=("committed_issue_count", "median"),
        **{k: (k, "mean") for k in outcomes}).reset_index()

    per_project = []
    for proj, g in ne.groupby("project"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            row = {"project": proj, "n": len(g)}
            for k in ("carry_ratio", "label_delay", "late_days_capped"):
                row[f"spearman_load_vs_{k}"] = float(spearmanr(g["scope_to_capacity"], g[k])[0])
        per_project.append(row)
    pp = pd.DataFrame(per_project)

    rng = np.random.default_rng(SEED)
    projs = ne["project"].unique()
    idx = {p: np.where(ne["project"].to_numpy() == p)[0] for p in projs}
    lq = ne["load_quartile"].to_numpy()
    summary = {"n_projects": int(len(projs)), "n_sprints": int(len(ne))}
    for k in ("carry_ratio", "label_delay"):
        v = ne[k].to_numpy(float)
        point = float(v[lq == 3].mean() - v[lq == 0].mean())
        diffs = []
        for _ in range(N_BOOT):
            r = np.concatenate([idx[p] for p in rng.choice(projs, len(projs), replace=True)])
            diffs.append(v[r][lq[r] == 3].mean() - v[r][lq[r] == 0].mean())
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        col = f"spearman_load_vs_{k}"
        summary[k] = {"heaviest_minus_lightest_quartile": point, "ci95_cluster_bootstrap": [float(lo), float(hi)],
                      "mean_within_project_spearman": float(pp[col].mean()),
                      "projects_with_positive_spearman": int((pp[col] > 0).sum())}
    return quart, pp, summary


# ----------------------------------------------------------------------- main
def main():
    d = load_phase2_frame()
    done_rows, late_rows, calib_rows = [], [], []

    train, test = time_ordered_split(d)
    adv = SprintAdvisor().fit(train)
    fc_late = adv.forecast_late(test)
    done_rows += [score_done(test, adv.forecast_completed(test), "quantile_gradient_boosting", "time_ordered"),
                  score_done(test, done_baseline(train, test), "scope_x_rate_quantiles", "time_ordered")]
    late_rows += [score_late(test, fc_late, "gradient_boosting", "time_ordered"),
                  score_late(test, late_baseline(train, test), "global_train_quantiles", "time_ordered")]
    c = calibration(test, fc_late["p_late"].to_numpy()); c.insert(0, "protocol", "time_ordered"); calib_rows.append(c)
    ne_test = test[test["nonempty"]]
    spill_p = adv.p_any_spillover(ne_test)
    spill = {"time_ordered": {
        "auc": float(roc_auc_score(ne_test["label_spillover"], spill_p)),
        "brier": float(brier_score_loss(ne_test["label_spillover"], spill_p)),
        "brier_base_rate": float(brier_score_loss(
            ne_test["label_spillover"], np.full(len(ne_test), train[train["nonempty"]]["label_spillover"].mean()))),
        "spillover_rate": float(ne_test["label_spillover"].mean())}}

    d0 = d.reset_index(drop=True)
    parts = {k: [] for k in ("done", "done_b", "late", "late_b", "spill")}
    for proj in sorted(d0["project"].unique()):
        m = (d0["project"] == proj).to_numpy()
        tr, te = d0.loc[~m], d0.loc[m]
        a = SprintAdvisor(with_project=False).fit(tr)
        parts["done"].append(a.forecast_completed(te)); parts["done_b"].append(done_baseline(tr, te))
        parts["late"].append(a.forecast_late(te)); parts["late_b"].append(late_baseline(tr, te))
        parts["spill"].append(pd.Series(a.p_any_spillover(te[te["nonempty"]]), index=te[te["nonempty"]].index))
    cat = {k: pd.concat(v).loc[d0.index] if k != "spill" else pd.concat(v) for k, v in parts.items()}
    done_rows += [score_done(d0, cat["done"], "quantile_gradient_boosting", "lopo"),
                  score_done(d0, cat["done_b"], "scope_x_rate_quantiles", "lopo")]
    late_rows += [score_late(d0, cat["late"], "gradient_boosting", "lopo"),
                  score_late(d0, cat["late_b"], "global_train_quantiles", "lopo")]
    c = calibration(d0, cat["late"]["p_late"].to_numpy()); c.insert(0, "protocol", "lopo"); calib_rows.append(c)
    ne0 = d0[d0["nonempty"]]
    sp = cat["spill"].loc[ne0.index].to_numpy()
    spill["lopo"] = {"auc": float(roc_auc_score(ne0["label_spillover"], sp)),
                     "brier": float(brier_score_loss(ne0["label_spillover"], sp)),
                     "spillover_rate": float(ne0["label_spillover"].mean())}

    dq, lq_, cal = pd.DataFrame(done_rows), pd.DataFrame(late_rows), pd.concat(calib_rows)
    dq.to_csv(phase2.RESULTS_DIR / "advisor_done_forecast_quality.csv", index=False)
    lq_.to_csv(phase2.RESULTS_DIR / "advisor_late_forecast_quality.csv", index=False)
    cal.to_csv(phase2.RESULTS_DIR / "advisor_late_calibration.csv", index=False)

    quart, pp, over = overload_study(d)
    quart.to_csv(phase2.RESULTS_DIR / "advisor_load_quartiles.csv", index=False)
    pp.to_csv(phase2.RESULTS_DIR / "advisor_overload_check.csv", index=False)

    # worked what-if examples (held-out nonempty sprints)
    t = test[test["nonempty"]].sort_values("committed_issue_count")
    picks = [t.iloc[int(len(t) * q)] for q in (0.25, 0.5, 0.9)]
    out = []
    for sprint in picks:
        planned = int(sprint["committed_issue_count"])
        grid = sorted({max(1, int(round(planned * f))) for f in (0.25, 0.5, 0.75, 1.0, 1.25, 1.5)})
        w = adv.what_if(sprint, grid)
        w.insert(0, "sprint_id", sprint["sprint_id"]); w.insert(1, "project", sprint["project"])
        w.insert(2, "actually_planned", planned); w.insert(3, "actually_completed", int(sprint["completed_issue_count"]))
        w.insert(4, "actually_late_days", round(float(sprint["late_days_capped"]), 1))
        out.append(w)
    ex = pd.concat(out, ignore_index=True)
    ex.to_csv(phase2.RESULTS_DIR / "advisor_whatif_examples.csv", index=False)

    pd.set_option("display.width", 240); pd.set_option("display.max_columns", 30)
    print("== completed-issues intervals"); print(dq.round(3).to_string(index=False))
    print("\n== late-closure forecasts"); print(lq_.round(3).to_string(index=False))
    print("\n== P(late) calibration (quartiles of predicted probability)"); print(cal.round(3).to_string(index=False))
    print("\n== P(any spillover)"); print(json.dumps(spill, indent=2))
    print("\n== load quartiles"); print(quart.round(3).to_string(index=False))
    print(pp.round(3).to_string(index=False)); print(json.dumps(over, indent=2))
    print("\n== what-if examples"); print(ex.to_string(index=False))
    (phase2.RESULTS_DIR / "advisor_summary.json").write_text(
        json.dumps({"overload": over, "spillover_probability": spill}, indent=2))


if __name__ == "__main__":
    main()
