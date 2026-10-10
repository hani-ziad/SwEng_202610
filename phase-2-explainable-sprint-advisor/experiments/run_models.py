"""Phase 2, step 1: how much of each outcome can planning-time information predict?

Five tasks (see phase2/models.py), two outcome families:
  spillover:     carry_share, issues_done, spillover (yes/no)
  late closure:  late_days, late_closure (yes/no)

Protocols: Phase 1's time-ordered split, and leave-one-project-out (LOPO,
project identity removed). Two feature sets: ``strict`` (history that was really
known at planning time + parallel-sprint count) and ``legacy`` (Phase 1's
history features). Every learned model is compared with no-learning baselines.
Differences against the task's primary baseline use a project-cluster bootstrap
(resample the 8 projects, 2,000 times), as in Phase 1.

Usage (from the phase-2 folder):  python -m experiments.run_models
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import phase2
from phase2.data import FEATURE_SETS, load_phase2_frame, population, time_ordered_split
from phase2.models import (
    CLF_MODELS,
    REG_BASELINES,
    REG_MODELS,
    TASKS,
    classification_metrics,
    clf_baseline_scores,
    feature_columns,
    make_classifier,
    make_regressor,
    regression_metrics,
)

N_BOOT = 2000
SEED = 42


# ---------------------------------------------------------------- predictions
def _fit_predict(task, train, test, numeric, with_project):
    cols = feature_columns(numeric, with_project)
    preds = {}
    if task.kind == "reg":
        for name in REG_MODELS:
            m = make_regressor(name, numeric, with_project).fit(train[cols], train[task.target])
            preds[name] = m.predict(test[cols])
        for name, fn in REG_BASELINES[task.target].items():
            preds[name] = np.asarray(fn(train, test, task.target), dtype=float)
    else:
        for name in CLF_MODELS:
            m = make_classifier(name, numeric, with_project).fit(train[cols], train[task.target])
            preds[name] = m.predict_proba(test[cols])[:, 1]
        preds.update(clf_baseline_scores(train, test, task.target, with_project))
    return preds


def predict_time_ordered(task, d, numeric):
    train, test = time_ordered_split(d)
    return test.reset_index(drop=True), _fit_predict(task, train, test, numeric, True)


def predict_lopo(task, d, numeric):
    d = d.reset_index(drop=True)
    store = None
    for proj in sorted(d["project"].unique()):
        mask = (d["project"] == proj).to_numpy()
        preds = _fit_predict(task, d.loc[~mask], d.loc[mask], numeric, False)
        if store is None:
            store = {k: np.zeros(len(d)) for k in preds}
        for k, v in preds.items():
            store[k][mask] = v
    return d, store


# ------------------------------------------------------------------ bootstrap
def _cluster_index(test):
    projects = test["project"].to_numpy()
    uniq = np.unique(projects)
    return uniq, {p: np.where(projects == p)[0] for p in uniq}


def boot_mae_delta(test, y, p_model, p_base, rng):
    err_m, err_b = np.abs(y - p_model), np.abs(y - p_base)
    uniq, idx = _cluster_index(test)
    point = float(err_m.mean() - err_b.mean())
    deltas = []
    for _ in range(N_BOOT):
        rows = np.concatenate([idx[p] for p in rng.choice(uniq, size=len(uniq), replace=True)])
        deltas.append(err_m[rows].mean() - err_b[rows].mean())
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return point, float(lo), float(hi)


def boot_auc_delta(test, y, s_model, s_base, rng):
    uniq, idx = _cluster_index(test)
    point = roc_auc_score(y, s_model) - roc_auc_score(y, s_base)
    deltas = []
    for _ in range(N_BOOT):
        rows = np.concatenate([idx[p] for p in rng.choice(uniq, size=len(uniq), replace=True)])
        yy = y[rows]
        if yy.min() == yy.max():
            continue
        deltas.append(roc_auc_score(yy, s_model[rows]) - roc_auc_score(yy, s_base[rows]))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return float(point), float(lo), float(hi)


# ----------------------------------------------------------------------- main
def main():
    d_all = load_phase2_frame()
    d_all.to_csv(phase2.RESULTS_DIR / "phase2_modeling_frame.csv", index=False)
    rng = np.random.default_rng(SEED)
    rows, boot_rows = [], []

    for key, task in TASKS.items():
        d = population(d_all, task.population)
        baselines = set(REG_BASELINES[task.target]) if task.kind == "reg" else {
            "always_positive", "train_base_rate", "scope_only_logreg", "project_base_rate"}
        models = REG_MODELS if task.kind == "reg" else CLF_MODELS
        for fset, numeric in FEATURE_SETS.items():
            for protocol, fn in [("time_ordered", predict_time_ordered), ("lopo", predict_lopo)]:
                test, preds = fn(task, d, numeric)
                y = test[task.target].to_numpy(float)
                for name, p in preds.items():
                    if name == "project_base_rate" and protocol == "lopo":
                        continue
                    m = (regression_metrics(y, p) if task.kind == "reg"
                         else classification_metrics(y, p))
                    rows.append({"task": key, "outcome": task.outcome, "feature_set": fset,
                                 "protocol": protocol, "model": name,
                                 "kind": "baseline" if name in baselines else "learned", **m})
                compare = ([task.primary_baseline, "prior_history"] if task.kind == "reg"
                           else [task.primary_baseline, "project_base_rate"])
                compare = list(dict.fromkeys(b for b in compare if b in preds))
                for name in models:
                    for base in compare:
                        if task.kind == "reg":
                            pt, lo, hi = boot_mae_delta(test, y, preds[name], preds[base], rng)
                            metric = "delta_mae"
                        else:
                            pt, lo, hi = boot_auc_delta(test, y, preds[name], preds[base], rng)
                            metric = "delta_auc"
                        boot_rows.append({"task": key, "feature_set": fset, "protocol": protocol,
                                          "model": name, "baseline": base,
                                          "metric": metric, "delta": pt, "ci_low": lo, "ci_high": hi,
                                          "ci_excludes_zero": bool(lo > 0 or hi < 0)})
            print(f"done {key} / {fset}", flush=True)

    res, boot = pd.DataFrame(rows), pd.DataFrame(boot_rows)
    res.to_csv(phase2.RESULTS_DIR / "model_results.csv", index=False)
    boot.to_csv(phase2.RESULTS_DIR / "model_bootstrap.csv", index=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 500)
    for key, task in TASKS.items():
        r = res[(res["task"] == key) & (res["feature_set"] == "strict")]
        cols = (["protocol", "model", "kind", "n", "mae", "r2", "spearman"] if task.kind == "reg"
                else ["protocol", "model", "kind", "positive_rate", "f1", "roc_auc", "pr_auc", "brier"])
        print(f"\n=== {key} ({task.description}) -- strict features")
        print(r[cols].round(3).to_string(index=False))
    print("\nModel minus primary baseline (strict). Reg: MAE (neg = better). Clf: AUC (pos = better).")
    print(boot[(boot["feature_set"] == "strict") & (boot["baseline"] == boot["task"].map({k: t.primary_baseline for k, t in TASKS.items()}))].round(3).to_string(index=False))
    summary = {
        "n_sprints_full": int(len(d_all)), "n_sprints_nonempty": int(d_all["nonempty"].sum()),
        "n_bootstrap_resamples": N_BOOT,
        "tasks": {k: {"population": t.population, "primary_baseline": t.primary_baseline}
                  for k, t in TASKS.items()},
    }
    (phase2.RESULTS_DIR / "model_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
