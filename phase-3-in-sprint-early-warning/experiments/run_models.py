"""Phase 3, step 1: does knowing how the sprint is going improve the prediction?

Four tasks (as in Phase 2, but made at a checkpoint inside the sprint):
  spillover     any committed issue not done by planned end        (nonempty sprints)
  carry_share   share of committed issues not done by planned end  (nonempty sprints)
  late_closure  closed more than 1 day after planned end           (all sprints)
  late_days     days closed after planned end, capped at 30        (all sprints)

Checkpoints: 25%, 50%, 75%, 100% of the planned duration. Sprints already closed
at the checkpoint are dropped (nothing left to predict). At 100% spillover and
carry_share are fully determined by the state, so only late closure is run there.

Each progress-aware model is compared with
  * the same algorithm using planning-time features only (Phase 2's strict set)
  * a no-learning rule built from progress (remaining share of committed scope)
  * simple baselines (always-positive F1, training base rate / median)
Protocols: Phase 1/2's time-ordered split and leave-one-project-out (project removed).
Confidence intervals: project-cluster bootstrap (resample the 8 projects, 2,000 times).

Usage (from the phase-3 folder):  python -m experiments.run_models
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import rankdata

import phase3
from phase2.data import STRICT_NUMERIC
from phase2.models import (
    CLF_MODELS,
    REG_MODELS,
    classification_metrics,
    feature_columns,
    make_classifier,
    make_regressor,
    regression_metrics,
)
from phase3.data import CHECKPOINT_NUMERIC, CHECKPOINTS, load_checkpoint_frame

N_BOOT = 2000
SEED = 42

# key: (population, target, kind, checkpoints, no-learning rule, rule description)
TASKS = {
    "spillover": ("nonempty", "label_spillover", "clf", (0.25, 0.5, 0.75), "remaining_share"),
    "carry_share": ("nonempty", "carry_ratio", "reg", (0.25, 0.5, 0.75), "remaining_share"),
    "late_closure": ("full", "label_delay", "clf", CHECKPOINTS, "remaining_issues"),
    "late_days": ("full", "late_days_capped", "reg", CHECKPOINTS, None),
}
FEATURE_SETS = {"progress": CHECKPOINT_NUMERIC, "planning": STRICT_NUMERIC}


def fast_auc(y, s):
    y = np.asarray(y)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def rule_scores(frame, rule):
    if rule == "remaining_share":
        return (1.0 - frame["done_share"]).to_numpy(float)
    if rule == "remaining_issues":
        return frame["remaining_n"].to_numpy(float)
    return None


def fit_predict(kind, train, test, numeric, with_project, target):
    cols = feature_columns(numeric, with_project)
    out = {}
    names = CLF_MODELS if kind == "clf" else REG_MODELS
    for name in names:
        if kind == "clf":
            m = make_classifier(name, numeric, with_project).fit(train[cols], train[target])
            out[name] = m.predict_proba(test[cols])[:, 1]
        else:
            m = make_regressor(name, numeric, with_project).fit(train[cols], train[target])
            out[name] = m.predict(test[cols])
    return out


def run_protocol(kind, pop, protocol, target, split_col):
    """Return (test frame, {feature_set: {algo: preds}}) for one protocol."""
    if protocol == "time_ordered":
        train, test = pop[pop[split_col] == "train"], pop[pop[split_col] == "test"]
        preds = {fs: fit_predict(kind, train, test, num, True, target) for fs, num in FEATURE_SETS.items()}
        return test.reset_index(drop=True), preds, train
    pop = pop.reset_index(drop=True)
    store = {fs: None for fs in FEATURE_SETS}
    for proj in sorted(pop["project"].unique()):
        mask = (pop["project"] == proj).to_numpy()
        for fs, num in FEATURE_SETS.items():
            p = fit_predict(kind, pop.loc[~mask], pop.loc[mask], num, False, target)
            if store[fs] is None:
                store[fs] = {k: np.zeros(len(pop)) for k in p}
            for k, v in p.items():
                store[fs][k][mask] = v
    return pop, store, None


def cluster_boot(test, fn, rng):
    """fn(row_index) -> statistic; resample projects."""
    projects = test["project"].to_numpy()
    uniq = np.unique(projects)
    idx = {p: np.where(projects == p)[0] for p in uniq}
    point = fn(np.arange(len(test)))
    vals = []
    for _ in range(N_BOOT):
        rows = np.concatenate([idx[p] for p in rng.choice(uniq, size=len(uniq), replace=True)])
        v = fn(rows)
        if np.isfinite(v):
            vals.append(v)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def main():
    frame = load_checkpoint_frame()
    frame.to_csv(phase3.RESULTS_DIR / "phase3_checkpoint_frame.csv", index=False)
    rng = np.random.default_rng(SEED)
    rows, boot_rows = [], []

    for key, (popname, target, kind, cps, rule) in TASKS.items():
        for cp in cps:
            sub = frame[(frame["checkpoint"] == cp) & frame["open_at_checkpoint"]]
            if popname == "nonempty":
                sub = sub[sub["nonempty"]]
            split_col = f"split_{popname}"
            for protocol in ("time_ordered", "lopo"):
                test, preds, train = run_protocol(kind, sub, protocol, target, split_col)
                y = test[target].to_numpy(float)
                extra = {}
                if kind == "clf":
                    extra["always_positive"] = np.ones(len(test))
                    base = train[target].mean() if train is not None else sub[target].mean()
                    extra["train_base_rate"] = np.full(len(test), base)
                else:
                    med = train[target].median() if train is not None else sub[target].median()
                    extra["train_median"] = np.full(len(test), med)
                    if rule == "remaining_share":
                        extra["rule_remaining_share"] = rule_scores(test, rule)
                if kind == "clf" and rule:
                    extra["rule_" + rule] = rule_scores(test, rule)

                def log(name, fset, p, learned):
                    m = regression_metrics(y, p) if kind == "reg" else classification_metrics(y, p)
                    rows.append({"task": key, "checkpoint": cp, "protocol": protocol, "feature_set": fset,
                                 "model": name, "kind": "learned" if learned else "baseline",
                                 "n_train": None if train is None else len(train), **m})

                for fs, ps in preds.items():
                    for name, p in ps.items():
                        log(name, fs, p, True)
                for name, p in extra.items():
                    if name.startswith("rule_") and kind == "clf":
                        log(name, "rule", p, False)
                    elif name.startswith("rule_"):
                        log(name, "rule", p, False)
                    else:
                        log(name, "baseline", p, False)

                names = CLF_MODELS if kind == "clf" else REG_MODELS
                rule_key = next((k for k in extra if k.startswith("rule_")), None)
                for name in names:
                    comps = [("planning_same_algo", preds["planning"][name])]
                    if rule_key:
                        comps.append((rule_key, extra[rule_key]))
                    elif kind == "reg":
                        comps.append(("train_median", extra["train_median"]))
                    for cname, cp_pred in comps:
                        if kind == "clf":
                            fn = lambda r, a=preds["progress"][name], b=cp_pred: (
                                fast_auc(y[r].astype(int), a[r]) - fast_auc(y[r].astype(int), b[r]))
                            metric = "delta_auc"
                        else:
                            fn = lambda r, a=preds["progress"][name], b=cp_pred: float(
                                np.mean(np.abs(y[r] - a[r])) - np.mean(np.abs(y[r] - b[r])))
                            metric = "delta_mae"
                        pt, lo, hi = cluster_boot(test, fn, rng)
                        boot_rows.append({"task": key, "checkpoint": cp, "protocol": protocol,
                                          "model": name, "versus": cname, "metric": metric,
                                          "delta": pt, "ci_low": lo, "ci_high": hi,
                                          "ci_excludes_zero": bool(lo > 0 or hi < 0)})
            print(f"done {key} @ {cp}", flush=True)

    res, boot = pd.DataFrame(rows), pd.DataFrame(boot_rows)
    res.to_csv(phase3.RESULTS_DIR / "checkpoint_model_results.csv", index=False)
    boot.to_csv(phase3.RESULTS_DIR / "checkpoint_model_bootstrap.csv", index=False)
    summary = {"n_sprint_checkpoints": int(len(frame)), "n_bootstrap_resamples": N_BOOT,
               "checkpoints": list(CHECKPOINTS)}
    (phase3.RESULTS_DIR / "checkpoint_summary.json").write_text(json.dumps(summary, indent=2))
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 500)
    best = res[(res.kind == "learned")]
    print(best[best.task == "late_closure"].pivot_table(
        index=["protocol", "checkpoint"], columns=["feature_set", "model"], values="roc_auc").round(3))


if __name__ == "__main__":
    main()
