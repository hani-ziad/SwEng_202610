"""Experiment 6 (added 8 Nov 2026): the corrected labels scored within each project.

Pooled AUC also rewards telling projects apart (the CCECE audit found late closure at .751
pooled but .634 within a project). This reruns experiment 5's fixed time split for both
corrected labels and scores every model and rule inside each project, then averages over
the projects that have both classes in the test period. The interval for gradient boosting
and for its gap to the past-history rule resamples sprints inside each project (2,000 times).

    python -m experiments.e6_within_project        (about 2 minutes)
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import phase4
import phase2.data as p2d
from experiments.e5_relabelled import TASKS, evaluate
from phase4.relabel import load_relabelled_frame, population

N_BOOT = 2000
RULE = {"late": "Past lateness (rule)", "carry": "Past carry share (rule)"}
SHOW = ["Gradient boosting", "Random forest", "Logistic regression", "Committed count (rule)"]


def per_project(y, s, proj):
    return {p: roc_auc_score(y[proj == p], s[proj == p]) for p in np.unique(proj)
            if 0 < y[proj == p].sum() < (proj == p).sum()}


def main():
    d = load_relabelled_frame()
    rows, gaps = [], []
    for task, target in TASKS.items():
        tr, te = p2d.time_ordered_split(population(d, task))
        _, scores = evaluate(task, tr, te, target)
        y, proj = te[target].to_numpy(int), te["project"].to_numpy()
        for m in SHOW + [RULE[task]]:
            s = np.asarray(scores[m], float)
            per = per_project(y, s, proj)
            rows.append({"task": task, "model": m, "n_test": len(y), "n_pos": int(y.sum()),
                         "pooled_auc": round(roc_auc_score(y, s), 3),
                         "within_auc": round(float(np.mean(list(per.values()))), 3), "projects": len(per),
                         **{f"auc_{p}": round(v, 3) for p, v in per.items()}})
        rng = np.random.default_rng(42)
        a, b = np.asarray(scores["Gradient boosting"], float), np.asarray(scores[RULE[task]], float)
        groups = [np.flatnonzero(proj == p) for p in np.unique(proj)]
        gb, diff = [], []
        for _ in range(N_BOOT):
            ga, dd = [], []
            for g in groups:
                i = g[rng.integers(0, len(g), len(g))]
                if 0 < y[i].sum() < len(i):
                    x = roc_auc_score(y[i], a[i]); ga.append(x); dd.append(x - roc_auc_score(y[i], b[i]))
            gb.append(np.mean(ga)); diff.append(np.mean(dd))
        w = {r["model"]: r["within_auc"] for r in rows if r["task"] == task}
        gaps.append({"task": task, "gb_within": w["Gradient boosting"],
                     "gb_within_lo": round(float(np.percentile(gb, 2.5)), 3),
                     "gb_within_hi": round(float(np.percentile(gb, 97.5)), 3),
                     "rule": RULE[task], "rule_within": w[RULE[task]],
                     "gap": round(w["Gradient boosting"] - w[RULE[task]], 3),
                     "gap_lo": round(float(np.percentile(diff, 2.5)), 3),
                     "gap_hi": round(float(np.percentile(diff, 97.5)), 3)})
        print(task, "done", flush=True)
    pd.DataFrame(rows).to_csv(phase4.RESULTS / "e6_within_project.csv", index=False)
    pd.DataFrame(gaps).to_csv(phase4.RESULTS / "e6_within_project_gaps.csv", index=False)
    print(pd.DataFrame(rows)[["task", "model", "pooled_auc", "within_auc", "projects"]].to_string(index=False))
    print(json.dumps(gaps, indent=1))


if __name__ == "__main__":
    main()
