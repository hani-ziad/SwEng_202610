"""Phase 4, experiment 1: repeat every split 5 times and report mean +/- SD.

Feedback: one 70/30 split gives one number with no spread. Shuffle, repeat,
and report the average with its standard deviation.

Three schemes, 5 repeats each, same models and features as Phase 2:

  A. Random 70/30   sprints shuffled with a different seed each time (the
                    reviewer's request). Optimistic: a model can be trained on
                    a team's later sprints and tested on its earlier ones.
  B. Random 5-fold CV, repeated 5 times with different shuffles.
  C. Time-ordered   the leak-safe version. For each project, sprints are kept
                    in time order; repeat k trains on the first 40/50/60/70/80%
                    and tests on the next 20%. Nothing from the future is used.

For every repeat: AUC, and sensitivity, specificity and F1 at a threshold fixed
on the training data only (Youden's J on out-of-fold training predictions).
Per-repeat test scores are saved for the ROC figure (experiment 2).

    python -m experiments.e1_repeated_splits      (from the phase-4-fixes folder)
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

import phase4
from phase4.common import MODELS, N_REPEATS, SEED, TASKS, decision_metrics, fit_score, frame


def random_7030(d, target, k):
    tr, te = train_test_split(d, test_size=0.3, stratify=d[target], random_state=SEED + k)
    return [(tr, te)]


def random_cv(d, target, k):
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED + k)
    return [(d.iloc[a], d.iloc[b]) for a, b in cv.split(d, d[target])]


TIME_CUTS = [0.4, 0.5, 0.6, 0.7, 0.8]


def time_rolling(d, target, k):
    """Per project: train on the first cut%, test on the following 20% (time order kept)."""
    lo, hi = TIME_CUTS[k], TIME_CUTS[k] + 0.2
    tr, te = [], []
    for _, g in d.groupby("project"):
        g = g.sort_values("start_date")
        n = len(g)
        a, b = int(round(n * lo)), int(round(n * hi))
        tr.append(g.iloc[:a])
        te.append(g.iloc[a:b])
    return [(pd.concat(tr), pd.concat(te))]


SCHEMES = {
    "A. Random 70/30 (shuffled)": random_7030,
    "B. Random 5-fold CV (shuffled)": random_cv,
    "C. Time-ordered (leak-safe)": time_rolling,
}


def main():
    phase4.RESULTS.mkdir(exist_ok=True)
    rows, roc = [], []
    for task, spec in TASKS.items():
        d, target = frame(task), spec["target"]
        for scheme, splitter in SCHEMES.items():
            for k in range(N_REPEATS):
                folds = splitter(d, target, k)
                for model in MODELS:
                    ys, ss, ps, thrs = [], [], [], []
                    for tr, te in folds:
                        sc, thr = fit_score(model, tr, te, target, seed=SEED + k)
                        ys.append(te[target].to_numpy(int))
                        ss.append(sc)
                        ps.append((sc >= thr).astype(int))
                        thrs.append(thr)
                    y, sc, pred = np.concatenate(ys), np.concatenate(ss), np.concatenate(ps)
                    # with CV each fold keeps its own training-set threshold; the 0/1 decisions are pooled
                    m = decision_metrics(y, sc, pred)
                    m["threshold"] = float(np.mean(thrs))
                    rows.append({"task": task, "scheme": scheme, "repeat": k + 1, "model": model, **m})
                    if model == "Gradient boosting":
                        roc.append(pd.DataFrame({"task": task, "scheme": scheme, "repeat": k + 1, "y": y, "score": sc}))
                print(task, scheme, "repeat", k + 1, "done", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(phase4.RESULTS / "e1_repeats.csv", index=False)
    pd.concat(roc).to_csv(phase4.RESULTS / "e1_gb_test_scores.csv", index=False)

    keys = ["auc", "sensitivity", "specificity", "f1"]
    summ = res.groupby(["task", "scheme", "model"])[keys].agg(["mean", "std"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ = summ.reset_index()
    summ.round(3).to_csv(phase4.RESULTS / "e1_summary.csv", index=False)
    pretty = summ.copy()
    for k in keys:
        pretty[k] = pretty.apply(lambda r: f"{r[k + '_mean']:.3f} ± {r[k + '_std']:.3f}", axis=1)
    pretty = pretty[["task", "scheme", "model"] + keys]
    pretty.to_csv(phase4.RESULTS / "e1_summary_pretty.csv", index=False)
    pd.set_option("display.width", 220)
    print(pretty.to_string(index=False))


if __name__ == "__main__":
    main()
