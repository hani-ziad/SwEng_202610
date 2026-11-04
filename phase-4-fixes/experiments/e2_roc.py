"""Phase 4, experiment 2: ROC curves with uncertainty, one panel per problem.

Feedback: show ROC curves instead of the bar chart, one for late closure and one
for spillover, drawn as a range rather than a single line.

Figure 1 (figures/e2_roc_repeats.png): gradient boosting, mean ROC curve and a
+/- 1 SD band across the 5 repeats of experiment 1, for the shuffled 70/30
scheme and the time-ordered (leak-safe) scheme.

Figure 2 (figures/e2_roc_bootstrap.png): the paper's own split (each project's
first 70% of sprints train, last 30% test), with a 95% band from 2,000
bootstrap resamples of the test sprints. Scope-only LR is drawn as the baseline.

    python -m experiments.e2_roc      (run experiment 1 first)
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

import phase4  # noqa: E402
from phase2.data import time_ordered_split  # noqa: E402
from phase4.common import SEED, TASKS, fit_score, frame  # noqa: E402

GRID = np.linspace(0, 1, 201)
COL = {"A. Random 70/30 (shuffled)": "#c2410c", "C. Time-ordered (leak-safe)": "#1d4ed8",
       "Gradient boosting": "#1d4ed8", "Scope-only LR (baseline)": "#6b7280"}
N_BOOT = 2000


def interp_roc(y, s):
    fpr, tpr, _ = roc_curve(y, s)
    out = np.interp(GRID, fpr, tpr)
    out[0] = 0.0
    return out


def style(ax, title):
    ax.plot([0, 1], [0, 1], ls=":", c="#9ca3af", lw=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("False positive rate (1 - specificity)")
    ax.set_ylabel("True positive rate (sensitivity)")
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=8.5, frameon=False)


def fig_repeats():
    sc = pd.read_csv(phase4.RESULTS / "e1_gb_test_scores.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5))
    rows = []
    for ax, task in zip(axes, TASKS):
        for scheme in ["A. Random 70/30 (shuffled)", "C. Time-ordered (leak-safe)"]:
            curves, aucs = [], []
            for _, g in sc[(sc.task == task) & (sc.scheme == scheme)].groupby("repeat"):
                curves.append(interp_roc(g.y, g.score))
                aucs.append(roc_auc_score(g.y, g.score))
            m, sd = np.mean(curves, 0), np.std(curves, 0, ddof=1)
            c = COL[scheme]
            ax.plot(GRID, m, c=c, lw=2, label=f"{scheme.split('. ')[1]}: AUC {np.mean(aucs):.3f} ± {np.std(aucs, ddof=1):.3f}")
            ax.fill_between(GRID, np.clip(m - sd, 0, 1), np.clip(m + sd, 0, 1), color=c, alpha=0.18, lw=0)
            rows.append({"task": task, "scheme": scheme, "auc_mean": np.mean(aucs), "auc_sd": np.std(aucs, ddof=1)})
        style(ax, TASKS[task]["label"])
    fig.suptitle("Gradient boosting: mean ROC ± 1 SD over 5 repeats", fontsize=12)
    fig.tight_layout()
    fig.savefig(phase4.FIGURES / "e2_roc_repeats.png", dpi=200)
    return rows


def fig_bootstrap():
    rng = np.random.default_rng(SEED)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5))
    rows = []
    for ax, task in zip(axes, TASKS):
        d, target = frame(task), TASKS[task]["target"]
        tr, te = time_ordered_split(d)
        y = te[target].to_numpy(int)
        for model in ["Gradient boosting", "Scope-only LR (baseline)"]:
            s, _ = fit_score(model, tr, te, target)
            boots, baucs = [], []
            for _ in range(N_BOOT):
                i = rng.integers(0, len(y), len(y))
                if y[i].min() == y[i].max():
                    continue
                boots.append(interp_roc(y[i], s[i]))
                baucs.append(roc_auc_score(y[i], s[i]))
            lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
            auc = roc_auc_score(y, s)
            alo, ahi = np.percentile(baucs, [2.5, 97.5])
            c = COL[model]
            ax.plot(GRID, interp_roc(y, s), c=c, lw=2, label=f"{model}: AUC {auc:.3f} [{alo:.3f}, {ahi:.3f}]")
            ax.fill_between(GRID, lo, hi, color=c, alpha=0.18, lw=0)
            rows.append({"task": task, "model": model, "auc": auc, "ci_low": alo, "ci_high": ahi,
                         "n_test": len(y), "n_pos": int(y.sum())})
        style(ax, TASKS[task]["label"])
    fig.suptitle("Paper's time split (last 30% of each project): ROC with 95% bootstrap band", fontsize=12)
    fig.tight_layout()
    fig.savefig(phase4.FIGURES / "e2_roc_bootstrap.png", dpi=200)
    return rows


def main():
    phase4.FIGURES.mkdir(exist_ok=True)
    a = pd.DataFrame(fig_repeats())
    b = pd.DataFrame(fig_bootstrap())
    a.round(3).to_csv(phase4.RESULTS / "e2_roc_repeats_auc.csv", index=False)
    b.round(3).to_csv(phase4.RESULTS / "e2_roc_bootstrap_auc.csv", index=False)
    print(a.round(3).to_string(index=False))
    print(b.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
