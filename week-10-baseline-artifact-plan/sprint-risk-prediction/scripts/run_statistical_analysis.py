"""
Week 14 checklist item: statistical treatment.

Reports bootstrap 95% confidence intervals for F1 and ROC-AUC (time-ordered
split, full feature set) for logreg and random_forest on both labels, plus
a paired bootstrap significance test for the F1 and ROC-AUC difference
between them -- rather than reporting the single-run point estimates in
results/baseline_results.json as if they were exact.

XGBoost is not included here: the paired bootstrap needs per-row predicted
probabilities on the test set, which requires re-training the model in this
environment, and XGBoost isn't installable in the original development environment. Its point
estimate (results/baseline_results.json) is trusted from the author's own
run; a rigorous CI for it would need to be produced on a machine that has
xgboost installed, using this same bootstrap_ci() function against its
saved test-set probabilities.

Usage:
    python -m scripts.run_statistical_analysis
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from scripts.run_baseline import time_ordered_split
from src.baselines import make_logreg, make_random_forest
from src.data.load_tawos import load_reconstructed_sprints
from src.features import ALL_FEATURES, add_features, build_model_frame
from src.labels import add_labels

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
LABELS = ["label_delay", "label_spillover"]
N_BOOTSTRAP = 2000
RNG_SEED = 42


def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray, n_boot: int, rng: np.random.Generator):
    """Percentile-method 95% CI for F1 and ROC-AUC via resampling test rows with replacement."""
    n = len(y_true)
    f1s, aucs = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, yp, ypr = y_true[idx], y_pred[idx], y_proba[idx]
        if len(np.unique(yt)) < 2:
            continue  # ROC-AUC undefined for a resample with only one class present
        f1s.append(f1_score(yt, yp, zero_division=0))
        aucs.append(roc_auc_score(yt, ypr))
    f1s, aucs = np.array(f1s), np.array(aucs)
    return {
        "f1_mean": float(f1s.mean()),
        "f1_ci95": [float(np.percentile(f1s, 2.5)), float(np.percentile(f1s, 97.5))],
        "roc_auc_mean": float(aucs.mean()),
        "roc_auc_ci95": [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))],
        "n_valid_resamples": int(len(f1s)),
    }


def paired_bootstrap_test(y_true: np.ndarray, predA, probaA, predB, probaB, n_boot: int, rng: np.random.Generator):
    """
    Paired bootstrap test for whether model B beats model A: resample the
    SAME row indices for both models each iteration (paired), compute
    metric_B - metric_A, and report the two-sided p-value as the fraction
    of resamples where the sign of the difference flips relative to the
    observed difference (the standard paired-bootstrap significance test).
    """
    n = len(y_true)
    f1_diffs, auc_diffs = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        f1_diffs.append(f1_score(yt, predB[idx], zero_division=0) - f1_score(yt, predA[idx], zero_division=0))
        auc_diffs.append(roc_auc_score(yt, probaB[idx]) - roc_auc_score(yt, probaA[idx]))
    f1_diffs, auc_diffs = np.array(f1_diffs), np.array(auc_diffs)

    observed_f1_diff = f1_score(y_true, predB, zero_division=0) - f1_score(y_true, predA, zero_division=0)
    observed_auc_diff = roc_auc_score(y_true, probaB) - roc_auc_score(y_true, probaA)

    def two_sided_p(diffs, observed):
        # Fraction of bootstrap diffs on the opposite side of zero from the
        # observed effect, doubled (two-sided), capped at 1.0.
        if observed >= 0:
            p = (diffs <= 0).mean()
        else:
            p = (diffs >= 0).mean()
        return float(min(1.0, 2 * p))

    return {
        "observed_f1_diff (B-A)": float(observed_f1_diff),
        "f1_diff_ci95": [float(np.percentile(f1_diffs, 2.5)), float(np.percentile(f1_diffs, 97.5))],
        "f1_p_value": two_sided_p(f1_diffs, observed_f1_diff),
        "observed_roc_auc_diff (B-A)": float(observed_auc_diff),
        "roc_auc_diff_ci95": [float(np.percentile(auc_diffs, 2.5)), float(np.percentile(auc_diffs, 97.5))],
        "roc_auc_p_value": two_sided_p(auc_diffs, observed_auc_diff),
    }


def main() -> None:
    sprints = load_reconstructed_sprints()
    labeled = add_labels(sprints)
    featured = add_features(labeled)

    all_results = {}
    rng = np.random.default_rng(RNG_SEED)

    for label_col in LABELS:
        frame = build_model_frame(featured, label_col)
        train, test = time_ordered_split(frame)
        y_test = test[label_col].to_numpy()

        preds, probas = {}, {}
        for name, factory in [("logreg", make_logreg), ("random_forest", make_random_forest)]:
            model = factory()
            model.fit(train[ALL_FEATURES], train[label_col])
            proba = model.predict_proba(test[ALL_FEATURES])[:, 1]
            probas[name] = proba
            preds[name] = (proba >= 0.5).astype(int)

        label_results = {"n_test": len(test)}
        for name in ["logreg", "random_forest"]:
            label_results[name] = bootstrap_ci(y_test, preds[name], probas[name], N_BOOTSTRAP, rng)

        label_results["random_forest_vs_logreg"] = paired_bootstrap_test(
            y_test, preds["logreg"], probas["logreg"], preds["random_forest"], probas["random_forest"], N_BOOTSTRAP, rng
        )
        all_results[label_col] = label_results

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "statistical_analysis.json"
    out_path.write_text(json.dumps(all_results, indent=2, default=float))
    print(f"Wrote {out_path}\n")

    for label_col in LABELS:
        print(f"=== {label_col} (n_test={all_results[label_col]['n_test']}, {N_BOOTSTRAP} bootstrap resamples) ===")
        for name in ["logreg", "random_forest"]:
            r = all_results[label_col][name]
            print(
                f"  {name:14s} F1={r['f1_mean']:.3f} 95% CI [{r['f1_ci95'][0]:.3f}, {r['f1_ci95'][1]:.3f}]  "
                f"ROC-AUC={r['roc_auc_mean']:.3f} 95% CI [{r['roc_auc_ci95'][0]:.3f}, {r['roc_auc_ci95'][1]:.3f}]"
            )
        cmp = all_results[label_col]["random_forest_vs_logreg"]
        print(
            f"  random_forest - logreg: F1 diff={cmp['observed_f1_diff (B-A)']:.3f} "
            f"(p={cmp['f1_p_value']:.4f}), ROC-AUC diff={cmp['observed_roc_auc_diff (B-A)']:.3f} "
            f"(p={cmp['roc_auc_p_value']:.4f})\n"
        )


if __name__ == "__main__":
    main()
