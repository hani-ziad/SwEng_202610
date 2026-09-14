"""
Post-review addition: project-level cluster bootstrap.

A methodological check raised a pseudoreplication
concern about results/statistical_analysis.json's existing bootstrap: it
resamples individual TEST ROWS with replacement, treating each sprint as an
independent observation. But sprints are nested within projects, and
sprints from the same project are plausibly dependent -- they share a
team, a codebase, a general velocity level, and (per Section sec:importance
of the analysis) project identity itself is delay's single strongest feature.
Row-level resampling can therefore understate the true uncertainty: a
resample that happens to draw more rows from one project isn't actually an
independent "new sample" the way row-level bootstrap theory assumes.

This script instead resamples the 8 PROJECTS with replacement (so a
resample might contain a project 0, 2, or 3 times, and omit others
entirely), assembling each bootstrap replicate's test set from every row
belonging to each drawn project. This is the standard cluster bootstrap
for data with a natural grouping structure, and is a strictly more
conservative (wider) interval than the row-level version whenever
within-cluster correlation exists.

Same models (logreg, random_forest), same time-ordered split, same fixed
0.5 threshold as the rest of the analysis (Section sec:tuning) -- only the
resampling unit changes. XGBoost is excluded for the same reason as
run_statistical_analysis.py (not installable in the original development environment).

Usage:
    python -m scripts.run_cluster_bootstrap
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


def cluster_bootstrap_ci(test: pd.DataFrame, label_col: str, pred_col: str, proba_col: str, n_boot: int, rng: np.random.Generator):
    projects = test["project"].unique()
    n_projects = len(projects)
    # Pre-split test rows by project once, for fast resampling.
    by_project = {p: test[test["project"] == p] for p in projects}

    f1s, aucs, sizes = [], [], []
    for _ in range(n_boot):
        drawn = rng.choice(projects, size=n_projects, replace=True)
        resample = pd.concat([by_project[p] for p in drawn], ignore_index=True)
        yt = resample[label_col].to_numpy()
        if len(np.unique(yt)) < 2:
            continue
        f1s.append(f1_score(yt, resample[pred_col].to_numpy(), zero_division=0))
        aucs.append(roc_auc_score(yt, resample[proba_col].to_numpy()))
        sizes.append(len(resample))
    f1s, aucs = np.array(f1s), np.array(aucs)
    return {
        "n_projects": int(n_projects),
        "f1_mean": float(f1s.mean()),
        "f1_ci95": [float(np.percentile(f1s, 2.5)), float(np.percentile(f1s, 97.5))],
        "roc_auc_mean": float(aucs.mean()),
        "roc_auc_ci95": [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))],
        "n_valid_resamples": int(len(f1s)),
        "mean_resample_size": float(np.mean(sizes)) if sizes else None,
    }


def cluster_paired_test(test: pd.DataFrame, label_col: str, predA_col: str, probaA_col: str, predB_col: str, probaB_col: str, n_boot: int, rng: np.random.Generator):
    projects = test["project"].unique()
    n_projects = len(projects)
    by_project = {p: test[test["project"] == p] for p in projects}

    f1_diffs, auc_diffs = [], []
    for _ in range(n_boot):
        drawn = rng.choice(projects, size=n_projects, replace=True)
        resample = pd.concat([by_project[p] for p in drawn], ignore_index=True)
        yt = resample[label_col].to_numpy()
        if len(np.unique(yt)) < 2:
            continue
        f1_diffs.append(
            f1_score(yt, resample[predB_col].to_numpy(), zero_division=0)
            - f1_score(yt, resample[predA_col].to_numpy(), zero_division=0)
        )
        auc_diffs.append(
            roc_auc_score(yt, resample[probaB_col].to_numpy())
            - roc_auc_score(yt, resample[probaA_col].to_numpy())
        )
    f1_diffs, auc_diffs = np.array(f1_diffs), np.array(auc_diffs)

    observed_f1_diff = (
        f1_score(test[label_col], test[predB_col], zero_division=0)
        - f1_score(test[label_col], test[predA_col], zero_division=0)
    )
    observed_auc_diff = roc_auc_score(test[label_col], test[probaB_col]) - roc_auc_score(
        test[label_col], test[probaA_col]
    )

    def two_sided_p(diffs, observed):
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
        test = test.copy()

        for name, factory in [("logreg", make_logreg), ("random_forest", make_random_forest)]:
            model = factory()
            model.fit(train[ALL_FEATURES], train[label_col])
            proba = model.predict_proba(test[ALL_FEATURES])[:, 1]
            test[f"{name}_proba"] = proba
            test[f"{name}_pred"] = (proba >= 0.5).astype(int)

        label_results = {"n_test": len(test), "n_projects": int(test["project"].nunique())}
        for name in ["logreg", "random_forest"]:
            label_results[name] = cluster_bootstrap_ci(
                test, label_col, f"{name}_pred", f"{name}_proba", N_BOOTSTRAP, rng
            )

        label_results["random_forest_vs_logreg"] = cluster_paired_test(
            test, label_col, "logreg_pred", "logreg_proba", "random_forest_pred", "random_forest_proba", N_BOOTSTRAP, rng
        )
        all_results[label_col] = label_results

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "cluster_bootstrap.json"
    out_path.write_text(json.dumps(all_results, indent=2, default=float))
    print(f"Wrote {out_path}\n")

    for label_col in LABELS:
        r = all_results[label_col]
        print(f"=== {label_col} (n_test={r['n_test']}, {r['n_projects']} projects, {N_BOOTSTRAP} cluster resamples) ===")
        for name in ["logreg", "random_forest"]:
            m = r[name]
            print(
                f"  {name:14s} F1={m['f1_mean']:.3f} 95% CI [{m['f1_ci95'][0]:.3f}, {m['f1_ci95'][1]:.3f}]  "
                f"ROC-AUC={m['roc_auc_mean']:.3f} 95% CI [{m['roc_auc_ci95'][0]:.3f}, {m['roc_auc_ci95'][1]:.3f}]"
            )
        cmp = r["random_forest_vs_logreg"]
        print(
            f"  random_forest - logreg: F1 diff={cmp['observed_f1_diff (B-A)']:.3f} "
            f"(p={cmp['f1_p_value']:.4f}), ROC-AUC diff={cmp['observed_roc_auc_diff (B-A)']:.3f} "
            f"(p={cmp['roc_auc_p_value']:.4f})\n"
        )


if __name__ == "__main__":
    main()
