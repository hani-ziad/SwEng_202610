"""
Round-5 review addition: decompose the spillover task into the two
sub-problems a single all-sprint AUC conflates, and re-run spillover under
the zero-committed-scope sensitivity analysis the review asked for.

363 of the 2,002 modeled sprints (18.1%) have committed_issue_count == 0.
Because label_spillover is (not_completed_count > 0) OR (completion_ratio <
0.8), and committed_story_points is mechanically 0
whenever committed_issue_count is 0 (verified below and in
tests/test_scope_decomposition.py), every one of these 363 sprints is
spillover-negative BY CONSTRUCTION, not because a model predicted anything.
The original spillover numbers in results/baseline_results.json report
performance on the FULL population (all 2,002 sprints) and therefore partly
reward a model for recognizing this mechanically-negative subgroup.

This script reports two things separately, as the review asked:

  Task A -- "does this sprint have any committed issue at all?"
            label: has_committed_issue = (committed_issue_count > 0)
            population: all 2,002 modeled sprints (unchanged, cold-start
            filtered, same as every other result in this repository)
            features: ONLY sprint_length_days + the 4 historical features
            (committed_story_points is deliberately
            EXCLUDED here -- they are deterministic functions of
            committed_issue_count for this population, so including them
            would make Task A tautological rather than a genuine
            prediction problem; see the docstring above)

  Task B -- "given a sprint that committed at least one issue, does it
            spill over?" -- i.e. the zero-committed-scope sensitivity
            analysis: label_spillover, restricted to the 1,639 sprints with
            committed_issue_count > 0, using the SAME full feature set
            (ALL_FEATURES) as the main spillover model, so this is
            a like-for-like re-run under a narrower population, not a
            different model.

Both tasks use the same time-ordered per-project split and LOPO scheme as
every other result in this repository, evaluated with the same
src.evaluate.evaluate() metrics, plus bootstrap 95% CIs (percentile method,
2,000 resamples) for F1, ROC-AUC, PR-AUC, and Brier score on the
time-ordered split.

XGBoost is not included: it is not installable in this reproduction
development environment (see requirements-lock.txt and the analysis's Implementation and
Reproducibility section), and re-running it here would require refitting
on a population it was never evaluated on in the original run, which this
original development environment could not do. Every number in this script's output comes from
heuristic / logreg / random_forest, refit and re-evaluated in this
environment -- nothing here is copied from baseline_results.json's
XGBoost point estimates.

Usage:
    python -m scripts.run_scope_decomposition
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

from scripts.run_baseline import time_ordered_split
from src.baselines import heuristic_predict, make_logreg, make_random_forest
from src.data.load_tawos import load_reconstructed_sprints
from src.evaluate import evaluate
from src.features import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    HISTORICAL_FEATURES,
    add_features,
    build_model_frame,
)
from src.labels import add_labels

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
N_BOOTSTRAP = 2000
RNG_SEED = 42

# Task A deliberately excludes committed_story_points: it is deterministically
# 0 whenever committed_issue_count == 0, so including it would let a model
# solve Task A partly by definition rather than by genuine signal.
TASK_A_FEATURES = HISTORICAL_FEATURES + ["sprint_length_days"]
TASK_A_CATEGORICAL = CATEGORICAL_FEATURES


def _make_preprocessor(numeric_features, categorical_features) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )


def _make_logreg(numeric_features, categorical_features) -> Pipeline:
    return Pipeline(
        steps=[
            ("prep", _make_preprocessor(numeric_features, categorical_features)),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )


def _make_rf(numeric_features, categorical_features) -> Pipeline:
    return Pipeline(
        steps=[
            ("prep", _make_preprocessor(numeric_features, categorical_features)),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=5,
                    min_samples_leaf=3,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def bootstrap_ci_multi(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray, n_boot: int, rng: np.random.Generator) -> dict:
    """Percentile-method 95% CIs for F1, ROC-AUC, PR-AUC, and Brier score."""
    from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, brier_score_loss

    n = len(y_true)
    f1s, aucs, praucs, briers = [], [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, yp, ypr = y_true[idx], y_pred[idx], y_proba[idx]
        if len(np.unique(yt)) < 2:
            continue
        f1s.append(f1_score(yt, yp, zero_division=0))
        aucs.append(roc_auc_score(yt, ypr))
        praucs.append(average_precision_score(yt, ypr))
        briers.append(brier_score_loss(yt, ypr))
    out = {}
    for name, vals in [("f1", f1s), ("roc_auc", aucs), ("pr_auc", praucs), ("brier", briers)]:
        arr = np.array(vals)
        out[f"{name}_mean"] = float(arr.mean()) if len(arr) else float("nan")
        out[f"{name}_ci95"] = (
            [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))] if len(arr) else [float("nan"), float("nan")]
        )
    out["n_valid_resamples"] = int(len(f1s))
    return out


def run_time_ordered(train: pd.DataFrame, test: pd.DataFrame, label_col: str, numeric_features, categorical_features, rng: np.random.Generator) -> dict:
    row = {}
    row["heuristic"] = evaluate(test[label_col], heuristic_predict(test))

    feat_cols = numeric_features + categorical_features
    for name, factory in [("logreg", _make_logreg), ("random_forest", _make_rf)]:
        model = factory(numeric_features, categorical_features)
        model.fit(train[feat_cols], train[label_col])
        proba = model.predict_proba(test[feat_cols])[:, 1]
        pred = (proba >= 0.5).astype(int)
        m = evaluate(test[label_col], pred, proba)
        m["bootstrap_ci95"] = bootstrap_ci_multi(
            test[label_col].to_numpy(), pred, proba, N_BOOTSTRAP, rng
        )
        row[name] = m
    return row


def run_lopo(frame: pd.DataFrame, label_col: str, numeric_features, categorical_features) -> dict:
    feat_cols = numeric_features + categorical_features
    lopo = {}
    for held_out in frame["project"].unique():
        train = frame[frame["project"] != held_out]
        test = frame[frame["project"] == held_out]
        if len(train) == 0 or len(test) == 0 or test[label_col].nunique() < 1:
            continue
        row = {}
        row["heuristic"] = evaluate(test[label_col], heuristic_predict(test))
        for name, factory in [("logreg", _make_logreg), ("random_forest", _make_rf)]:
            model = factory(numeric_features, categorical_features)
            model.fit(train[feat_cols], train[label_col])
            proba = model.predict_proba(test[feat_cols])[:, 1]
            pred = (proba >= 0.5).astype(int)
            row[name] = evaluate(test[label_col], pred, proba)
        lopo[held_out] = row
    return lopo


def main() -> None:
    sprints = load_reconstructed_sprints()
    labeled = add_labels(sprints)
    featured = add_features(labeled)

    full_frame = build_model_frame(featured, "label_spillover")
    n_total = len(full_frame)
    n_zero_scope = int((full_frame["committed_issue_count"] == 0).sum())

    # Sanity check the premise stated in this script's docstring, rather
    # than assuming it: fail loudly if a future data/feature change makes
    # committed_story_points non-degenerate at
    # committed_issue_count == 0 (which would break Task A's rationale for
    # excluding them).
    zero_rows = full_frame[full_frame["committed_issue_count"] == 0]
    assert (zero_rows["committed_story_points"] == 0).all(), (
        "committed_story_points is no longer always 0 when "
        "committed_issue_count == 0 -- re-examine whether Task A should "
        "still exclude it."
    )
    assert zero_rows["label_spillover"].eq(0).all(), (
        "Not every committed_issue_count == 0 sprint is spillover-negative "
        "-- the 'spillover-negative by construction' claim needs revising."
    )

    rng = np.random.default_rng(RNG_SEED)

    print(f"Total modeled sprints: {n_total}; zero-committed-scope sprints: {n_zero_scope} "
          f"({100 * n_zero_scope / n_total:.1f}%)\n")

    # ---- Task A: has_committed_issue, full population ----
    task_a_frame = full_frame.copy()
    task_a_frame["has_committed_issue"] = (task_a_frame["committed_issue_count"] > 0).astype(int)
    train_a, test_a = time_ordered_split(task_a_frame)
    task_a_time_ordered = run_time_ordered(
        train_a, test_a, "has_committed_issue", TASK_A_FEATURES, TASK_A_CATEGORICAL, rng
    )
    task_a_lopo = run_lopo(task_a_frame, "has_committed_issue", TASK_A_FEATURES, TASK_A_CATEGORICAL)

    print("=== Task A: predicting has_committed_issue (n={}, positive rate={:.3f}) ===".format(
        len(task_a_frame), task_a_frame["has_committed_issue"].mean()
    ))
    for name, m in task_a_time_ordered.items():
        print(f"  {name:15s} f1={m['f1']:.3f}  roc_auc={m['roc_auc']:.3f}  pr_auc={m['pr_auc']:.3f}")

    # ---- Task B: label_spillover, restricted to committed_issue_count > 0 ----
    nonempty_frame = full_frame[full_frame["committed_issue_count"] > 0].reset_index(drop=True)
    train_b, test_b = time_ordered_split(nonempty_frame)
    task_b_time_ordered = run_time_ordered(
        train_b, test_b, "label_spillover", ALL_FEATURES[:-1], CATEGORICAL_FEATURES, rng
    )
    task_b_lopo = run_lopo(nonempty_frame, "label_spillover", ALL_FEATURES[:-1], CATEGORICAL_FEATURES)

    print("\n=== Task B: predicting label_spillover among committed_issue_count > 0 sprints "
          "(n={}, positive rate={:.3f}) ===".format(len(nonempty_frame), nonempty_frame["label_spillover"].mean()))
    for name, m in task_b_time_ordered.items():
        print(f"  {name:15s} f1={m['f1']:.3f}  roc_auc={m['roc_auc']:.3f}  pr_auc={m['pr_auc']:.3f}")

    # ---- For comparison: the ORIGINAL full-population spillover numbers ----
    # (re-derived here, not copied, so this script is self-contained)
    train_full, test_full = time_ordered_split(full_frame)
    full_scope_time_ordered = run_time_ordered(
        train_full, test_full, "label_spillover", ALL_FEATURES[:-1], CATEGORICAL_FEATURES, rng
    )
    print("\n=== For comparison: original full-population spillover (n={}, positive rate={:.3f}) ===".format(
        len(full_frame), full_frame["label_spillover"].mean()
    ))
    for name, m in full_scope_time_ordered.items():
        print(f"  {name:15s} f1={m['f1']:.3f}  roc_auc={m['roc_auc']:.3f}  pr_auc={m['pr_auc']:.3f}")

    payload = {
        "note": (
            "Round-5 review addition. Task A = predicting has_committed_issue "
            "(full population, n={n_total}); Task B = predicting label_spillover "
            "restricted to committed_issue_count > 0 (n={n_nonempty}) -- this is "
            "also the zero-committed-scope sensitivity analysis. "
            "full_population_spillover_for_comparison is the original "
            "(unfiltered) spillover result, re-derived here for a like-for-like "
            "comparison. XGBoost is excluded throughout (not installable in this "
            "development environment); every number here is heuristic/logreg/random_forest, "
            "freshly refit in this environment."
        ).format(n_total=n_total, n_nonempty=len(nonempty_frame)),
        "n_total_modeled": n_total,
        "n_zero_committed_scope": n_zero_scope,
        "pct_zero_committed_scope": round(100 * n_zero_scope / n_total, 1),
        "n_nonempty_committed_scope": len(nonempty_frame),
        "task_a_has_committed_issue": {
            "features": TASK_A_FEATURES + TASK_A_CATEGORICAL,
            "n": len(task_a_frame),
            "positive_rate": float(task_a_frame["has_committed_issue"].mean()),
            "time_ordered": task_a_time_ordered,
            "leave_one_project_out": task_a_lopo,
        },
        "task_b_conditional_spillover": {
            "features": ALL_FEATURES,
            "n": len(nonempty_frame),
            "positive_rate": float(nonempty_frame["label_spillover"].mean()),
            "time_ordered": task_b_time_ordered,
            "leave_one_project_out": task_b_lopo,
        },
        "full_population_spillover_for_comparison": {
            "features": ALL_FEATURES,
            "n": len(full_frame),
            "positive_rate": float(full_frame["label_spillover"].mean()),
            "time_ordered": full_scope_time_ordered,
        },
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "scope_decomposition.json"
    out_path.write_text(json.dumps(payload, indent=2, default=float))
    print(f"\nWrote results to {out_path}")

    # Per-project LOPO summary CSV for Task B (the sensitivity analysis'
    # "per-project results" requirement).
    rows = []
    for held_out, per_model in task_b_lopo.items():
        for model_name, m in per_model.items():
            rows.append({
                "project": held_out,
                "model": model_name,
                "n_test": m["n"],
                "positive_rate": round(m["positive_rate"], 3),
                "f1": round(m["f1"], 3),
                "roc_auc": round(m["roc_auc"], 3) if m["roc_auc"] == m["roc_auc"] else None,
                "pr_auc": round(m["pr_auc"], 3) if m["pr_auc"] == m["pr_auc"] else None,
                "brier": round(m["brier"], 3) if m["brier"] == m["brier"] else None,
            })
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "scope_decomposition_task_b_per_project.csv", index=False)
    print(f"Wrote per-project Task B summary to {RESULTS_DIR / 'scope_decomposition_task_b_per_project.csv'}")


if __name__ == "__main__":
    main()
