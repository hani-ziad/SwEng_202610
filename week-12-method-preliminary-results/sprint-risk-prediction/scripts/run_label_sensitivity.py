"""
Post-review addition: label-definition sensitivity analysis.

A sensitivity analysis is included around the
delay/spillover threshold choices, and building this script is what
surfaced a real discrepancy between the analysis's originally-stated
spillover definition (delivered story points < 80% of committed story
points) and what src/labels.py actually computes (see the correction note
in the analysis's Method/Labels subsection). This script does NOT modify
labels.py -- the existing, already-validated pipeline stays untouched --
it reimplements each alternative label definition locally, purely for this
sensitivity check, and reruns the same random-forest model (the analysis's
best-F1 model on both labels) end to end for each variant.

Three things are varied:
  1. Delay tolerance: 0 / 1 (paper's choice) / 3 / 7 days late.
  2. Spillover "incomplete issue count" threshold: >=1 (paper's choice,
     i.e. "any" incomplete issue) / >=2 / >=3 incomplete issues.
  3. Spillover under the ORIGINALLY INTENDED definition (story-point
     ratio < 0.8), as a robustness check against a materially different
     operationalization, not just a nearby threshold.

Usage:
    python -m scripts.run_label_sensitivity
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_baseline import time_ordered_split
from src.baselines import make_random_forest
from src.data.load_tawos import load_reconstructed_sprints
from src.evaluate import evaluate
from src.features import ALL_FEATURES, add_features
from src.labels import add_labels

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

DELAY_TOLERANCES_DAYS = [0, 1, 3, 7]
SPILLOVER_MIN_INCOMPLETE = [1, 2, 3]


def _fit_eval(featured: pd.DataFrame, label_series: pd.Series, label_name: str) -> dict:
    """
    featured: the standard, unmodified add_features() output (so every
    planning-time/historical feature is identical to the main pipeline;
    only the LABEL varies across sensitivity runs).
    label_series: alternative label values, indexed like `featured`.
    """
    frame = featured.copy()
    frame[label_name] = label_series.reindex(frame.index).astype(int)
    frame = frame.dropna(subset=["prior_sprint_completion_ratio"]).reset_index(drop=True)

    train, test = time_ordered_split(frame)
    model = make_random_forest()
    model.fit(train[ALL_FEATURES], train[label_name])
    proba = model.predict_proba(test[ALL_FEATURES])[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = evaluate(test[label_name], pred, proba)
    metrics["prevalence_full_data"] = float(frame[label_name].mean())
    metrics["n_test"] = len(test)
    return metrics


def main() -> None:
    sprints = load_reconstructed_sprints()
    # Standard label/feature pipeline, unmodified -- this gives us the
    # correct "completion_ratio" and "delivered_story_points" columns that
    # add_features()'s historical rolling stats depend on, plus the exact
    # same planning-time/historical features used everywhere else in the
    # paper. Only the target column is swapped out per sensitivity variant
    # below; the feature values themselves never change.
    labeled = add_labels(sprints)
    featured = add_features(labeled)

    delay_days = (featured["complete_date"] - featured["end_date"]).dt.total_seconds() / 86400.0
    safe_denom_sp = featured["committed_story_points"].replace(0, np.nan)
    completion_ratio_sp = featured["delivered_story_points"] / safe_denom_sp

    results = {"delay_tolerance_days": {}, "spillover_min_incomplete": {}, "spillover_story_point_ratio": {}}

    # 1. Delay tolerance sensitivity.
    for tol in DELAY_TOLERANCES_DAYS:
        label = (delay_days > tol).astype(int)
        results["delay_tolerance_days"][str(tol)] = _fit_eval(featured, label, "label_delay")

    # 2. Spillover "at least K incomplete issues" sensitivity.
    for k in SPILLOVER_MIN_INCOMPLETE:
        label = (featured["not_completed_count"] >= k).astype(int)
        results["spillover_min_incomplete"][str(k)] = _fit_eval(featured, label, "label_spillover")

    # 3. Spillover under the originally-intended story-point-ratio < 0.8 definition.
    sp_label = (completion_ratio_sp < 0.8).fillna(False).astype(int)
    results["spillover_story_point_ratio"]["lt_0.8"] = _fit_eval(featured, sp_label, "label_spillover")

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "label_sensitivity.json"
    out_path.write_text(json.dumps(results, indent=2, default=float))
    print(f"Wrote {out_path}\n")

    print("=== Delay tolerance sensitivity (random forest, time-ordered) ===")
    for tol, m in results["delay_tolerance_days"].items():
        print(f"  tolerance={tol:>2s}d  prevalence={m['prevalence_full_data']:.3f}  F1={m['f1']:.3f}  ROC-AUC={m['roc_auc']:.3f}")

    print("\n=== Spillover 'at least K incomplete issues' sensitivity ===")
    for k, m in results["spillover_min_incomplete"].items():
        print(f"  K>={k}  prevalence={m['prevalence_full_data']:.3f}  F1={m['f1']:.3f}  ROC-AUC={m['roc_auc']:.3f}")

    print("\n=== Spillover under original story-point-ratio<0.8 definition ===")
    m = results["spillover_story_point_ratio"]["lt_0.8"]
    print(f"  prevalence={m['prevalence_full_data']:.3f}  F1={m['f1']:.3f}  ROC-AUC={m['roc_auc']:.3f}")


if __name__ == "__main__":
    main()
