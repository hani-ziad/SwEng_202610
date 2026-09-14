"""
Week 14 checklist item: error / failure analysis.

Trains random_forest (the strongest model with per-row predictions
reproducible in this environment -- see docs/data_access_and_versioning.md
on XGBoost) on the time-ordered split for both labels, then characterizes
the false positives and false negatives on the test set: which projects
they cluster in, and how their feature values compare to correctly
classified sprints.

Note on this script's random forest vs. results/baseline_results.json's:
this analysis needs per-row predictions (which project each error falls
in, which specific sprints are most confidently wrong) that the
headline results file does not store, so this script fits its own
random forest in this environment rather than reusing baseline results.
As documented in run_ablation.py, RandomForestClassifier's exact output
can drift by a couple hundredths of F1 across scikit-learn versions
even with a fixed seed -- so the *exact* outcome counts here (e.g. the
precise number of delay false positives) can differ slightly from
Table 4/9's headline numbers, though the qualitative pattern (which
projects and which feature ranges the errors cluster in) is stable
across reruns.

Usage:
    python -m scripts.run_error_analysis
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from scipy.stats import mannwhitneyu

from scripts.run_baseline import time_ordered_split
from src.baselines import make_random_forest
from src.data.load_tawos import load_reconstructed_sprints
from src.features import ALL_FEATURES, add_features, build_model_frame
from src.labels import add_labels

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
LABELS = ["label_delay", "label_spillover"]
NUMERIC_COLS_TO_PROFILE = [
    "committed_issue_count",
    "committed_story_points",
    "sprint_length_days",
    "prior_avg_completion_ratio_3",
    "prior_sprint_carryover",
]


def classify_outcome(y_true: int, y_pred: int) -> str:
    if y_true == 1 and y_pred == 1:
        return "true_positive"
    if y_true == 0 and y_pred == 0:
        return "true_negative"
    if y_true == 0 and y_pred == 1:
        return "false_positive"
    return "false_negative"


def main() -> None:
    sprints = load_reconstructed_sprints()
    labeled = add_labels(sprints)
    featured = add_features(labeled)

    report = {}
    for label_col in LABELS:
        frame = build_model_frame(featured, label_col)
        train, test = time_ordered_split(frame)

        model = make_random_forest()
        model.fit(train[ALL_FEATURES], train[label_col])
        proba = model.predict_proba(test[ALL_FEATURES])[:, 1]
        pred = (proba >= 0.5).astype(int)

        test = test.copy()
        test["y_true"] = test[label_col].to_numpy()
        test["y_pred"] = pred
        test["y_proba"] = proba
        test["outcome"] = [classify_outcome(t, p) for t, p in zip(test["y_true"], test["y_pred"])]

        counts = test["outcome"].value_counts().to_dict()

        # Where do false positives/negatives cluster by project?
        fp_by_project = test[test["outcome"] == "false_positive"]["project"].value_counts().to_dict()
        fn_by_project = test[test["outcome"] == "false_negative"]["project"].value_counts().to_dict()
        total_by_project = test["project"].value_counts().to_dict()

        # Feature profile: mean feature values for each outcome group, so we
        # can see how false positives/negatives differ from true positives/negatives.
        profile = (
            test.groupby("outcome")[NUMERIC_COLS_TO_PROFILE].mean().round(2).to_dict(orient="index")
        )

        # Statistical test (added after external review, which correctly
        # pointed out that "false positives are statistically
        # indistinguishable from true positives" is a claim that needs an
        # actual test, not just a mean-value comparison): two-sided
        # Mann-Whitney U per profiled feature, false_positive vs.
        # true_positive rows. Non-parametric because these count/point
        # features are strongly right-skewed, not normally distributed.
        fp_rows = test[test["outcome"] == "false_positive"]
        tp_rows = test[test["outcome"] == "true_positive"]
        fp_vs_tp_tests = {}
        if len(fp_rows) >= 3 and len(tp_rows) >= 3:
            for col in NUMERIC_COLS_TO_PROFILE:
                try:
                    stat, p = mannwhitneyu(fp_rows[col], tp_rows[col], alternative="two-sided")
                    fp_vs_tp_tests[col] = {"u_stat": float(stat), "p_value": float(p)}
                except ValueError:
                    fp_vs_tp_tests[col] = {"u_stat": None, "p_value": None}

        # The most confidently WRONG predictions (highest proba on an actual
        # negative; lowest proba on an actual positive) -- the sprints the
        # model was most surprised by.
        most_confident_fp = (
            test[test["outcome"] == "false_positive"]
            .nlargest(5, "y_proba")[["sprint_id", "project", "y_proba"] + NUMERIC_COLS_TO_PROFILE]
            .to_dict(orient="records")
        )
        most_confident_fn = (
            test[test["outcome"] == "false_negative"]
            .nsmallest(5, "y_proba")[["sprint_id", "project", "y_proba"] + NUMERIC_COLS_TO_PROFILE]
            .to_dict(orient="records")
        )

        report[label_col] = {
            "n_test": len(test),
            "outcome_counts": counts,
            "false_positive_rate_by_project": {
                p: round(fp_by_project.get(p, 0) / total_by_project[p], 3) for p in total_by_project
            },
            "false_negative_rate_by_project": {
                p: round(fn_by_project.get(p, 0) / total_by_project[p], 3) for p in total_by_project
            },
            "feature_profile_by_outcome": profile,
            "false_positive_vs_true_positive_mannwhitneyu": fp_vs_tp_tests,
            "most_confident_false_positives": most_confident_fp,
            "most_confident_false_negatives": most_confident_fn,
        }

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "error_analysis.json"
    out_path.write_text(json.dumps(report, indent=2, default=float))
    print(f"Wrote {out_path}\n")

    for label_col in LABELS:
        r = report[label_col]
        print(f"=== {label_col} (n_test={r['n_test']}) ===")
        print("Outcome counts:", r["outcome_counts"])
        print("FP rate by project:", r["false_positive_rate_by_project"])
        print("FN rate by project:", r["false_negative_rate_by_project"])
        print("FP vs TP Mann-Whitney U p-values:", r["false_positive_vs_true_positive_mannwhitneyu"])
        print()


if __name__ == "__main__":
    main()
