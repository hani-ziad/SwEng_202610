"""
Post-review addition: leave-one-project-out WITHOUT the project-identity
feature, as the primary cross-project generalization comparator.

A methodological check identified an important subtlety in
results/baseline_results.json's existing leave-one-project-out (LOPO) run:
that run includes `project` (one-hot encoded) in the feature set. For a
held-out project never seen during training, OneHotEncoder(handle_unknown=
"ignore") makes that project's own one-hot column all zeros at test time --
so the model is scored using a project-identity feature that is, for every
single held-out row, uninformative by construction (it can't recognize the
project it has never seen). Interpreting the original LOPO numbers as "how
well does project identity transfer" is therefore muddled: part of what
looks like a "planning-time features generalize" story is actually "the
project dummy contributes nothing here, one way or the other."

This script reruns LOPO using ONLY the 8 planning-time + historical
features (src.features.PLANNING_TIME_FEATURES + HISTORICAL_FEATURES),
with no project feature at all, for logreg and random_forest. This is a
cleaner, more defensible primary comparator for cross-project
generalization: every feature used is, in principle, computable for a
genuinely new project, unlike a project dummy that is either absent
(if a new project's own value is unseen) or leaking training-set project
identity (if it is present and the training procedure implicitly memorizes
project-level base rates via non-project features correlated with it).

The original with-project LOPO numbers (results/baseline_results.json)
are retained in the analysis as a secondary comparison, not replaced --
comparing the two isolates how much of the original LOPO signal was
coming from features available in principle to a genuinely new project.

XGBoost is excluded for the same reason as elsewhere in this repo (not
installable in the original development environment).

Usage:
    python -m scripts.run_lopo_no_project
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.load_tawos import load_reconstructed_sprints
from src.evaluate import evaluate
from src.features import HISTORICAL_FEATURES, PLANNING_TIME_FEATURES, add_features
from src.labels import add_labels

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
LABELS = ["label_delay", "label_spillover"]
NO_PROJECT_FEATURES = PLANNING_TIME_FEATURES + HISTORICAL_FEATURES


def _model(name: str) -> Pipeline:
    prep = StandardScaler()
    if name == "logreg":
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    elif name == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=300, max_depth=5, min_samples_leaf=3, class_weight="balanced", random_state=42
        )
    else:
        raise ValueError(name)
    return Pipeline(steps=[("prep", prep), ("clf", clf)])


def main() -> None:
    sprints = load_reconstructed_sprints()
    labeled = add_labels(sprints)
    featured = add_features(labeled)
    featured = featured.dropna(subset=["prior_sprint_completion_ratio"]).reset_index(drop=True)

    all_results = {}
    for label_col in LABELS:
        frame = featured[list(dict.fromkeys(NO_PROJECT_FEATURES + [label_col, "project", "sprint_id", "start_date"]))]

        lopo = {}
        for held_out in frame["project"].unique():
            train = frame[frame["project"] != held_out]
            test = frame[frame["project"] == held_out]
            if len(train) == 0 or len(test) == 0 or test[label_col].nunique() < 1:
                continue
            row = {}
            for model_name in ["logreg", "random_forest"]:
                model = _model(model_name)
                model.fit(train[NO_PROJECT_FEATURES], train[label_col])
                proba = model.predict_proba(test[NO_PROJECT_FEATURES])[:, 1]
                pred = (proba >= 0.5).astype(int)
                row[model_name] = evaluate(test[label_col], pred, proba)
            lopo[held_out] = row
        all_results[label_col] = lopo

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "lopo_no_project.json"
    out_path.write_text(json.dumps(all_results, indent=2, default=float))
    print(f"Wrote {out_path}\n")

    # Compare mean ROC-AUC per label against the WITH-project LOPO already
    # in baseline_results.json, for logreg and random_forest.
    baseline_path = RESULTS_DIR / "baseline_results.json"
    baseline = json.loads(baseline_path.read_text()) if baseline_path.exists() else None

    for label_col in LABELS:
        print(f"=== {label_col}: LOPO without project identity ===")
        rows = []
        for model_name in ["logreg", "random_forest"]:
            aucs = [all_results[label_col][p][model_name]["roc_auc"] for p in all_results[label_col]]
            f1s = [all_results[label_col][p][model_name]["f1"] for p in all_results[label_col]]
            mean_auc_noproj = sum(aucs) / len(aucs)
            mean_f1_noproj = sum(f1s) / len(f1s)
            rows.append((model_name, mean_f1_noproj, mean_auc_noproj, min(aucs), max(aucs)))

            if baseline is not None:
                with_proj = baseline[label_col]["leave_one_project_out"]
                aucs_wp = [with_proj[p][model_name]["roc_auc"] for p in with_proj if model_name in with_proj[p]]
                f1s_wp = [with_proj[p][model_name]["f1"] for p in with_proj if model_name in with_proj[p]]
                mean_auc_wp = sum(aucs_wp) / len(aucs_wp)
                mean_f1_wp = sum(f1s_wp) / len(f1s_wp)
                print(
                    f"  {model_name:14s} no-project: F1={mean_f1_noproj:.3f} ROC-AUC={mean_auc_noproj:.3f} "
                    f"(range {min(aucs):.3f}-{max(aucs):.3f})  |  "
                    f"with-project: F1={mean_f1_wp:.3f} ROC-AUC={mean_auc_wp:.3f}"
                )
        print()


if __name__ == "__main__":
    main()
