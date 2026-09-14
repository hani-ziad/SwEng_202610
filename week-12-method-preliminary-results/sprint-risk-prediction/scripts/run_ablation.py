"""
Week 14 checklist item: feature-group ablations.

Systematically drops feature groups defined in src/features.py and
re-measures logreg + random forest on both labels, time-ordered split,
using the real TAWOS data -- to see which features actually carry the
signal, rather than just reporting the full-feature-set numbers.

XGBoost is intentionally excluded from the ablation grid: its hyperparameters
were tuned for the full feature set, and 4 models x 4 feature configs x 2
labels would triple the runtime for a stage that already has XGBoost's
full-feature-set numbers in results/baseline_results.json. logreg and
random forest, which don't need XGBoost's optional dependency, are enough
to see whether a feature group matters.

Configurations:
  - full            : planning-time + historical + project (the main result)
  - planning_only   : planning-time + project (drop the 4 historical features)
  - historical_only : historical + project (drop the 4 planning-time features)
  - no_project      : planning-time + historical, WITHOUT the project dummy
                       (tests whether the model leans on project identity
                       itself rather than the sprint's own characteristics)

IMPORTANT -- cross-environment consistency (added after external review):
The "full" configuration is, by definition, the same experiment already
reported in results/baseline_results.json (Table 4 of the analysis). It is
NOT independently retrained here. A previous implementation *did*
retrain it from scratch, and that surfaced a real reproducibility gap: the
environment used for this ablation pass has scikit-learn 1.8.0, while
baseline_results.json was produced on the author's own machine (required
for XGBoost, which cannot install in the original development environment) with an unpinned,
almost certainly different scikit-learn version. Logistic regression
reproduced to 3 decimal places across both environments (convex
optimization has no meaningful version sensitivity here); random forest
did not -- it differed by up to 0.02 F1 on delay with an *identical* fixed
random_state=42, because RandomForestClassifier's exact tie-breaking at
ambiguous splits depends on library internals, not just the seed. Rather
than let that environment artifact masquerade as a substantive ablation
finding (dropping zero features should never change a number), the "full"
row now always copies its logreg/random_forest metrics directly from
baseline_results.json, so Table 8 is consistent with Table 4 by
construction. Only the three genuinely-ablated configurations involve a
fresh model fit in this environment.

Usage:
    python -m scripts.run_ablation
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.load_tawos import load_reconstructed_sprints
from src.evaluate import evaluate
from src.features import (
    CATEGORICAL_FEATURES,
    HISTORICAL_FEATURES,
    PLANNING_TIME_FEATURES,
    add_features,
)
from src.labels import add_labels
from scripts.run_baseline import time_ordered_split

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
BASELINE_RESULTS_PATH = RESULTS_DIR / "baseline_results.json"
LABELS = ["label_delay", "label_spillover"]

CONFIGS = {
    "full": PLANNING_TIME_FEATURES + HISTORICAL_FEATURES + CATEGORICAL_FEATURES,
    "planning_only": PLANNING_TIME_FEATURES + CATEGORICAL_FEATURES,
    "historical_only": HISTORICAL_FEATURES + CATEGORICAL_FEATURES,
    "no_project": PLANNING_TIME_FEATURES + HISTORICAL_FEATURES,
}


def _preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    transformers = [("num", StandardScaler(), numeric_cols)]
    if categorical_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols))
    return ColumnTransformer(transformers=transformers)


def _model(name: str, numeric_cols: list[str], categorical_cols: list[str]) -> Pipeline:
    prep = _preprocessor(numeric_cols, categorical_cols)
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
    # Same cold-start filter as build_model_frame(), applied once up front so
    # every ablation config trains/tests on identical rows.
    featured = featured.dropna(subset=["prior_sprint_completion_ratio"]).reset_index(drop=True)

    baseline = json.loads(BASELINE_RESULTS_PATH.read_text())

    all_results = {}
    for label_col in LABELS:
        all_results[label_col] = {}
        for config_name, feature_cols in CONFIGS.items():
            frame = featured[list(dict.fromkeys(feature_cols + [label_col, "project", "sprint_id", "start_date"]))]
            train, test = time_ordered_split(frame)

            numeric_cols = [c for c in feature_cols if c != "project"]
            categorical_cols = [c for c in feature_cols if c == "project"]

            config_results = {}
            if config_name == "full":
                # Copy straight from baseline_results.json rather than
                # retraining -- see the module docstring. This is the exact
                # same experiment already reported as Table 4; retraining it
                # here just reintroduces cross-environment scikit-learn
                # version noise that has nothing to do with the ablation.
                for model_name in ["logreg", "random_forest"]:
                    config_results[model_name] = baseline[label_col]["time_ordered"][model_name]
            else:
                for model_name in ["logreg", "random_forest"]:
                    model = _model(model_name, numeric_cols, categorical_cols)
                    model.fit(train[feature_cols], train[label_col])
                    proba = model.predict_proba(test[feature_cols])[:, 1]
                    pred = (proba >= 0.5).astype(int)
                    config_results[model_name] = evaluate(test[label_col], pred, proba)
            config_results["n_test"] = len(test)
            config_results["n_features"] = len(feature_cols)
            all_results[label_col][config_name] = config_results

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "ablation_results.json"
    out_path.write_text(json.dumps(all_results, indent=2, default=float))
    print(f"Wrote {out_path}\n")

    print("=== Ablation summary (time-ordered split) ===")
    rows = []
    for label_col in LABELS:
        for config_name, res in all_results[label_col].items():
            for model_name in ["logreg", "random_forest"]:
                m = res[model_name]
                rows.append(
                    {
                        "label": label_col,
                        "config": config_name,
                        "n_features": res["n_features"],
                        "model": model_name,
                        "f1": round(m["f1"], 3),
                        "roc_auc": round(m["roc_auc"], 3) if m["roc_auc"] == m["roc_auc"] else None,
                        "pr_auc": round(m["pr_auc"], 3) if m["pr_auc"] == m["pr_auc"] else None,
                    }
                )
    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False))
    summary.to_csv(RESULTS_DIR / "ablation_summary.csv", index=False)


if __name__ == "__main__":
    main()
