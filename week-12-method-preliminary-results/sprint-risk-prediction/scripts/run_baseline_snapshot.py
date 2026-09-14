"""Run the current baseline using Change_Log-reconstructed sprint-start scope.

This is the scientifically preferred runner for the final paper. It fails fast
if the richer TAWOS export is missing instead of silently falling back to the
static Issue.Sprint_ID mapping.

Usage:
    python -m scripts.run_baseline_snapshot
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.load_tawos import load_change_log, load_dataset
from src.data.planning_snapshot import derive_planning_time_aggregates
from src.evaluate import evaluate
from src.features import add_features
from src.labels import add_labels

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results_fixed"
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "tawos"
LABELS = ["label_delay", "label_spillover"]
TIME_SPLIT_TRAIN_FRACTION = 0.7

# num_developers is intentionally excluded: current Assignee_ID and historical
# assignee Change_Log values are not guaranteed to share an identity namespace.
PLANNING_SAFE = ["committed_issue_count", "committed_story_points", "sprint_length_days"]
HISTORICAL = [
    "prior_avg_delivered_sp_3",
    "prior_avg_completion_ratio_3",
    "prior_sprint_carryover",
    "prior_sprint_completion_ratio",
]
CATEGORICAL = ["project"]
FEATURES = PLANNING_SAFE + HISTORICAL + CATEGORICAL
NUMERIC = PLANNING_SAFE + HISTORICAL


def _prep():
    return ColumnTransformer(
        [("num", StandardScaler(), NUMERIC), ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)]
    )


def _models():
    from xgboost import XGBClassifier
    return {
        "logreg": Pipeline([("prep", _prep()), ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))]),
        "random_forest": Pipeline([
            ("prep", _prep()),
            ("clf", RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=3, class_weight="balanced", random_state=42)),
        ]),
        "xgboost": Pipeline([
            ("prep", _prep()),
            ("clf", XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, eval_metric="logloss", random_state=42)),
        ]),
    }


def time_ordered_split(frame):
    train_parts, test_parts = [], []
    for _, g in frame.groupby("project"):
        g = g.sort_values("start_date")
        cut = max(1, int(len(g) * TIME_SPLIT_TRAIN_FRACTION))
        train_parts.append(g.iloc[:cut])
        test_parts.append(g.iloc[cut:])
    return pd.concat(train_parts), pd.concat(test_parts)


def heuristic_predict(frame):
    return (frame["prior_avg_completion_ratio_3"] < 0.8).astype(int).to_numpy()


def build_frame(featured, label):
    cols = list(dict.fromkeys(FEATURES + [label, "project", "sprint_id", "start_date", "has_prior_sprint"]))
    f = featured[cols].copy()
    f = f[f["has_prior_sprint"]].reset_index(drop=True)
    if f[FEATURES].isna().any().any():
        raise ValueError("Unexpected NaN in snapshot-safe feature frame")
    return f


def run_one(train, test, label):
    out = {"heuristic": evaluate(test[label], heuristic_predict(test))}
    for name, model in _models().items():
        model.fit(train[FEATURES], train[label])
        proba = model.predict_proba(test[FEATURES])[:, 1]
        pred = (proba >= 0.5).astype(int)
        out[name] = evaluate(test[label], pred, proba)
    return out


def main():
    sprints, issues = load_dataset(RAW_DIR, modeling_only=True)
    changes = load_change_log(RAW_DIR, required=True)
    agg, audit = derive_planning_time_aggregates(sprints, issues, changes)
    labeled = add_labels(agg)
    featured = add_features(labeled)

    results = {"audit": audit, "features": FEATURES, "labels": {}}
    for label in LABELS:
        frame = build_frame(featured, label)
        train, test = time_ordered_split(frame)
        entry = {"time_ordered": run_one(train, test, label), "lopo": {}}
        for held_out in frame["project"].unique():
            tr = frame[frame["project"] != held_out]
            te = frame[frame["project"] == held_out]
            entry["lopo"][held_out] = run_one(tr, te, label)
        results["labels"][label] = entry

    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / "baseline_snapshot_results.json"
    path.write_text(json.dumps(results, indent=2, default=float))
    print(json.dumps(audit, indent=2))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
