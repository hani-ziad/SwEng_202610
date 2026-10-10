"""Tasks, models, baselines and metrics for the Phase 2 study.

Five prediction tasks cover both outcomes of the project:

  spillover side (nonempty sprints, n=1,691)
    carry_share   regression      share of committed issues not done by planned end
    issues_done   regression      number of committed issues done by planned end
    spillover     classification  at least one committed issue not done (Phase 1 label)
  late-closure side (all modeled sprints, n=2,602)
    late_days     regression      days closed after planned end (0 if on time, cap 30)
    late_closure  classification  closed more than 1 day late (Phase 1 label)

XGBoost is not required: scikit-learn's HistGradientBoosting plays the
gradient-boosting role so Phase 2 runs on the base scientific stack.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
REG_MODELS = ["ridge", "random_forest", "gradient_boosting", "gb_median"]
CLF_MODELS = ["logreg", "random_forest", "gradient_boosting"]


@dataclass(frozen=True)
class Task:
    key: str
    population: str      # 'nonempty' or 'full'
    target: str
    kind: str            # 'reg' or 'clf'
    outcome: str         # 'spillover' or 'late closure'
    primary_baseline: str
    description: str


TASKS = {
    t.key: t
    for t in [
        Task("carry_share", "nonempty", "carry_ratio", "reg", "spillover", "train_median",
             "share of committed issues not done by planned end"),
        Task("issues_done", "nonempty", "completed_issue_count", "reg", "spillover", "scope_x_rate",
             "number of committed issues done by planned end"),
        Task("spillover", "nonempty", "label_spillover", "clf", "spillover", "scope_only_logreg",
             "at least one committed issue not done (Phase 1 label)"),
        Task("late_days", "full", "late_days_capped", "reg", "late closure", "train_median",
             "days closed after the planned end (0 if on time, capped at 30)"),
        Task("late_closure", "full", "label_delay", "clf", "late closure", "scope_only_logreg",
             "closed more than 1 day after the planned end (Phase 1 label)"),
    ]
}


def _prep(numeric: list[str], with_project: bool, scale: bool = True) -> ColumnTransformer:
    steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scale", StandardScaler()))
    transformers = [("num", Pipeline(steps), numeric)]
    if with_project:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), ["project"]))
    return ColumnTransformer(transformers)


def feature_columns(numeric: list[str], with_project: bool) -> list[str]:
    return list(numeric) + (["project"] if with_project else [])


def make_regressor(name: str, numeric: list[str], with_project: bool = True) -> Pipeline:
    if name == "ridge":
        est = Ridge(alpha=1.0)
    elif name == "random_forest":
        est = RandomForestRegressor(
            n_estimators=300, max_depth=5, min_samples_leaf=3, random_state=SEED, n_jobs=-1
        )
    elif name == "gradient_boosting":
        est = HistGradientBoostingRegressor(
            max_depth=3, learning_rate=0.05, max_iter=200, random_state=SEED
        )
    elif name == "gb_median":      # absolute-error loss: predicts the median, matches MAE
        est = HistGradientBoostingRegressor(
            loss="absolute_error", max_depth=3, learning_rate=0.05, max_iter=200, random_state=SEED
        )
    else:
        raise ValueError(name)
    return Pipeline([("prep", _prep(numeric, with_project)), ("model", est)])


def make_classifier(name: str, numeric: list[str], with_project: bool = True) -> Pipeline:
    if name == "logreg":
        est = LogisticRegression(max_iter=1000, class_weight="balanced")
    elif name == "random_forest":     # same settings as Phase 1
        est = RandomForestClassifier(
            n_estimators=300, max_depth=5, min_samples_leaf=3,
            class_weight="balanced", random_state=SEED, n_jobs=-1,
        )
    elif name == "gradient_boosting":
        est = HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=200, random_state=SEED
        )
    else:
        raise ValueError(name)
    return Pipeline([("prep", _prep(numeric, with_project)), ("model", est)])


SCOPE_ONLY = ["committed_issue_count", "committed_story_points", "sprint_length_days"]


# ---- no-learning baselines: (train, test) -> predictions --------------------
def _fill(series: pd.Series, fallback: float) -> np.ndarray:
    return series.fillna(fallback).to_numpy(float)


def bl_train_mean(train, test, target):
    return np.full(len(test), train[target].mean())


def bl_train_median(train, test, target):
    return np.full(len(test), train[target].median())


def bl_scope_x_rate(train, test, target):
    rate = train["completed_issue_count"].sum() / train["committed_issue_count"].sum()
    if target == "carry_ratio":
        return np.full(len(test), 1.0 - rate)
    return (test["committed_issue_count"] * rate).to_numpy(float)


def bl_prior_history(train, test, target):
    """Predict from the team's own already-closed recent sprints; no fitting."""
    if target == "carry_ratio":
        return 1.0 - _fill(test["hist_avg_completion_ratio_3"], 1.0 - train[target].mean())
    if target == "completed_issue_count":
        return _fill(test["hist_avg_completed_issues_3"], train[target].mean())
    return _fill(test["hist_avg_late_days_3"], train[target].median())


def bl_project_median(train, test, target):
    med = train.groupby("project")[target].median()
    return test["project"].map(med).fillna(train[target].median()).to_numpy(float)


REG_BASELINES = {
    "carry_ratio": {"train_mean": bl_train_mean, "train_median": bl_train_median, "prior_history": bl_prior_history,
                    "scope_x_rate": bl_scope_x_rate},
    "completed_issue_count": {"train_mean": bl_train_mean, "prior_history": bl_prior_history,
                              "scope_x_rate": bl_scope_x_rate},
    "late_days_capped": {"train_median": bl_train_median, "train_mean": bl_train_mean,
                         "prior_history": bl_prior_history, "project_median": bl_project_median},
}


def clf_baseline_scores(train, test, target, with_project: bool) -> dict[str, np.ndarray]:
    """Probability-like scores from simple baselines for a classification task."""
    out = {"always_positive": np.ones(len(test)),
           "train_base_rate": np.full(len(test), train[target].mean())}
    lr = LogisticRegression(max_iter=1000, class_weight="balanced")
    pipe = Pipeline([("prep", _prep(SCOPE_ONLY, False)), ("model", lr)])
    pipe.fit(train[SCOPE_ONLY], train[target])
    out["scope_only_logreg"] = pipe.predict_proba(test[SCOPE_ONLY])[:, 1]
    if with_project:
        rate = train.groupby("project")[target].mean()
        out["project_base_rate"] = test["project"].map(rate).fillna(train[target].mean()).to_numpy(float)
    return out


# ---- metrics ----------------------------------------------------------------
def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rho = float("nan")
    if np.std(y_pred) > 0 and np.std(y_true) > 0:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rho = float(spearmanr(y_true, y_pred)[0])
    return {
        "n": int(len(y_true)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "spearman": rho,
    }


def classification_metrics(y_true, score, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    score = np.asarray(score, dtype=float)
    pred = (score >= threshold).astype(int)
    out = {
        "n": int(len(y_true)),
        "positive_rate": float(y_true.mean()),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float("nan"), "pr_auc": float("nan"), "brier": float("nan"),
    }
    if len(set(y_true)) > 1 and np.std(score) > 0:
        out["roc_auc"] = float(roc_auc_score(y_true, score))
        out["pr_auc"] = float(average_precision_score(y_true, score))
        out["brier"] = float(brier_score_loss(y_true, np.clip(score, 0, 1)))
    return out
