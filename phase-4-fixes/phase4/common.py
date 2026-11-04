"""Shared pieces: tasks, models, the threshold rule and the metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from phase2.data import STRICT_NUMERIC, load_phase2_frame, population
from phase2.models import SCOPE_ONLY, feature_columns, make_classifier

SEED = 42
N_REPEATS = 5

# the two classification problems of the paper
TASKS = {
    "late_closure": {"population": "full", "target": "label_delay",
                     "label": "Late closure (all 2,602 sprints)"},
    "spillover": {"population": "nonempty", "target": "label_spillover",
                  "label": "Spillover (1,691 sprints with committed work)"},
}

# learned models use Phase 2's strict planning-time features + project, Phase 2 settings
MODELS = {
    "Gradient boosting": ("gradient_boosting", STRICT_NUMERIC, True),
    "Random forest": ("random_forest", STRICT_NUMERIC, True),
    "Logistic regression": ("logreg", STRICT_NUMERIC, True),
    "Scope-only LR (baseline)": ("logreg", SCOPE_ONLY, False),
}


def frame(task: str) -> pd.DataFrame:
    d = population(load_phase2_frame(), TASKS[task]["population"])
    return d.sort_values(["project", "start_date", "sprint_id"]).reset_index(drop=True)


def build(model: str):
    name, numeric, with_project = MODELS[model]
    return make_classifier(name, numeric, with_project), feature_columns(numeric, with_project)


def youden_threshold(y, score) -> float:
    """Threshold that maximises sensitivity + specificity - 1."""
    order = np.unique(score)
    best, best_t = -1.0, 0.5
    pos, neg = (y == 1).sum(), (y == 0).sum()
    for t in order:
        pred = score >= t
        sens = (pred & (y == 1)).sum() / max(pos, 1)
        spec = (~pred & (y == 0)).sum() / max(neg, 1)
        if sens + spec - 1 > best:
            best, best_t = sens + spec - 1, t
    return float(best_t)


def fit_score(model: str, train: pd.DataFrame, test: pd.DataFrame, target: str, seed: int = SEED):
    """Fit on train, score test, and fix the decision threshold on the TRAINING data only
    (Youden's J on 5-fold out-of-fold training predictions), never on the test set."""
    est, cols = build(model)
    y = train[target].to_numpy(int)
    inner = StratifiedKFold(5, shuffle=True, random_state=seed)
    oof = cross_val_predict(est, train[cols], y, cv=inner, method="predict_proba")[:, 1]
    thr = youden_threshold(y, oof)
    est.fit(train[cols], y)
    return est.predict_proba(test[cols])[:, 1], thr


def metrics(y, score, thr) -> dict:
    """AUC plus sensitivity, specificity and F1 at a given threshold."""
    m = decision_metrics(y, score, (np.asarray(score) >= thr).astype(int))
    m["threshold"] = thr
    return m


def decision_metrics(y, score, pred) -> dict:
    y, pred = np.asarray(y, int), np.asarray(pred, int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "auc": roc_auc_score(y, score) if 0 < y.sum() < len(y) else np.nan,
        "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
        "specificity": tn / (tn + fp) if tn + fp else np.nan,
        "f1": f1_score(y, pred, zero_division=0),
        "n": len(y), "n_pos": int(y.sum()), "n_neg": int(len(y) - y.sum()),
    }
