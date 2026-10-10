"""Phase 2, step 2: which kinds of information drive the models?

Exact group-level Shapley attributions (see phase2/shapley.py) for five models
trained with the strict (planning-time-valid) features, explained on held-out
sprints of the time-ordered split:

  spillover side      P(any spillover) random forest   | issues completed (gradient boosting)
                      carryover share (gradient boosting)
  late-closure side   P(closed >1 day late) random forest | days late (median gradient boosting)

For each model we report each feature group's mean absolute attribution (in the
model's own output units) and its share of the total.

Usage (from the phase-2 folder):  python -m experiments.run_explanations
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import phase2
from phase2.data import FEATURE_SETS, GROUPS, load_phase2_frame, population, time_ordered_split
from phase2.models import TASKS, feature_columns, make_classifier, make_regressor
from phase2.shapley import exact_group_shapley

SEED = 42
N_EXPLAIN = 150
N_BACKGROUND = 40

# (task, model name, output unit)
SPECS = [
    ("spillover", "random_forest", "probability of any spillover"),
    ("issues_done", "gradient_boosting", "issues completed"),
    ("carry_share", "gradient_boosting", "share of scope carried over"),
    ("late_closure", "random_forest", "probability of closing >1 day late"),
    ("late_days", "gb_median", "days late (median model)"),
]


def main():
    d_all = load_phase2_frame()
    numeric = FEATURE_SETS["strict"]
    groups = GROUPS["strict"]
    cols = feature_columns(numeric, True)
    rng = np.random.default_rng(SEED)

    rows, per_sprint = [], {}
    for key, model_name, unit in SPECS:
        task = TASKS[key]
        train, test = time_ordered_split(population(d_all, task.population))
        if task.kind == "clf":
            est = make_classifier(model_name, numeric, True).fit(train[cols], train[task.target])
            predict = lambda f, est=est: est.predict_proba(f[cols])[:, 1]
        else:
            est = make_regressor(model_name, numeric, True).fit(train[cols], train[task.target])
            predict = lambda f, est=est: est.predict(f[cols])
        ex = test.iloc[rng.choice(len(test), size=min(N_EXPLAIN, len(test)), replace=False)]
        bg = train.iloc[rng.choice(len(train), size=N_BACKGROUND, replace=False)]
        phi = exact_group_shapley(predict, ex[cols], bg[cols], groups)
        names = list(groups)
        mean_abs = phi[names].abs().mean()
        for g in names:
            rows.append({"model": key, "outcome": task.outcome, "algorithm": model_name,
                         "output_unit": unit, "group": g,
                         "mean_abs_attribution": float(mean_abs[g]),
                         "share_of_total": float(mean_abs[g] / mean_abs.sum())})
        gap = float(np.abs(phi[names].sum(axis=1) + phi["base_value"] - predict(ex[cols])).max())
        print(f"{key}: max efficiency gap = {gap:.2e}")
        per_sprint[key] = (ex, phi)

    out = pd.DataFrame(rows)
    out.to_csv(phase2.RESULTS_DIR / "explanation_group_importance.csv", index=False)

    frames = []
    for key, (ex, phi) in per_sprint.items():
        f = ex[["sprint_id", "project", "committed_issue_count"]].copy()
        f.insert(0, "model", key)
        for g in phi.columns:
            if g != "base_value":
                f[f"attr__{g}"] = phi[g].to_numpy()
        frames.append(f)
    pd.concat(frames, ignore_index=True).to_csv(phase2.RESULTS_DIR / "explanation_per_sprint.csv", index=False)

    pd.set_option("display.width", 200)
    wide = out.pivot(index="group", columns="model", values="share_of_total").mul(100).round(0)
    print(wide[[k for k, _, _ in SPECS]].to_string())


if __name__ == "__main__":
    main()
