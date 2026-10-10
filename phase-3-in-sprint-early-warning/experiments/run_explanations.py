"""Phase 3, step 2: what do the in-sprint models use at the 50% checkpoint?

Exact group-level Shapley attributions (Phase 2's implementation) for four
progress-aware models, explained on held-out sprints of the time-ordered split:

  spillover (random forest), carry_share (gradient boosting),
  late_closure (random forest), late_days (median gradient boosting)

Usage (from the phase-3 folder):  python -m experiments.run_explanations
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import phase3
from phase2.models import feature_columns, make_classifier, make_regressor
from phase2.shapley import exact_group_shapley
from phase3.data import CHECKPOINT_NUMERIC, GROUPS, load_checkpoint_frame

SEED = 42
CHECKPOINT = 0.5
N_EXPLAIN = 100
N_BACKGROUND = 30

SPECS = [
    # key, population, target, kind, algorithm, unit
    ("spillover", "nonempty", "label_spillover", "clf", "random_forest", "probability of any spillover"),
    ("carry_share", "nonempty", "carry_ratio", "reg", "gradient_boosting", "share of scope carried over"),
    ("late_closure", "full", "label_delay", "clf", "random_forest", "probability of closing >1 day late"),
    ("late_days", "full", "late_days_capped", "reg", "gb_median", "days late (median model)"),
]


def main():
    frame = load_checkpoint_frame()
    rng = np.random.default_rng(SEED)
    cols = feature_columns(CHECKPOINT_NUMERIC, True)
    rows = []
    for key, popname, target, kind, algo, unit in SPECS:
        sub = frame[(frame["checkpoint"] == CHECKPOINT) & frame["open_at_checkpoint"]]
        if popname == "nonempty":
            sub = sub[sub["nonempty"]]
        train, test = sub[sub[f"split_{popname}"] == "train"], sub[sub[f"split_{popname}"] == "test"]
        if kind == "clf":
            est = make_classifier(algo, CHECKPOINT_NUMERIC, True).fit(train[cols], train[target])
            predict = lambda f, est=est: est.predict_proba(f[cols])[:, 1]
        else:
            est = make_regressor(algo, CHECKPOINT_NUMERIC, True).fit(train[cols], train[target])
            predict = lambda f, est=est: est.predict(f[cols])
        ex = test.iloc[rng.choice(len(test), size=min(N_EXPLAIN, len(test)), replace=False)]
        bg = train.iloc[rng.choice(len(train), size=N_BACKGROUND, replace=False)]
        phi = exact_group_shapley(predict, ex[cols], bg[cols], GROUPS)
        names = list(GROUPS)
        mean_abs = phi[names].abs().mean()
        for g in names:
            rows.append({"model": key, "algorithm": algo, "output_unit": unit, "group": g,
                         "mean_abs_attribution": float(mean_abs[g]),
                         "share_of_total": float(mean_abs[g] / mean_abs.sum())})
        gap = float(np.abs(phi[names].sum(axis=1) + phi["base_value"] - predict(ex[cols])).max())
        print(f"{key}: max efficiency gap = {gap:.2e}", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(phase3.RESULTS_DIR / "explanation_group_importance.csv", index=False)
    wide = out.pivot(index="group", columns="model", values="share_of_total").mul(100).round(0)
    print(wide[[k for k, *_ in SPECS]].to_string())


if __name__ == "__main__":
    main()
