"""Training-side quantities (added 15 Nov 2026, night, after the second external review).

1. The model lines shown to G quoted the model's AUC and the delivery model's error measured on the
   held-out sprints, which include the replay sample. This script recomputes both from 5-fold
   out-of-fold predictions on the training sprints only, for the sensitivity rerun of G
   (study/sensitivity.py).
2. The paper compares fixed warning policies across cost ratios. Here M's two thresholds (planning,
   halfway) are chosen on training out-of-fold scores to minimise the scenario cost of the whole
   sequential policy, separately for each scenario and cost ratio, and then applied unchanged to the
   replay sample.

Usage (from the ccece-2027-ai-scrum-master folder):  python -m study.train_side
Writes study/results/train_side.json and study/results/cost_optimal_M.csv.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict

import study
import phase3
from phase2.data import STRICT_NUMERIC, load_phase2_frame, population, time_ordered_split
from phase2.models import feature_columns, make_classifier, make_regressor
from phase3.data import CHECKPOINT_NUMERIC
from study.build import NO_RESOLUTION, SEED
from study.scenarios import SCEN, per_sprint

R = study.HERE / "results"
RATIOS = [0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
RUNS = (1, 2, 3)


def oof_clf(train, numeric):
    est = make_classifier("gradient_boosting", numeric, True)
    cols = feature_columns(numeric, True)
    y = train["label_delay"].to_numpy(int)
    return y, cross_val_predict(est, train[cols], y, cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                                method="predict_proba")[:, 1]


def oof_reg(train, numeric):
    tr = train[train["project"] != NO_RESOLUTION]
    est = make_regressor("gradient_boosting", numeric, True)
    cols = feature_columns(numeric, True)
    y = tr["carry_ratio"].to_numpy(float)
    p = np.clip(cross_val_predict(est, tr[cols], y, cv=KFold(5, shuffle=True, random_state=SEED)), 0, 1)
    return float(mean_absolute_error(y, p))


def main():
    base = population(load_phase2_frame(), "full")
    tr0, _ = time_ordered_split(base)
    ck = pd.read_csv(phase3.RESULTS_DIR / "phase3_checkpoint_frame.csv",
                     parse_dates=["start_date", "end_date", "complete_date", "checkpoint_time"])
    h = ck[(ck.checkpoint == 0.5) & ck.open_at_checkpoint]
    tr5 = h[h.split_full == "train"]

    y0, s0 = oof_clf(tr0, STRICT_NUMERIC)
    y5, s5 = oof_clf(tr5, CHECKPOINT_NUMERIC)
    tr0n = tr0[tr0["nonempty"]] if "nonempty" in tr0 else tr0
    tr5n = tr5[tr5["nonempty"]] if "nonempty" in tr5 else tr5
    out = {"train_oof_auc": {"0.0": float(roc_auc_score(y0, s0)), "0.5": float(roc_auc_score(y5, s5))},
           "train_oof_mae_unfinished": {"0.0": oof_reg(tr0n, STRICT_NUMERIC), "0.5": oof_reg(tr5n, CHECKPOINT_NUMERIC)},
           "n_train": {"0.0": int(len(y0)), "0.5": int(len(y5))}}

    # sequential training policy: sprints in both training sets
    a = pd.DataFrame({"sprint_id": tr0["sprint_id"].values, "s0": s0, "late": y0})
    b = pd.DataFrame({"sprint_id": tr5["sprint_id"].values, "s5": s5})
    t = a.merge(b, on="sprint_id", how="inner")
    out["n_train_sequential"] = int(len(t))
    grid = np.unique(np.quantile(np.r_[s0, s5], np.linspace(0, 1, 61)))
    grid = np.r_[grid, 1.01]                       # 1.01 = never warn at that checkpoint

    d = pd.read_csv(R / "replay_all_arms.csv")
    A = d[d.checkpoint == 0.0].set_index("sprint_id")
    B = d[d.checkpoint == 0.5].set_index("sprint_id").loc[A.index]
    wt = A["weight"].to_numpy()
    rows = []
    for name, e in SCEN.items():
        for ratio in RATIOS:
            best = None
            for g0 in grid:
                w0 = (t.s0.to_numpy() >= g0)
                for g5 in grid:
                    w5 = (t.s5.to_numpy() >= g5)
                    warned = (w0 | w5).astype(float)
                    eff = np.where(w0, e[0], np.where(w5, e[1], 0.0))
                    c = float(np.mean(ratio * warned + t.late.to_numpy() * (1 - eff)))
                    if best is None or c < best[0] - 1e-12:
                        best = (c, g0, g5)
            _, g0, g5 = best
            tmp = pd.DataFrame({"w": (A.m_score.to_numpy() >= g0).astype(int)}, index=A.index)
            tmp5 = pd.DataFrame({"w": (B.m_score.to_numpy() >= g5).astype(int)}, index=A.index)
            opt = np.average(ratio * ((tmp.w + tmp5.w) > 0) + A.late.to_numpy()
                             * (1 - np.where(tmp.w > 0, e[0], np.where(tmp5.w > 0, e[1], 0.0))), weights=wt)
            fixed = np.average(per_sprint(A, B, "m_warn", e, ratio), weights=wt)
            g = np.mean([np.average(per_sprint(A, B, f"G{r}_warn", e, ratio), weights=wt) for r in RUNS])
            never = np.average(A.late.to_numpy(), weights=wt)
            rows.append({"scenario": name, "kappa_over_F": ratio, "t0": float(g0), "t50": float(g5),
                         "warn_rate_c0": float(tmp.w.mean()), "warn_rate_c50": float(tmp5.w.mean()),
                         "cost_M_fixed": float(fixed), "cost_M_opt": float(opt), "cost_G": float(g),
                         "cost_never": float(never), "G_minus_Mopt": float(g - opt)})
    df = pd.DataFrame(rows)
    df.to_csv(R / "cost_optimal_M.csv", index=False)
    out["cost_optimal_M"] = df.round(4).to_dict("records")
    (R / "train_side.json").write_text(json.dumps(out, indent=1))
    pd.set_option("display.width", 220)
    print(json.dumps({k: v for k, v in out.items() if k != "cost_optimal_M"}, indent=1))
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
