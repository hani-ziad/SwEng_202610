"""
Run a project-level jackknife comparison for the time-ordered test set.

The script recomputes the RF-minus-logistic-regression F1 difference after excluding one project at a time, making project influence visible.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score

from scripts.run_baseline import time_ordered_split
from src.baselines import make_logreg, make_random_forest
from src.data.load_tawos import load_reconstructed_sprints
from src.features import ALL_FEATURES, add_features, build_model_frame
from src.labels import add_labels

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def _fit_predict(label_col: str) -> pd.DataFrame:
    sprints = load_reconstructed_sprints()
    labeled = add_labels(sprints)
    featured = add_features(labeled)
    frame = build_model_frame(featured, label_col)
    train, test = time_ordered_split(frame)

    out = test[["project", label_col]].copy().reset_index(drop=True)
    for name, factory in [("logreg", make_logreg), ("random_forest", make_random_forest)]:
        model = factory()
        model.fit(train[ALL_FEATURES], train[label_col])
        proba = model.predict_proba(test[ALL_FEATURES])[:, 1]
        out[f"pred_{name}"] = (proba >= 0.5).astype(int)
    return out


def jackknife(preds: pd.DataFrame, label_col: str) -> dict:
    full_f1_rf = f1_score(preds[label_col], preds["pred_random_forest"])
    full_f1_lr = f1_score(preds[label_col], preds["pred_logreg"])
    full_delta = full_f1_rf - full_f1_lr

    per_project = {}
    for proj in sorted(preds["project"].unique()):
        held_out = preds[preds["project"] != proj]
        f1_rf = f1_score(held_out[label_col], held_out["pred_random_forest"])
        f1_lr = f1_score(held_out[label_col], held_out["pred_logreg"])
        per_project[proj] = {
            "n_excluded": int((preds["project"] == proj).sum()),
            "f1_delta_excluding_this_project": round(f1_rf - f1_lr, 4),
        }

    deltas = [v["f1_delta_excluding_this_project"] for v in per_project.values()]
    return {
        "full_sample_f1_delta_rf_minus_logreg": round(full_delta, 4),
        "per_project_exclusion": per_project,
        "jackknife_delta_min": round(min(deltas), 4),
        "jackknife_delta_max": round(max(deltas), 4),
        "jackknife_delta_range": round(max(deltas) - min(deltas), 4),
    }


def main() -> None:
    results = {}
    for label_col in ["label_delay", "label_spillover"]:
        preds = _fit_predict(label_col)
        results[label_col] = jackknife(preds, label_col)

    out_path = RESULTS_DIR / "project_jackknife.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Wrote {out_path}\n")
    for label_col, r in results.items():
        print(f"=== {label_col} ===")
        print(f"  full-sample F1 delta (RF - logreg): {r['full_sample_f1_delta_rf_minus_logreg']}")
        print(f"  jackknife delta range across 8 project-exclusions: "
              f"[{r['jackknife_delta_min']}, {r['jackknife_delta_max']}]")
        for proj, v in r["per_project_exclusion"].items():
            print(f"    exclude {proj:<28s} (n={v['n_excluded']:>3d}): delta={v['f1_delta_excluding_this_project']}")
        print()


if __name__ == "__main__":
    main()
