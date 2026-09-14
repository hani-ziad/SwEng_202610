"""
Create calibration and reliability diagrams for the time-ordered split.

The script plots binned predicted probabilities against observed outcome rates, which helps interpret Brier scores and probability quality.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve

from scripts.run_baseline import time_ordered_split
from src.baselines import make_logreg, make_random_forest
from src.data.load_tawos import load_reconstructed_sprints
from src.features import ALL_FEATURES, add_features, build_model_frame
from src.labels import add_labels

OUT_PATH = Path("results/calibration_reliability.pdf")
LABELS = [("label_delay", "Delay"), ("label_spillover", "Spillover")]
N_BINS = 8


def main() -> None:
    sprints = load_reconstructed_sprints()
    labeled = add_labels(sprints)
    featured = add_features(labeled)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.4), sharey=True)

    for ax, (label_col, title) in zip(axes, LABELS):
        frame = build_model_frame(featured, label_col)
        train, test = time_ordered_split(frame)

        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Perfect calibration")

        for name, factory, color, marker in [
            ("Logreg", make_logreg, "#1f77b4", "o"),
            ("RF", make_random_forest, "#d62728", "s"),
        ]:
            model = factory()
            model.fit(train[ALL_FEATURES], train[label_col])
            proba = model.predict_proba(test[ALL_FEATURES])[:, 1]
            frac_pos, mean_pred = calibration_curve(test[label_col], proba, n_bins=N_BINS, strategy="uniform")
            ax.plot(mean_pred, frac_pos, marker=marker, color=color, linewidth=1.3, markersize=4, label=name)

        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Mean predicted probability", fontsize=8.5)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("Observed frequency", fontsize=8.5)
    axes[0].legend(fontsize=7.5, loc="upper left", frameon=False)

    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
