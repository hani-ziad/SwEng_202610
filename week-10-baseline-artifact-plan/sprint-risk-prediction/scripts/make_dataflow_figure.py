"""
Create a data-flow and feature-timing diagram.

The figure shows when each feature and label is computed relative to a sprint timeline.
It is used as a visual check that planning-time features are separated from outcomes.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

OUT_PATH = Path("results/dataflow_timing.pdf")


def sprint_box(ax, x0, x1, y, label, color, height=0.5):
    ax.add_patch(Rectangle((x0, y), x1 - x0, height, facecolor=color, edgecolor="black", linewidth=0.8, alpha=0.85))
    ax.text((x0 + x1) / 2, y + height / 2, label, ha="center", va="center", fontsize=8)


def main() -> None:
    fig, ax = plt.subplots(figsize=(7.0, 3.0))

    y_sprint = 1.4
    h = 0.5

    # Three sprints in one project: N-2, N-1 (feed historical features), N (sprint of interest).
    sprint_box(ax, 0.0, 2.0, y_sprint, "Sprint $N{-}2$", "#dddddd", h)
    sprint_box(ax, 2.2, 4.2, y_sprint, "Sprint $N{-}1$", "#dddddd", h)
    sprint_box(ax, 4.4, 7.4, y_sprint, "Sprint $N$ (sprint of interest)", "#a6cee3", h)

    # Timeline axis.
    ax.annotate("", xy=(10.0, y_sprint + h / 2), xytext=(-0.3, y_sprint + h / 2),
                arrowprops=dict(arrowstyle="->", linewidth=1.0, color="black"))
    ax.text(10.05, y_sprint + h / 2, "time", va="center", fontsize=8)

    # Key timestamps for sprint N.
    start_x, end_x, complete_x = 4.4, 7.0, 8.6
    for x, label, ha in [
        (start_x, r"$\mathrm{start\_date}_N$", "center"),
        (end_x, r"$\mathrm{end\_date}_N$", "right"),
        (complete_x, r"$\mathrm{complete\_date}_N$", "left"),
    ]:
        ax.plot([x, x], [y_sprint - 0.05, y_sprint + h + 0.05], color="black", linewidth=0.8, linestyle=":")
        ax.text(x, y_sprint + h + 0.12, label, rotation=0, ha=ha, va="bottom", fontsize=7)

    # Historical features arrow: computed from sprints N-2, N-1, available before sprint N starts.
    ax.annotate(
        "",
        xy=(start_x - 0.05, y_sprint + h + 0.55),
        xytext=(0.2, y_sprint + h + 0.55),
        arrowprops=dict(arrowstyle="-|>", linewidth=1.1, color="#1f78b4"),
    )
    ax.text(
        2.1, y_sprint + h + 0.68,
        "historical features (prior_avg_*, prior_sprint_*):\ncomputed ONLY from sprints $N{-}2$, $N{-}1$",
        ha="center", va="bottom", fontsize=6.7, color="#1f78b4",
    )

    # Planning-time features: pinned at start_date, with the measurement-timing caveat.
    ax.annotate(
        "planning-time features\n(committed_issue_count, committed_story_points,\nnum_developers, sprint_length_days)\nnominally pinned here --\nsee measurement-timing caveat",
        xy=(start_x, y_sprint), xytext=(3.1, y_sprint - 1.65),
        fontsize=6.7, ha="center", va="top", color="#33a02c",
        arrowprops=dict(arrowstyle="-|>", linewidth=1.1, color="#33a02c", connectionstyle="arc3,rad=-0.25"),
    )

    # Spillover label: determined at end_date.
    ax.annotate(
        r"$y^{\mathrm{spill}}_N$ determined here:" "\nany committed issue not\ncompleted by end_date",
        xy=(end_x, y_sprint), xytext=(6.1, y_sprint - 1.65),
        fontsize=6.7, ha="center", va="top", color="#e31a1c",
        arrowprops=dict(arrowstyle="-|>", linewidth=1.1, color="#e31a1c", connectionstyle="arc3,rad=0.2"),
    )

    # Delay label: determined at complete_date.
    ax.annotate(
        r"$y^{\mathrm{delay}}_N$ determined here:" "\ncomplete_date - end_date\n> 1 day (late closure proxy)",
        xy=(complete_x, y_sprint), xytext=(8.9, y_sprint - 1.65),
        fontsize=6.7, ha="left", va="top", color="#e31a1c",
        arrowprops=dict(arrowstyle="-|>", linewidth=1.1, color="#e31a1c", connectionstyle="arc3,rad=-0.15"),
    )

    ax.set_xlim(-0.6, 10.4)
    ax.set_ylim(-2.1, 2.9)
    ax.axis("off")
    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
