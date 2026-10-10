"""Draw the Phase 2 attribution figure from results/explanation_group_importance.csv.

Usage (from the phase-2 folder):  python -m experiments.make_figure
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import phase2

SURFACE, INK, MUTED, BAR = "#fcfcfb", "#0b0b0b", "#52514e", "#2a78d6"
ORDER = ["scope", "capacity", "recent history", "past lateness", "parallel sprints",
         "sprint length", "project"]
PANELS = [
    ("spillover", "Spillover:\nany issue not done?"),
    ("issues_done", "Spillover:\nissues completed"),
    ("carry_share", "Spillover:\ncarryover share"),
    ("late_closure", "Late closure:\nclosed >1 day late?"),
    ("late_days", "Late closure:\ndays late"),
]


def main():
    df = pd.read_csv(phase2.RESULTS_DIR / "explanation_group_importance.csv")
    fig, axes = plt.subplots(1, 5, figsize=(14.5, 3.9), sharex=True, sharey=True, facecolor=SURFACE)
    for ax, (model, title) in zip(axes, PANELS):
        sub = df[df["model"] == model].set_index("group")["share_of_total"] * 100
        groups = ORDER[::-1]
        vals = [sub[g] for g in groups]
        ax.set_facecolor(SURFACE)
        bars = ax.barh(groups, vals, color=BAR, height=0.6)
        for b, v in zip(bars, vals):
            ax.text(v + 1.5, b.get_y() + b.get_height() / 2, f"{v:.0f}%",
                    va="center", ha="left", fontsize=8.5, color=INK)
        ax.set_title(title, fontsize=9.5, color=INK, loc="left")
        ax.set_xlim(0, 100)
        ax.tick_params(colors=MUTED, labelsize=8.5, length=0)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color("#d6d5d0")
        ax.xaxis.grid(True, color="#e8e7e3", linewidth=0.8)
        ax.set_axisbelow(True)
    fig.supxlabel("Share of total mean |Shapley attribution| (%)", fontsize=9, color=MUTED)
    fig.suptitle("Spillover and issues-completed follow scope and capacity; "
                 "late closure follows team habits (past lateness, project, sprint length)", fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.03, 1, 0.92))
    out = phase2.RESULTS_DIR / "explanation_group_shares.png"
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    print("wrote", out)


if __name__ == "__main__":
    main()
