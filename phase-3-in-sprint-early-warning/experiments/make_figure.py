"""Draw the Phase 3 figures from results/checkpoint_model_results.csv.

  early_warning_curves.png : prediction quality as the sprint progresses
  explanation_group_shares.png : what the 50%-checkpoint models use

Usage (from the phase-3 folder):  python -m experiments.make_figure
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import phase3

SURFACE, INK, MUTED, BLUE, GRAY, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#2a78d6", "#9b9a95", "#e8e7e3"


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=MUTED, labelsize=8.5, length=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#d6d5d0")
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xticks([0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["25%", "50%", "75%", "100%"])


def curves():
    res = pd.read_csv(phase3.RESULTS_DIR / "checkpoint_model_results.csv")
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9), facecolor=SURFACE)
    panels = [
        ("spillover", "roc_auc", "gradient_boosting", "Spillover (any issue not done)\nAUC, higher is better"),
        ("late_closure", "roc_auc", "gradient_boosting", "Late closure (>1 day late)\nAUC, higher is better"),
        ("carry_share", "mae", "gb_median", "Carryover share\nmean absolute error, lower is better"),
    ]
    for ax, (task, metric, algo, title) in zip(axes, panels):
        _style(ax)
        for protocol, ls in (("time_ordered", "-"), ("lopo", "--")):
            for fs, color, label in (("progress", BLUE, "with in-sprint progress"), ("planning", GRAY, "planning-time only")):
                s = res[(res.task == task) & (res.protocol == protocol) & (res.feature_set == fs) & (res.model == algo)]
                s = s.sort_values("checkpoint")
                ax.plot(s["checkpoint"], s[metric], ls, color=color, marker="o", ms=4, lw=1.8,
                        label=f"{label}" if protocol == "time_ordered" else None)
        if task == "carry_share":
            for protocol, ls in (("time_ordered", ":"),):
                s = res[(res.task == task) & (res.protocol == protocol) & (res.model == "rule_remaining_share")].sort_values("checkpoint")
                ax.plot(s["checkpoint"], s[metric], ls, color=INK, marker="s", ms=3.5, lw=1.4, label="rule: 1 - share done")
        ax.set_title(title, fontsize=9.5, color=INK, loc="left")
        ax.set_xlabel("point in the sprint", fontsize=8.5, color=MUTED)
    h0, l0 = axes[0].get_legend_handles_labels()
    h2, l2 = axes[2].get_legend_handles_labels()
    fig.legend(h0 + h2[-1:], l0 + l2[-1:], frameon=False, fontsize=8, labelcolor=MUTED,
               loc="lower right", ncol=3, bbox_to_anchor=(0.99, 0.0))
    fig.text(0.01, 0.01, "Solid: time-ordered split. Dashed: leave-one-project-out.", fontsize=8, color=MUTED)
    fig.suptitle("Progress inside the sprint transforms spillover prediction, but not late closure",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    out = phase3.RESULTS_DIR / "early_warning_curves.png"
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    print("wrote", out)


def shares():
    df = pd.read_csv(phase3.RESULTS_DIR / "explanation_group_importance.csv")
    order = ["scope", "recent history", "past lateness", "sprint length", "parallel sprints",
             "progress so far", "time pressure", "scope change", "project"]
    panels = [("spillover", "Spillover:\nany issue not done?"), ("carry_share", "Spillover:\ncarryover share"),
              ("late_closure", "Late closure:\nclosed >1 day late?"), ("late_days", "Late closure:\ndays late")]
    fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.9), sharex=True, sharey=True, facecolor=SURFACE)
    for ax, (model, title) in zip(axes, panels):
        sub = df[df["model"] == model].set_index("group")["share_of_total"] * 100
        groups = order[::-1]
        vals = [sub[g] for g in groups]
        ax.set_facecolor(SURFACE)
        bars = ax.barh(groups, vals, color=BLUE, height=0.6)
        for b, v in zip(bars, vals):
            ax.text(v + 1.5, b.get_y() + b.get_height() / 2, f"{v:.0f}%", va="center", fontsize=8.5, color=INK)
        ax.set_title(title, fontsize=9.5, color=INK, loc="left")
        ax.set_xlim(0, 100)
        ax.tick_params(colors=MUTED, labelsize=8.5, length=0)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color("#d6d5d0")
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
    fig.supxlabel("Share of total mean |Shapley attribution| (%), 50% checkpoint", fontsize=9, color=MUTED)
    fig.suptitle("Spillover models read progress; late-closure models still read team habits",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.03, 1, 0.92))
    out = phase3.RESULTS_DIR / "explanation_group_shares.png"
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    print("wrote", out)


def main():
    curves()
    if (phase3.RESULTS_DIR / "explanation_group_importance.csv").exists():
        shares()


if __name__ == "__main__":
    main()
