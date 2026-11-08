"""Fig. 1 of the CCECE paper: compact pipeline, sized for one IEEE column (3.5 in)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 7.5})
W, H = 3.5, 1.7
fig = plt.figure(figsize=(W, H), dpi=300)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
bw, bh, gap = 1.06, 0.6, 0.12
x0 = (W - 3 * bw - 2 * gap) / 2
rows = [
    [("TAWOS export", "8 projects\n2,657 sprint rows"),
     ("Sprint-start", "change-log replay\nfallbacks logged"),
     ("Planning time", "two labels\nstrict history")],
    [("In-sprint", "checkpoints at\n25, 50, 75, 100%"),
     ("Evaluation", "time split \u00d75, LOPO\n10 external projects"),
     ("Audit", "count and rate rules\nleakage cases")],
]
ys = [H - 0.04 - bh, 0.04]
centers = []
for r, (row, y) in enumerate(zip(rows, ys)):
    for c, (title, body) in enumerate(row):
        x = x0 + c * (bw + gap)
        ax.add_patch(FancyBboxPatch((x, y), bw, bh, boxstyle="round,pad=0,rounding_size=0.05",
                                    fc="#f0f0f0" if (r, c) != (0, 0) else "#e2e2e2", ec="#333333", lw=0.7))
        ax.text(x + bw / 2, y + bh - 0.13, title, ha="center", va="center", fontweight="bold", fontsize=7)
        ax.text(x + bw / 2, y + bh / 2 - 0.1, body, ha="center", va="center", fontsize=6.4, linespacing=1.15, color="#222222")
        centers.append((x, y))
arrow = dict(arrowstyle="-|>", mutation_scale=7, lw=0.8, color="#444444")
for r in range(2):
    for c in range(2):
        x, y = x0 + c * (bw + gap) + bw, ys[r] + bh / 2
        ax.add_patch(FancyArrowPatch((x + 0.02, y), (x + gap - 0.02, y), **arrow))
# row 1 end to row 2 start: down, left, down
xa, xb = x0 + 2 * (bw + gap) + bw / 2, x0 + bw / 2
ymid = (ys[0] + ys[1] + bh) / 2
ax.plot([xa, xa, xb], [ys[0] - 0.01, ymid, ymid], color="#444444", lw=0.8)
ax.add_patch(FancyArrowPatch((xb, ymid), (xb, ys[1] + bh + 0.01), **arrow))
fig.savefig("fig1_pipeline.png", dpi=300)
print("ok")
