"""Scenario costs (added 15 Nov 2026 after the external review).

The cost in score.py rests on one assumed effectiveness, e(0)=0.5 and e(0.5)=0.25. A warning's
real effect is never observed in replay, so this script repeats the cost under three scenarios
and adds two trivial policies: never warn and always warn at planning.

  zero     e(0)=0,   e(0.5)=0      (a warning changes nothing)
  default  e(0)=0.5, e(0.5)=0.25   (the paper's main setting)
  strong   e(0)=0.8, e(0.5)=0.5

Cost of a sprint, in units of F:  kappa/F * warned + late * (1 - e(first warning)), and
late * 1 when the arm never warned. Means are weighted to the pool by project, as in score.py.
Agent arms are averaged over the three runs. Intervals for G - M: 1,000 bootstrap resamples of
sprints within project.

Usage (from the ccece-2027-ai-scrum-master folder):  python -m study.scenarios
Writes study/results/cost_scenarios.csv and cost_scenarios.json.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import study

OUT = study.HERE / "results"
SCEN = {"zero": (0.0, 0.0), "default": (0.5, 0.25), "strong": (0.8, 0.5)}
RATIOS = [0.02, 0.1, 0.5]
RUNS = (1, 2, 3)
SEED = 2027


def per_sprint(a, b, col, e, ratio):
    w0, w5 = a[col].to_numpy(), b[col].to_numpy()
    warned = ((w0 + w5) > 0).astype(float)
    eff = np.where(w0 > 0, e[0], np.where(w5 > 0, e[1], 0.0))
    return ratio * warned + a["late"].to_numpy() * (1 - eff)


def main():
    d = pd.read_csv(OUT / "replay_all_arms.csv")
    d["never"] = 0
    d["always"] = 1
    a = d[d.checkpoint == 0.0].set_index("sprint_id")
    b = d[d.checkpoint == 0.5].set_index("sprint_id").loc[a.index]
    wt = a["weight"].to_numpy()
    proj = a["project"].to_numpy()
    groups = [np.flatnonzero(proj == p) for p in np.unique(proj)]
    rng = np.random.default_rng(SEED)
    boot = [np.concatenate([rng.choice(g, len(g), replace=True) for g in groups]) for _ in range(1000)]

    rows = []
    for name, e in SCEN.items():
        for ratio in RATIOS:
            c = {"M": per_sprint(a, b, "m_warn", e, ratio),
                 "rule": per_sprint(a, b, "rule_warn", e, ratio),
                 "never": per_sprint(a, b, "never", e, ratio),
                 "always": per_sprint(a, b, "always", e, ratio),
                 "L": np.mean([per_sprint(a, b, f"L{r}_warn", e, ratio) for r in RUNS], axis=0),
                 "G": np.mean([per_sprint(a, b, f"G{r}_warn", e, ratio) for r in RUNS], axis=0)}
            mean = {k: float(np.average(v, weights=wt)) for k, v in c.items()}
            diff = c["G"] - c["M"]
            bs = [np.average(diff[i], weights=wt[i]) for i in boot]
            rows.append({"scenario": name, "e0": e[0], "e50": e[1], "kappa_over_F": ratio,
                         **{f"cost_{k}": v for k, v in mean.items()},
                         "cheapest": min(mean, key=mean.get),
                         "G_minus_M": mean["G"] - mean["M"],
                         "G_minus_M_lo": float(np.percentile(bs, 2.5)),
                         "G_minus_M_hi": float(np.percentile(bs, 97.5))})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "cost_scenarios.csv", index=False)
    # the range of kappa/F (step 0.01) over which each arm is cheaper than both trivial policies
    bands = {}
    grid = np.round(np.arange(0.01, 0.61, 0.01), 2)
    for name, e in SCEN.items():
        bands[name] = {}
        for arm in ("M", "rule", "L", "G"):
            win = []
            for ratio in grid:
                if arm in ("L", "G"):
                    c = np.mean([np.average(per_sprint(a, b, f"{arm}{r}_warn", e, ratio), weights=wt) for r in RUNS])
                else:
                    c = np.average(per_sprint(a, b, "m_warn" if arm == "M" else "rule_warn", e, ratio), weights=wt)
                triv = min(np.average(per_sprint(a, b, k, e, ratio), weights=wt) for k in ("never", "always"))
                win.append(c < triv)
            idx = np.flatnonzero(win)
            bands[name][arm] = None if not len(idx) else {
                "from": float(grid[idx[0]]), "to": float(grid[idx[-1]]),
                "contiguous": bool(idx[-1] - idx[0] + 1 == len(idx))}
    (OUT / "cost_scenarios.json").write_text(json.dumps(
        {"rows": df.round(4).to_dict("records"), "beats_trivial_band": bands}, indent=1))
    print(json.dumps(bands, indent=1))
    pd.set_option("display.width", 200)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
