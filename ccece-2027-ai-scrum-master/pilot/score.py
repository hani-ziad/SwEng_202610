"""Step 2 of the pilot: score the three arms on the 40 replayed sprints.

Reads results/replay_points.csv (model arm and labels, from build.py) and the agent answers in
agent_outputs/*.jsonl (saved verbatim from the agent runs; audit.json lists every tool call).

Reports, per arm and checkpoint: warn rate, sensitivity, specificity, F1, and the AUC of the
stated probability; the agreement rate A of each agent arm with the model; and the cost of
each arm under the alarm cost model of the paper, as a curve over the ratio kappa/F:

  C_a(s) = kappa * w_a(s) + F * y_s * (1 - w_a(s) * e(c*_a(s)))   (in units of F)

with e(0) = 0.5 and e(0.5) = 0.25 (assumed: a warning at planning can avert half of the cost
of a late close, one at mid-sprint a quarter), and Delta = min(C_M, C_L) - C_G.

The sample is balanced (20 late, 20 on time), so costs are reweighted to the late rate of the
eligible test sprints (134 of 472). Intervals: 2,000 bootstrap resamples of the 40 sprints.

Usage (from the ccece-2027-ai-scrum-master folder):  python -m pilot.score
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

import pilot

SEED = 42
N_BOOT = 2000
E = {0.0: 0.5, 0.5: 0.25}
RATIOS = [0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
OUT = pilot.HERE / "results"


def load():
    rp = pd.read_csv(OUT / "replay_points.csv")
    man = json.loads((pilot.HERE / "prompts" / "manifest.json").read_text())
    rows = []
    for m in man:
        lines = [l for l in (pilot.HERE / "agent_outputs" / f"{m['prompt']}.jsonl").read_text().splitlines() if l.strip()]
        got = [json.loads(l) for l in lines]
        assert sorted(g["id"] for g in got) == sorted(m["boards"]), m["prompt"]
        for g in got:
            rows.append({"board_id": g["id"], "checkpoint": m["checkpoint"], "arm": m["arm"],
                         "warn": int(bool(g["warn"])), "p_late": float(g["p_late"]), "message": g["message"]})
    ag = pd.DataFrame(rows)
    wide = rp.copy()
    for arm in ("L", "G"):
        a = ag[ag.arm == arm].rename(columns={"warn": f"{arm}_warn", "p_late": f"{arm}_p", "message": f"{arm}_msg"})
        wide = wide.merge(a.drop(columns="arm"), on=["board_id", "checkpoint"], how="left")
    wide["M_warn"], wide["M_p"] = wide["m_warn"], wide["m_score"]
    assert wide[["L_warn", "G_warn"]].notna().all().all()
    return wide


def arm_metrics(d):
    out = []
    for c, g in d.groupby("checkpoint"):
        y = g["late"].to_numpy()
        for arm in ("M", "L", "G"):
            w, p = g[f"{arm}_warn"].to_numpy(), g[f"{arm}_p"].to_numpy()
            out.append({"checkpoint": c, "arm": arm, "n": len(g), "warn_rate": w.mean(),
                        "sensitivity": w[y == 1].mean(), "specificity": 1 - w[y == 0].mean(),
                        "f1": f1_score(y, w), "auc": roc_auc_score(y, p),
                        "agreement_with_M": (w == g["M_warn"].to_numpy()).mean()})
    return pd.DataFrame(out)


def per_sprint(d):
    """One row per sprint: label, and per arm whether and when it first warned."""
    s = d.pivot_table(index="sprint_id", columns="checkpoint", values=["late", "M_warn", "L_warn", "G_warn"])
    r = pd.DataFrame({"late": s[("late", 0.0)]})
    for arm in ("M", "L", "G"):
        w0, w5 = s[(f"{arm}_warn", 0.0)], s[(f"{arm}_warn", 0.5)]
        r[f"{arm}_w"] = ((w0 + w5) > 0).astype(int)
        r[f"{arm}_e"] = np.where(w0 > 0, E[0.0], np.where(w5 > 0, E[0.5], 0.0))
    return r


def costs(r, ratio, weights):
    c = {}
    for arm in ("M", "L", "G"):
        per = ratio * r[f"{arm}_w"] + r["late"] * (1 - r[f"{arm}_e"])
        c[arm] = np.average(per, weights=weights)
    c["delta"] = min(c["M"], c["L"]) - c["G"]
    c["G_minus_M"] = c["M"] - c["G"]
    return c


def main():
    d = load()
    d.to_csv(OUT / "replay_all_arms.csv", index=False)
    am = arm_metrics(d)
    am.to_csv(OUT / "arm_metrics.csv", index=False)

    rng = np.random.default_rng(SEED)
    sprints = d["sprint_id"].unique()
    r = per_sprint(d).loc[sprints]
    p_late = 134 / 472
    wts = np.where(r["late"] == 1, p_late / 0.5, (1 - p_late) / 0.5)

    # point estimates and bootstrap over sprints
    boot_idx = [rng.integers(0, len(sprints), len(sprints)) for _ in range(N_BOOT)]

    def ci(fn):
        vals = []
        for ix in boot_idx:
            try:
                vals.append(fn(ix))
            except ValueError:
                continue
        return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]

    by_sprint = {sid: d[d.sprint_id == sid] for sid in sprints}

    def sub(ix):
        return pd.concat([by_sprint[sprints[i]] for i in ix], ignore_index=True)

    summary = {"n_sprints": len(sprints), "n_replay_points_per_arm": int(len(d))}
    pooled = {}
    for arm in ("L", "G"):
        pooled[f"A_{arm}"] = float((d[f"{arm}_warn"] == d["M_warn"]).mean())
    pooled["A_L_vs_G"] = float((d["L_warn"] == d["G_warn"]).mean())
    subs = [sub(ix) for ix in boot_idx[:N_BOOT]]
    for arm in ("L", "G"):
        v = [float((s[f"{arm}_warn"] == s["M_warn"]).mean()) for s in subs]
        pooled[f"A_{arm}_ci"] = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    # overrides: points where G disagrees with M
    dis = d[d["G_warn"] != d["M_warn"]]
    pooled["G_overrides"] = int(len(dis))
    pooled["G_overrides_correct"] = int((dis["G_warn"] == dis["late"]).sum())
    pooled["G_overrides_to_warn"] = int((dis["G_warn"] == 1).sum())
    for c in (0.0, 0.5):
        for arm in ("L", "G"):
            v = []
            for s in subs:
                g = s[s.checkpoint == c]
                if 0 < g.late.sum() < len(g):
                    v.append(roc_auc_score(g.late, g[f"{arm}_p"]) - roc_auc_score(g.late, g.M_p))
            g = d[d.checkpoint == c]
            pooled[f"auc_{arm}_minus_M_c{int(c * 100)}"] = float(
                roc_auc_score(g.late, g[f"{arm}_p"]) - roc_auc_score(g.late, g.M_p))
            pooled[f"auc_{arm}_minus_M_c{int(c * 100)}_ci"] = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    summary["pooled"] = pooled

    curve = []
    for k in RATIOS:
        pt = costs(r, k, wts)
        bd, bg = [], []
        for ix in boot_idx:
            rr, ww = r.iloc[ix], wts[ix]
            cc = costs(rr, k, ww)
            bd.append(cc["delta"]); bg.append(cc["G_minus_M"])
        curve.append({"kappa_over_F": k, **{f"cost_{a}": pt[a] for a in ("M", "L", "G")},
                      "delta": pt["delta"], "delta_lo": float(np.percentile(bd, 2.5)),
                      "delta_hi": float(np.percentile(bd, 97.5)),
                      "M_minus_G": pt["G_minus_M"], "M_minus_G_lo": float(np.percentile(bg, 2.5)),
                      "M_minus_G_hi": float(np.percentile(bg, 97.5))})
    cv = pd.DataFrame(curve)
    cv.to_csv(OUT / "cost_curve.csv", index=False)
    summary["arm_metrics"] = am.round(3).to_dict("records")
    summary["cost_curve"] = cv.round(4).to_dict("records")
    summary["warned_any"] = {a: int(r[f"{a}_w"].sum()) for a in ("M", "L", "G")}
    (OUT / "pilot_summary.json").write_text(json.dumps(summary, indent=1))
    pd.set_option("display.width", 200)
    print(am.round(3).to_string(index=False))
    print(json.dumps(pooled, indent=1))
    print(cv.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
