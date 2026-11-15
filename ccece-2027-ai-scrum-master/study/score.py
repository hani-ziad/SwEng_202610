"""Step 3 of the study: score the arms over three runs.

Per run, arm and checkpoint: warn rate, sensitivity, specificity, AUC of the stated probability
(pooled and averaged within projects), mean stated probability and Brier score (calibration;
the sample's late rate is close to the pool's), agreement with M (raw and Cohen's kappa), and, for the
unfinished share, Spearman correlation and mean absolute error (MAE) against the truth on the
seven projects with resolution dates. Baselines: M itself; the rule "warn if at least half of the
team's last three sprints closed late"; and for the unfinished share, the team's own recent
unfinished share (1 - average completion ratio of its last three sprints).

Across runs: mean and range of every metric, how often the three runs gave the same call, and
the spread of stated probabilities. Intervals: 2,000 bootstrap resamples of sprints, drawn
within project, on run-averaged quantities. Costs use the alarm cost model of the paper,
averaged over runs and reweighted to the pool of 472 eligible sprints by project.

Writes study/results/: replay_all_arms.csv, arm_metrics_by_run.csv, summary.json,
messages.csv (every warning with its board), cost_curve.csv.

Usage (from the ccece-2027-ai-scrum-master folder):  python -m study.score
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

import study

SEED = 2027
N_BOOT = 2000
RUNS = (1, 2, 3)
E = {0.0: 0.5, 0.5: 0.25}
RATIOS = [0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
OUT = study.HERE / "results"


def kappa(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    po = (a == b).mean()
    pa, pb = a.mean(), b.mean()
    pe = pa * pb + (1 - pa) * (1 - pb)
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def fleiss(mat):
    """Fleiss' kappa for binary ratings, mat = items x raters (0/1)."""
    mat = np.asarray(mat, float)
    n = mat.shape[1]
    p1 = mat.sum(1)
    p_i = (p1 * (p1 - 1) + (n - p1) * (n - p1 - 1)) / (n * (n - 1))
    pbar = p_i.mean()
    q = mat.mean()
    pe = q ** 2 + (1 - q) ** 2
    return float((pbar - pe) / (1 - pe))


def within_auc(y, s, groups):
    vals = []
    for g in np.unique(groups):
        k = groups == g
        if y[k].sum() >= 2 and (1 - y[k]).sum() >= 2:
            vals.append(roc_auc_score(y[k], s[k]))
    return float(np.mean(vals)) if vals else float("nan"), len(vals)


def load():
    rp = pd.read_csv(OUT / "replay_points.csv")
    man = json.loads((study.HERE / "prompts" / "manifest.json").read_text())
    rows = []
    for m in man:
        f = study.HERE / "agent_outputs" / f"run{m['run']}" / f"{m['prompt']}.jsonl"
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            g = json.loads(line)
            rows.append({"run": m["run"], "board_id": g["id"], "checkpoint": m["checkpoint"], "arm": m["arm"],
                         "warn": int(bool(g["warn"])), "p_late": float(g["p_late"]),
                         "unf": float(g["unfinished"]), "message": g.get("message", "") or ""})
    ag = pd.DataFrame(rows)
    assert len(ag) == len(rp) * 2 * len(RUNS), len(ag)
    wide = rp.copy()
    for run in RUNS:
        for arm in ("L", "G"):
            a = ag[(ag.run == run) & (ag.arm == arm)].drop(columns=["run", "arm"])
            a = a.rename(columns={"warn": f"{arm}{run}_warn", "p_late": f"{arm}{run}_p",
                                  "unf": f"{arm}{run}_unf", "message": f"{arm}{run}_msg"})
            wide = wide.merge(a, on=["board_id", "checkpoint"], how="left")
    assert not wide.filter(like="_warn").isna().any().any()
    return wide, ag


def arm_cols(arm, run):
    if arm == "M":
        return "m_warn", "m_score", "m_unf"
    if arm == "rule":
        return "rule_warn", "hist_late_rate_3", "hist_unf_f"
    return f"{arm}{run}_warn", f"{arm}{run}_p", f"{arm}{run}_unf"


def metrics(d, arm, run):
    out = []
    for c, g in d.groupby("checkpoint"):
        w, p, u = arm_cols(arm, run)
        y = g.late.to_numpy()
        wv, pv = g[w].to_numpy(), g[p].to_numpy(float)
        wa, n_proj = within_auc(y, pv, g.project.to_numpy())
        s = g[g.unf_scored == 1]
        out.append({"run": run, "arm": arm, "checkpoint": c, "warn_rate": wv.mean(),
                    "sensitivity": wv[y == 1].mean(), "specificity": 1 - wv[y == 0].mean(),
                    "auc": roc_auc_score(y, pv), "auc_within_project": wa, "n_projects_within": n_proj,
                    "mean_p": float(pv.mean()), "late_rate": float(y.mean()),
                    "brier": float(np.mean((pv - y) ** 2)),
                    "agree_M": (wv == g.m_warn.to_numpy()).mean(), "kappa_M": kappa(wv, g.m_warn),
                    "unf_spearman": spearmanr(s[u], s.unfinished).statistic,
                    "unf_mae": float(np.abs(s[u] - s.unfinished).mean())})
    return out


def costs(d, wcol_by_cp, ratio):
    """Mean cost per sprint in units of F, weighted to the pool. wcol_by_cp: {c: column} of warn flags."""
    a = d[d.checkpoint == 0.0].set_index("sprint_id")
    b = d[d.checkpoint == 0.5].set_index("sprint_id").loc[a.index]
    w0 = a[wcol_by_cp[0.0]].to_numpy()
    w5 = b[wcol_by_cp[0.5]].to_numpy()
    warned = ((w0 + w5) > 0).astype(int)
    eff = np.where(w0 > 0, E[0.0], np.where(w5 > 0, E[0.5], 0.0))
    per = ratio * warned + a["late"].to_numpy() * (1 - eff)
    return float(np.average(per, weights=a["weight"].to_numpy()))


def main():
    d, ag = load()
    d["rule_warn"] = (d.hist_late_rate_3 >= 0.5).astype(int)
    fill = d.loc[d.checkpoint == 0.0, "hist_unf"].mean()
    d["hist_unf_f"] = d.hist_unf.fillna(fill)
    d.to_csv(OUT / "replay_all_arms.csv", index=False)

    rows = metrics(d, "M", 0) + metrics(d, "rule", 0)
    for run in RUNS:
        rows += metrics(d, "L", run) + metrics(d, "G", run)
    am = pd.DataFrame(rows)
    am.to_csv(OUT / "arm_metrics_by_run.csv", index=False)

    summ = {"n_sprints": int(d.sprint_id.nunique()), "n_late": int(d[d.checkpoint == 0].late.sum()),
            "runs": len(RUNS), "hist_unf_missing": int(d[d.checkpoint == 0].hist_unf.isna().sum())}
    # mean and range over runs, per arm and checkpoint
    agg = {}
    for (arm, c), g in am.groupby(["arm", "checkpoint"]):
        agg[f"{arm}_c{int(c * 100)}"] = {k: {"mean": float(g[k].mean()), "min": float(g[k].min()),
                                             "max": float(g[k].max())}
                                         for k in ["warn_rate", "sensitivity", "specificity", "auc", "mean_p", "brier",
                                                   "auc_within_project", "agree_M", "kappa_M",
                                                   "unf_spearman", "unf_mae"]}
    summ["by_arm"] = agg

    # pooled over both checkpoints, per run
    pooled = {}
    for arm in ("L", "G"):
        a = [float((d[f"{arm}{r}_warn"] == d.m_warn).mean()) for r in RUNS]
        k = [kappa(d[f"{arm}{r}_warn"], d.m_warn) for r in RUNS]
        pooled[f"A_{arm}"] = {"mean": float(np.mean(a)), "min": min(a), "max": max(a)}
        pooled[f"kappa_{arm}"] = {"mean": float(np.mean(k)), "min": min(k), "max": max(k)}
        # departures from M
        dep_n, dep_ok, dep_silent = [], [], []
        for r in RUNS:
            dd = d[d[f"{arm}{r}_warn"] != d.m_warn]
            dep_n.append(len(dd)); dep_ok.append(int((dd[f"{arm}{r}_warn"] == dd.late).sum()))
            dep_silent.append(int((dd[f"{arm}{r}_warn"] == 0).sum()))
        pooled[f"departures_{arm}"] = {"n": dep_n, "correct": dep_ok, "to_silent": dep_silent,
                                       "n_points": int(len(d))}
        # run-to-run consistency
        mat = d[[f"{arm}{r}_warn" for r in RUNS]].to_numpy()
        pooled[f"same_call_all_runs_{arm}"] = float((mat.min(1) == mat.max(1)).mean())
        pooled[f"fleiss_runs_{arm}"] = fleiss(mat)
        pooled[f"p_late_sd_{arm}"] = float(d[[f"{arm}{r}_p" for r in RUNS]].std(axis=1, ddof=1).mean())
        pooled[f"unf_sd_{arm}"] = float(d[[f"{arm}{r}_unf" for r in RUNS]].std(axis=1, ddof=1).mean())
        # rule agreement at c = 0
        g0 = d[d.checkpoint == 0.0]
        pooled[f"agree_rule_{arm}_c0"] = [float((g0[f"{arm}{r}_warn"] == g0.rule_warn).mean()) for r in RUNS]
    g0 = d[d.checkpoint == 0.0]
    pooled["agree_rule_M_c0"] = float((g0.m_warn == g0.rule_warn).mean())
    # calls that changed between checkpoints
    ch = {}
    for name, col in [("M", "m_warn")] + [(f"{a}{r}", f"{a}{r}_warn") for a in "LG" for r in RUNS]:
        p = d.pivot_table(index="sprint_id", columns="checkpoint", values=col)
        ch[name] = int((p[0.0] != p[0.5]).sum())
    pooled["sprints_with_changed_call"] = ch
    summ["pooled"] = pooled

    # bootstrap on run-averaged quantities, resampling sprints within project
    for arm in ("L", "G"):
        d[f"{arm}_pbar"] = d[[f"{arm}{r}_p" for r in RUNS]].mean(axis=1)
        d[f"{arm}_ubar"] = d[[f"{arm}{r}_unf" for r in RUNS]].mean(axis=1)
        d[f"{arm}_abar"] = np.mean([(d[f"{arm}{r}_warn"] == d.m_warn) for r in RUNS], axis=0)
    sprints = d[["sprint_id", "project"]].drop_duplicates()
    groups = {p: g.sprint_id.to_numpy() for p, g in sprints.groupby("project")}
    by_sprint = {s: g for s, g in d.groupby("sprint_id")}
    rng = np.random.default_rng(SEED)

    def stats(x):
        o = {}
        for arm in ("L", "G"):
            o[f"A_{arm}"] = x[f"{arm}_abar"].mean()
        for c in (0.0, 0.5):
            g = x[x.checkpoint == c]
            if g.late.nunique() < 2:
                return None
            am_ = roc_auc_score(g.late, g.m_score)
            for arm in ("L", "G"):
                o[f"auc_{arm}_minus_M_c{int(c * 100)}"] = roc_auc_score(g.late, g[f"{arm}_pbar"]) - am_
            o[f"auc_M_minus_rule_c{int(c * 100)}"] = am_ - roc_auc_score(g.late, g.hist_late_rate_3)
            s = g[g.unf_scored == 1]
            rm = spearmanr(s.m_unf, s.unfinished).statistic
            rh = spearmanr(s.hist_unf_f, s.unfinished).statistic
            for arm in ("L", "G"):
                ra = spearmanr(s[f"{arm}_ubar"], s.unfinished).statistic
                o[f"unf_rho_{arm}_minus_M_c{int(c * 100)}"] = ra - rm
                o[f"unf_rho_{arm}_minus_hist_c{int(c * 100)}"] = ra - rh
                o[f"unf_mae_{arm}_minus_M_c{int(c * 100)}"] = (np.abs(s[f"{arm}_ubar"] - s.unfinished).mean()
                                                              - np.abs(s.m_unf - s.unfinished).mean())
            o[f"unf_rho_M_minus_hist_c{int(c * 100)}"] = rm - rh
        return o

    point = stats(d)
    boots = []
    for _ in range(N_BOOT):
        ids = np.concatenate([rng.choice(v, len(v), replace=True) for v in groups.values()])
        x = pd.concat([by_sprint[i] for i in ids], ignore_index=True)
        s = stats(x)
        if s is not None:
            boots.append(s)
    bt = pd.DataFrame(boots)
    summ["contrasts"] = {k: {"point": float(v), "lo": float(bt[k].quantile(0.025)),
                             "hi": float(bt[k].quantile(0.975))} for k, v in point.items()}
    summ["n_boot_used"] = len(bt)

    # cost curve, averaged over runs, with bootstrap on the run-averaged difference
    curve = []
    sp = d.sprint_id.unique()
    for ratio in RATIOS:
        cM = costs(d, {0.0: "m_warn", 0.5: "m_warn"}, ratio)
        cL = np.mean([costs(d, {0.0: f"L{r}_warn", 0.5: f"L{r}_warn"}, ratio) for r in RUNS])
        cG = np.mean([costs(d, {0.0: f"G{r}_warn", 0.5: f"G{r}_warn"}, ratio) for r in RUNS])
        bd, bg = [], []
        rr = np.random.default_rng(SEED + 1)
        for _ in range(500):
            ids = np.concatenate([rr.choice(v, len(v), replace=True) for v in groups.values()])
            x = pd.concat([by_sprint[i].assign(sprint_id=f"{i}#{k}") for k, i in enumerate(ids)], ignore_index=True)
            m_ = costs(x, {0.0: "m_warn", 0.5: "m_warn"}, ratio)
            l_ = np.mean([costs(x, {0.0: f"L{r}_warn", 0.5: f"L{r}_warn"}, ratio) for r in RUNS])
            g_ = np.mean([costs(x, {0.0: f"G{r}_warn", 0.5: f"G{r}_warn"}, ratio) for r in RUNS])
            bd.append(min(m_, l_) - g_); bg.append(m_ - g_)
        curve.append({"kappa_over_F": ratio, "cost_M": cM, "cost_L": cL, "cost_G": cG,
                      "delta": min(cM, cL) - cG, "delta_lo": float(np.percentile(bd, 2.5)),
                      "delta_hi": float(np.percentile(bd, 97.5)), "M_minus_G": cM - cG,
                      "M_minus_G_lo": float(np.percentile(bg, 2.5)), "M_minus_G_hi": float(np.percentile(bg, 97.5))})
    cv = pd.DataFrame(curve)
    cv.to_csv(OUT / "cost_curve.csv", index=False)
    summ["cost_curve"] = cv.round(4).to_dict("records")

    # every warning, for the fact check and the coding
    boards = {(b["board_id"], b["checkpoint"]): b for b in map(json.loads, (OUT / "boards.jsonl").read_text().splitlines())}
    msgs = ag[ag.warn == 1].copy()
    msgs["board_text"] = [boards[(b, c)]["text"] for b, c in zip(msgs.board_id, msgs.checkpoint)]
    msgs["model_text"] = [boards[(b, c)]["model_text"] for b, c in zip(msgs.board_id, msgs.checkpoint)]
    lab = d.drop_duplicates(["board_id", "checkpoint"]).set_index(["board_id", "checkpoint"])
    msgs["late"] = [int(lab.loc[(b, c), "late"]) for b, c in zip(msgs.board_id, msgs.checkpoint)]
    msgs = msgs[["run", "arm", "board_id", "checkpoint", "late", "message", "board_text", "model_text"]]
    msgs.to_csv(OUT / "messages.csv", index=False)
    summ["n_warnings"] = {f"{a}{r}": int(((msgs.arm == a) & (msgs.run == r)).sum()) for a in "LG" for r in RUNS}
    summ["n_warnings_total"] = int(len(msgs))

    (OUT / "summary.json").write_text(json.dumps(summ, indent=1, default=float))
    pd.set_option("display.width", 220)
    print(am.groupby(["arm", "checkpoint"]).mean(numeric_only=True).round(3).drop(columns="run").to_string())
    print(json.dumps(summ["pooled"], indent=1, default=float))
    print(json.dumps({k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in summ["contrasts"].items()}, indent=1))
    print(cv.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
