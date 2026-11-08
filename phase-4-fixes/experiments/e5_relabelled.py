"""Phase 4, experiment 5: rerun with the corrected labels (added 8 Nov 2026).

Labels: see phase4/relabel.py. In short, late closure gets a next-working-day
grace and spillover is replaced by heavy carry-over (half or more of the
committed issues unfinished at the planned end), without Confluence Cloud.

Every learned model is compared with rules that need no training:

  committed count      the number of committed issues (bigger = riskier)
  past lateness        share of the team's last 3 known sprints that were late
  past carry share     1 - average completion ratio of the last 3 known sprints
  remaining count      open committed issues at the checkpoint (in-sprint)
  remaining share      1 - share done at the checkpoint (in-sprint)

Three parts:

  A. Planning time, TAWOS. Shuffled 70/30 and time-ordered cuts (5 repeats
     each, as in experiment 1), and Phase 1's fixed time split with 95%
     bootstrap intervals and a paired bootstrap of the gap between gradient
     boosting and the best rule. The "best rule" is picked on the training
     side, never on the test side.
  B. In-sprint checkpoints (25%, 50%, 75%), Phase 3's frame and split, sprints
     still open at the checkpoint, Confluence Cloud excluded (progress is read
     from resolution dates).
  C. The 10 outside projects, trained on TAWOS only, as in experiment 3.

    python -m experiments.e5_relabelled        (from the phase-4-fixes folder)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

import phase4
import phase2.data as p2d
from phase2.models import SCOPE_ONLY, feature_columns, make_classifier
from phase4.common import MODELS, N_REPEATS, SEED, decision_metrics, fit_score, youden_threshold
from phase4.relabel import BROKEN_RESOLUTION, load_relabelled_frame, population

N_BOOT = 2000
MIN_CLASS = 5
TIME_CUTS = [0.4, 0.5, 0.6, 0.7, 0.8]
TASKS = {"late": "label_late", "carry": "label_heavy_carry"}
TASK_NAMES = {"late": "Late closure (next working day)", "carry": "Heavy carry-over (>= 50% unfinished)"}
SENSITIVITY = {"late": ["label_late_3d", "label_late_1d"],
               "carry": ["label_carry_20", "label_carry_30", "label_spillover"]}
PROGRESS = ["elapsed_frac", "remaining_days", "done_share", "done_n", "remaining_n", "remaining_sp", "burn_gap",
            "required_rate", "observed_rate", "added_ratio", "added_done_n", "removed_ratio", "sp_change_ratio"]


# ---------------------------------------------------------------- rules
def rule_scores(task: str, d: pd.DataFrame, fill: dict | None = None) -> dict[str, np.ndarray]:
    fill = fill or {}
    out = {"Committed count (rule)": d["committed_issue_count"].to_numpy(float)}
    if task == "late":
        out["Past lateness (rule)"] = d["hist_late_rate_3"].fillna(fill.get("late", 0.0)).to_numpy(float)
    else:
        out["Past carry share (rule)"] = (1 - d["hist_avg_completion_ratio_3"]).fillna(
            fill.get("carry", 0.5)).to_numpy(float)
    if "remaining_n" in d:
        out["Remaining count (rule)"] = d["remaining_n"].to_numpy(float)
        out["Remaining share (rule)"] = (1 - d["done_share"]).to_numpy(float)
    return out


def rule_fill(train: pd.DataFrame) -> dict:
    return {"late": float(train["hist_late_rate_3"].mean()),
            "carry": float((1 - train["hist_avg_completion_ratio_3"]).mean())}


def safe_auc(y, s):
    y = np.asarray(y, int)
    return roc_auc_score(y, s) if 0 < y.sum() < len(y) else np.nan


def evaluate(task, train, test, target, seed=SEED, models=MODELS):
    """Learned models + rules on one split. Thresholds come from the training side only."""
    rows, scores = [], {}
    for m in models:
        s, thr = fit_score(m, train, test, target, seed=seed)
        scores[m] = s
        rows.append({"model": m, **decision_metrics(test[target], s, (s >= thr).astype(int)), "threshold": thr})
    fill = rule_fill(train)
    tr_rules, te_rules = rule_scores(task, train, fill), rule_scores(task, test, fill)
    y_tr = train[target].to_numpy(int)
    for name, s in te_rules.items():
        thr = youden_threshold(y_tr, tr_rules[name])
        scores[name] = s
        r = {"model": name, **decision_metrics(test[target], s, (s >= thr).astype(int)), "threshold": thr,
             "train_auc": safe_auc(y_tr, tr_rules[name])}
        rows.append(r)
    return rows, scores


# ---------------------------------------------------------------- splits
def shuffled(d, target, k):
    return train_test_split(d, test_size=0.3, stratify=d[target], random_state=SEED + k)


def time_cut(d, target, k):
    lo, hi = TIME_CUTS[k], TIME_CUTS[k] + 0.2
    tr, te = [], []
    for _, g in d.groupby("project"):
        g = g.sort_values("start_date")
        a, b = int(round(len(g) * lo)), int(round(len(g) * hi))
        tr.append(g.iloc[:a]); te.append(g.iloc[a:b])
    return pd.concat(tr), pd.concat(te)


SCHEMES = {"A. Random 70/30 (shuffled)": shuffled, "C. Time-ordered (leak-safe)": time_cut}


def boot_ci(y, s, rng):
    y, s = np.asarray(y, int), np.asarray(s, float)
    v = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(y):
            v.append(roc_auc_score(y[i], s[i]))
    return tuple(np.percentile(v, [2.5, 97.5])) if v else (np.nan, np.nan)


def paired_delta(y, s1, s2, rng):
    y = np.asarray(y, int)
    v = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(y):
            v.append(roc_auc_score(y[i], s1[i]) - roc_auc_score(y[i], s2[i]))
    v = np.asarray(v)
    return float(np.mean(v)), *np.percentile(v, [2.5, 97.5]), float((v <= 0).mean())


def best_rule(rows):
    """Best rule by TRAINING-side AUC (the test side is never used to choose)."""
    r = [x for x in rows if x["model"].endswith("(rule)")]
    return max(r, key=lambda x: x["train_auc"])["model"]


# ---------------------------------------------------------------- part A
def part_a(d, rng):
    reps, fixed, deltas, sens = [], [], [], []
    for task, target in TASKS.items():
        pop = population(d, task)
        for scheme, split in SCHEMES.items():
            for k in range(N_REPEATS):
                tr, te = split(pop, target, k)
                rows, _ = evaluate(task, tr, te, target, seed=SEED + k)
                reps += [{"task": task, "scheme": scheme, "repeat": k + 1, **r} for r in rows]
            print("A", task, scheme, "done", flush=True)
        # Phase 1's fixed time split (first 70% of each project)
        tr, te = p2d.time_ordered_split(pop)
        rows, scores = evaluate(task, tr, te, target)
        y = te[target].to_numpy(int)
        for r in rows:
            lo, hi = boot_ci(y, scores[r["model"]], rng)
            fixed.append({"task": task, **r, "auc_lo": lo, "auc_hi": hi})
        br = best_rule(rows)
        m, lo, hi, p = paired_delta(y, scores["Gradient boosting"], scores[br], rng)
        deltas.append({"task": task, "split": "fixed time split", "checkpoint": 0.0, "model": "Gradient boosting",
                       "rule": br, "delta_auc": m, "lo": lo, "hi": hi, "p_le_0": p, "n": len(y),
                       "n_pos": int(y.sum())})
        # other definitions of the same outcome, same population and time cuts
        for alt in SENSITIVITY[task]:
            for k in range(N_REPEATS):
                tr, te = time_cut(pop, alt, k)
                rows, _ = evaluate(task, tr, te, alt, seed=SEED + k,
                                   models={m: MODELS[m] for m in ("Gradient boosting", "Scope-only LR (baseline)")})
                sens += [{"task": task, "label": alt, "repeat": k + 1, **r} for r in rows]
        print("A", task, "fixed split and sensitivity done", flush=True)
    return pd.DataFrame(reps), pd.DataFrame(fixed), deltas, pd.DataFrame(sens)


# ---------------------------------------------------------------- part B
def checkpoint_frame(d):
    cp = pd.read_csv(phase4.ROOT.parent / "phase-3-in-sprint-early-warning" / "results" / "phase3_checkpoint_frame.csv")
    keep = ["sprint_id", "checkpoint", "open_at_checkpoint", "split_full", "split_nonempty"] + PROGRESS
    cp = cp[keep].merge(d, on="sprint_id", how="inner")
    return cp[~cp["project"].isin(BROKEN_RESOLUTION)].reset_index(drop=True)


def part_b(d, rng):
    cp = checkpoint_frame(d)
    feats = {"Gradient boosting, planning features": ("gradient_boosting", p2d.STRICT_NUMERIC),
             "Gradient boosting, progress features": ("gradient_boosting", p2d.STRICT_NUMERIC + PROGRESS),
             "Logistic regression, progress features": ("logreg", p2d.STRICT_NUMERIC + PROGRESS)}
    rows, deltas = [], []
    for task, target in TASKS.items():
        side = "split_full" if task == "late" else "split_nonempty"
        for c in (0.25, 0.5, 0.75):
            g = cp[(cp["checkpoint"] == c) & cp["open_at_checkpoint"] & cp[side].notna()]
            if task == "carry":
                g = g[g["carry_valid"]]
            tr, te = g[g[side] == "train"], g[g[side] == "test"]
            y_tr, y = tr[target].to_numpy(int), te[target].to_numpy(int)
            scores, part = {}, []
            for name, (algo, num) in feats.items():
                est, cols = make_classifier(algo, num, True), feature_columns(num, True)
                oof = cross_val_predict(est, tr[cols], y_tr, cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                                        method="predict_proba")[:, 1]
                thr = youden_threshold(y_tr, oof)
                est.fit(tr[cols], y_tr)
                s = est.predict_proba(te[cols])[:, 1]
                scores[name] = s
                part.append({"model": name, **decision_metrics(y, s, (s >= thr).astype(int))})
            fill = rule_fill(tr)
            r_tr, r_te = rule_scores(task, tr, fill), rule_scores(task, te, fill)
            for name, s in r_te.items():
                thr = youden_threshold(y_tr, r_tr[name])
                scores[name] = s
                part.append({"model": name, **decision_metrics(y, s, (s >= thr).astype(int)),
                             "train_auc": safe_auc(y_tr, r_tr[name])})
            for r in part:
                lo, hi = boot_ci(y, scores[r["model"]], rng)
                rows.append({"task": task, "checkpoint": c, **r, "auc_lo": lo, "auc_hi": hi})
            br = best_rule(part)
            m, lo, hi, p = paired_delta(y, scores["Gradient boosting, progress features"], scores[br], rng)
            deltas.append({"task": task, "split": "Phase 3 time split", "checkpoint": c,
                           "model": "Gradient boosting, progress features", "rule": br, "delta_auc": m,
                           "lo": lo, "hi": hi, "p_le_0": p, "n": len(y), "n_pos": int(y.sum())})
            print("B", task, c, "done", flush=True)
    return pd.DataFrame(rows), deltas


# ---------------------------------------------------------------- part C
EXT = phase4.ROOT / "external" / "data" / "raw"
SP_COLS = {"committed_story_points", "hist_avg_delivered_sp_3"}
NO_SP = [c for c in p2d.STRICT_NUMERIC if c not in SP_COLS]
EXT_MODELS = {"Gradient boosting": ("gradient_boosting", NO_SP), "Random forest": ("random_forest", NO_SP),
              "Logistic regression": ("logreg", NO_SP),
              "Scope-only LR (baseline)": ("logreg", [c for c in SCOPE_ONLY if c not in SP_COLS])}


def part_c(tawos, rng):
    ext = []
    for inst in ("mariadb", "mongodb"):
        b = load_relabelled_frame(EXT / inst)
        b["instance"] = inst
        ext.append(b)
    ext = pd.concat(ext, ignore_index=True)
    rows = []
    for task, target in TASKS.items():
        tr, te_all = population(tawos, task), population(ext, task)
        y_tr = tr[target].to_numpy(int)
        fitted = {}
        for m, (algo, num) in EXT_MODELS.items():
            est, cols = make_classifier(algo, num, False), feature_columns(num, False)
            oof = cross_val_predict(est, tr[cols], y_tr, cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                                    method="predict_proba")[:, 1]
            fitted[m] = (est.fit(tr[cols], y_tr), cols, youden_threshold(y_tr, oof))
        fill = rule_fill(tr)
        r_tr = rule_scores(task, tr, fill)
        for proj, te in te_all.groupby("project"):
            y = te[target].to_numpy(int)
            sc = {m: (e.predict_proba(te[c])[:, 1], t) for m, (e, c, t) in fitted.items()}
            for name, s in rule_scores(task, te, fill).items():
                sc[name] = (s, youden_threshold(y_tr, r_tr[name]))
            for m, (s, thr) in sc.items():
                r = decision_metrics(y, s, (s >= thr).astype(int))
                ok = r["n_pos"] >= MIN_CLASS and r["n_neg"] >= MIN_CLASS
                lo, hi = boot_ci(y, s, rng) if ok else (np.nan, np.nan)
                rows.append({"task": task, "model": m, "instance": te["instance"].iloc[0], "project": proj, **r,
                             "interpretable": ok, "auc_lo": lo, "auc_hi": hi,
                             "ci_above_half": bool(ok and lo > 0.5)})
        print("C", task, "done", flush=True)
    res = pd.DataFrame(rows)
    desc = ext.groupby(["instance", "project"]).agg(
        sprints=("sprint_id", "size"), nonempty=("nonempty", "sum"), late_1d=("label_late_1d", "mean"),
        late_next_working_day=("label_late", "mean")).reset_index()
    ne = ext[ext["nonempty"]].groupby("project").agg(any_spillover=("label_spillover", "mean"),
                                                     heavy_carry=("label_heavy_carry", "mean"))
    return res, desc.merge(ne, left_on="project", right_index=True, how="left")


# ---------------------------------------------------------------- outputs
def fmt(m, s):
    return f"{m:.3f} ± {s:.3f}"


def label_table(d):
    t = d.groupby("project").agg(
        sprints=("sprint_id", "size"), late_1d=("label_late_1d", "mean"), late_3d=("label_late_3d", "mean"),
        late_next_working_day=("label_late", "mean")).reset_index()
    ne = d[d["nonempty"]].groupby("project").agg(
        nonempty=("sprint_id", "size"), any_spillover=("label_spillover", "mean"),
        carry_20=("label_carry_20", "mean"), carry_30=("label_carry_30", "mean"),
        heavy_carry=("label_heavy_carry", "mean"), median_carry_share=("carry_ratio", "median")).reset_index()
    t = t.merge(ne, on="project", how="left")
    t["resolution_dates_usable"] = ~t["project"].isin(BROKEN_RESOLUTION)
    allrow = {"project": "All 8 projects", "sprints": len(d), "late_1d": d.label_late_1d.mean(),
              "late_3d": d.label_late_3d.mean(), "late_next_working_day": d.label_late.mean()}
    v = d[d["carry_valid"]]
    allrow.update({"nonempty": len(v), "any_spillover": v.label_spillover.mean(), "carry_20": v.label_carry_20.mean(),
                   "carry_30": v.label_carry_30.mean(), "heavy_carry": v.label_heavy_carry.mean(),
                   "median_carry_share": v.carry_ratio.median(), "resolution_dates_usable": True})
    return pd.concat([t, pd.DataFrame([allrow])], ignore_index=True)


def main():
    phase4.RESULTS.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    d = load_relabelled_frame()
    out = phase4.RESULTS
    label_table(d).round(3).to_csv(out / "e5_label_definitions.csv", index=False)

    reps, fixed, deltas_a, sens = part_a(d, rng)
    reps.to_csv(out / "e5_repeats.csv", index=False)
    keys = ["auc", "sensitivity", "specificity", "f1"]
    summ = reps.groupby(["task", "scheme", "model"])[keys].agg(["mean", "std"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ = summ.reset_index()
    summ.round(3).to_csv(out / "e5_summary.csv", index=False)
    pretty = summ[["task", "scheme", "model"]].copy()
    for k in keys:
        pretty[k] = [fmt(a, b) for a, b in zip(summ[f"{k}_mean"], summ[f"{k}_std"])]
    pretty.to_csv(out / "e5_summary_pretty.csv", index=False)
    fixed.round(4).to_csv(out / "e5_fixed_split.csv", index=False)
    ss = sens.groupby(["task", "label", "model"])["auc"].agg(["mean", "std"]).reset_index()
    base = reps[reps.scheme.str.startswith("C")].groupby(["task", "model"])["auc"].agg(["mean", "std"]).reset_index()
    base.insert(1, "label", base["task"].map(TASKS))
    pd.concat([base, ss]).round(3).to_csv(out / "e5_label_sensitivity.csv", index=False)

    ins, deltas_b = part_b(d, rng)
    ins.round(4).to_csv(out / "e5_insprint.csv", index=False)
    pd.DataFrame(deltas_a + deltas_b).round(4).to_csv(out / "e5_model_vs_best_rule.csv", index=False)

    ext, desc = part_c(d, rng)
    ext.round(4).to_csv(out / "e5_external_per_project.csv", index=False)
    desc.round(3).to_csv(out / "e5_external_projects.csv", index=False)
    ok = ext[ext["interpretable"]]
    avg = ok.groupby(["task", "model"]).agg(projects=("auc", "size"), auc_mean=("auc", "mean"),
                                            auc_std=("auc", "std"), ci_above_half=("ci_above_half", "sum"),
                                            auc_below_half=("auc", lambda a: int((a < 0.5).sum()))).reset_index()
    avg.round(3).to_csv(out / "e5_external_average.csv", index=False)
    figure()
    pd.set_option("display.width", 250)
    print(pretty.to_string(index=False))


def figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ins = pd.read_csv(phase4.RESULTS / "e5_insprint.csv")
    show = {"Gradient boosting, progress features": ("#1d4ed8", "o", "-"),
            "Gradient boosting, planning features": ("#1d4ed8", "o", ":"),
            "Remaining count (rule)": ("#b45309", "s", "-"),
            "Remaining share (rule)": ("#b45309", "s", ":"),
            "Committed count (rule)": ("#6b7280", "^", "--")}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, task in zip(axes, TASKS):
        r = ins[ins.task == task]
        for i, (m, (c, mk, ls)) in enumerate(show.items()):
            g = r[r.model == m].sort_values("checkpoint")
            x = g.checkpoint * 100 + (i - 2) * 0.8
            ax.errorbar(x, g.auc, yerr=[g.auc - g.auc_lo, g.auc_hi - g.auc], c=c, marker=mk, ls=ls,
                        capsize=2, lw=1.4, label=m.replace(" (rule)", ", no model"))
        ax.axhline(0.5, c="#9ca3af", ls=":", lw=1)
        n = r[r.model == "Committed count (rule)"]
        ax.set_title(f"{TASK_NAMES[task]}\n(test n = {int(n.n.min())}-{int(n.n.max())})", fontsize=10)
        ax.set_xticks([25, 50, 75])
        ax.set_xlabel("Checkpoint (% of planned sprint length)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("AUC with 95% bootstrap interval")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, fontsize=8.5, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("In-sprint warning with the corrected labels: model vs rules that need no training", fontsize=11)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(phase4.FIGURES / "e5_insprint_vs_rules.png", dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()
