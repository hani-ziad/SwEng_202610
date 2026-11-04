"""Phase 4, experiment 3: out-of-domain test on 10 external Jira projects.

Feedback: test on about 10 projects that were never used for training, and for
each one report sensitivity, specificity and F1.

Plan (fixed before any external outcome was looked at):
  * Train once on all 8 TAWOS projects (all 2,602 / 1,691 sprints), with Phase 2's
    strict planning-time features. No project one-hot (the test projects are new)
    and no story-point features (MariaDB records none), on both sides.
  * The decision threshold is fixed on TAWOS only: Youden's J on 5-fold
    out-of-fold TAWOS predictions. It is never tuned on an external project.
  * Test on every sprint of each external project that passes the frozen Phase 1
    cohort rules: late closure on all sprints, spillover on sprints with
    committed work.
  * Per project: n, positives, negatives, AUC, sensitivity, specificity, F1, with
    95% bootstrap intervals over that project's sprints (2,000 resamples).
    Averages across projects are given as mean +/- SD.
  * A metric is marked not interpretable when the project has fewer than 5
    sprints in either class (it cannot be estimated from so few).

External data: MariaDB Jira (MaxScale, ColumnStore) and MongoDB Jira (8 team
projects), pulled with external/pull_jira_browser.js and converted with
external/convert_jira.py into external/data/raw/<instance>/.

    python -m experiments.e3_external
"""
from __future__ import annotations

import functools
import json

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

import phase4  # noqa: I001  (must come first: puts Phases 1-3 on the path)
import phase3
import phase2.data as p2d
from phase2.models import SCOPE_ONLY
from phase4.common import SEED, decision_metrics, youden_threshold
from phase2.models import feature_columns, make_classifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict

EXT = phase4.ROOT / "external" / "data" / "raw"
INSTANCES = ["mariadb", "mongodb"]
SP_COLS = {"committed_story_points", "hist_avg_delivered_sp_3", "prior_avg_delivered_sp_3"}
STRICT_NO_SP = [c for c in p2d.STRICT_NUMERIC if c not in SP_COLS]
SCOPE_NO_SP = [c for c in SCOPE_ONLY if c not in SP_COLS]
MODELS = {
    "Gradient boosting": ("gradient_boosting", STRICT_NO_SP),
    "Random forest": ("random_forest", STRICT_NO_SP),
    "Logistic regression": ("logreg", STRICT_NO_SP),
    "Scope-only LR (baseline)": ("logreg", SCOPE_NO_SP),
}
TASKS = {"Late closure": ("full", "label_delay"), "Spillover": ("nonempty", "label_spillover")}
N_BOOT, MIN_CLASS = 2000, 5
_ORIG = p2d.load_reconstructed_sprints
_TAWOS_RAW = phase3.RAW_DIR


def use_raw(raw):
    """Point the frozen Phase 1-3 loaders at another raw folder (no code is changed)."""
    p2d.load_reconstructed_sprints = functools.partial(_ORIG, raw)
    phase3.RAW_DIR = raw


def boot(y, s, pred, rng):
    out = {"auc": [], "sensitivity": [], "specificity": [], "f1": []}
    n = len(y)
    for _ in range(N_BOOT):
        i = rng.integers(0, n, n)
        yy, ss, pp = y[i], s[i], pred[i]
        if 0 < yy.sum() < n:
            out["auc"].append(roc_auc_score(yy, ss))
            out["sensitivity"].append(pp[yy == 1].mean())
            out["specificity"].append(1 - pp[yy == 0].mean())
            out["f1"].append(f1_score(yy, pp, zero_division=0))
    return {k: tuple(np.percentile(v, [2.5, 97.5])) if v else (np.nan, np.nan) for k, v in out.items()}


def main():
    rng = np.random.default_rng(SEED)
    use_raw(_TAWOS_RAW)
    tawos = p2d.load_phase2_frame()
    ext = []
    for inst in INSTANCES:
        raw = EXT / inst
        if not (raw / "tawos_sprint.csv").exists():
            print("missing", raw)
            continue
        use_raw(raw)
        b = p2d.load_phase2_frame()
        b["instance"] = inst
        ext.append(b)
    use_raw(_TAWOS_RAW)
    ext = pd.concat(ext, ignore_index=True)

    rows = []
    for task, (pop, target) in TASKS.items():
        tr = p2d.population(tawos, pop)
        te_all = p2d.population(ext, pop)
        y_tr = tr[target].to_numpy(int)
        for model, (name, numeric) in MODELS.items():
            cols = feature_columns(numeric, False)
            est = make_classifier(name, numeric, False)
            oof = cross_val_predict(est, tr[cols], y_tr, cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                                    method="predict_proba")[:, 1]
            thr = youden_threshold(y_tr, oof)
            est.fit(tr[cols], y_tr)
            for proj, te in te_all.groupby("project"):
                y = te[target].to_numpy(int)
                s = est.predict_proba(te[cols])[:, 1]
                pred = (s >= thr).astype(int)
                m = decision_metrics(y, s, pred)
                ok = m["n_pos"] >= MIN_CLASS and m["n_neg"] >= MIN_CLASS
                ci = boot(y, s, pred, rng) if ok else {}
                r = {"task": task, "model": model, "instance": te["instance"].iloc[0], "project": proj,
                     **m, "threshold": thr, "interpretable": ok}
                for k, (lo, hi) in ci.items():
                    r[f"{k}_lo"], r[f"{k}_hi"] = lo, hi
                rows.append(r)
            print(task, model, "done", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(phase4.RESULTS / "e3_external_per_project.csv", index=False)
    keys = ["auc", "sensitivity", "specificity", "f1"]
    ok = res[res["interpretable"]]
    avg = ok.groupby(["task", "model"])[keys].agg(["mean", "std", "count"])
    avg.columns = [f"{a}_{b}" for a, b in avg.columns]
    avg = avg.reset_index()
    avg.round(3).to_csv(phase4.RESULTS / "e3_external_average.csv", index=False)
    desc = ext.groupby(["instance", "project"]).agg(
        sprints=("sprint_id", "size"), nonempty=("nonempty", "sum"),
        late_rate=("label_delay", "mean"),
        start=("start_date", "min"), end=("start_date", "max")).reset_index()
    sp = p2d.population(ext, "nonempty").groupby("project")["label_spillover"].mean().rename("spill_rate_nonempty")
    desc = desc.merge(sp, left_on="project", right_index=True, how="left")
    desc.to_csv(phase4.RESULTS / "e3_external_projects.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(desc.round(3).to_string(index=False))
    gb = res[res.model == "Gradient boosting"][["task", "project", "n", "n_pos", "n_neg", "auc", "sensitivity",
                                                 "specificity", "f1", "interpretable"]]
    print(gb.round(3).to_string(index=False))
    print(avg.round(3).to_string(index=False))


def figure():
    """Per-project AUC with 95% bootstrap interval, gradient boosting (run after main())."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    res = pd.read_csv(phase4.RESULTS / "e3_external_per_project.csv")
    res = res[res.model == "Gradient boosting"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    order = sorted(res.project.unique())
    for ax, task in zip(axes, TASKS):
        r = res[res.task == task].set_index("project").reindex(order)
        ypos = np.arange(len(order))
        for yi, (p, row) in zip(ypos, r.iterrows()):
            if row["interpretable"]:
                ax.errorbar(row["auc"], yi, xerr=[[row["auc"] - row["auc_lo"]], [row["auc_hi"] - row["auc"]]],
                            fmt="o", c="#1d4ed8", capsize=3)
                ax.text(1.02, yi, f"n={int(row['n'])} ({int(row['n_pos'])}+/{int(row['n_neg'])}-)",
                        va="center", fontsize=8, transform=ax.get_yaxis_transform())
            else:
                ax.text(0.5, yi, f"not interpretable: {int(row['n_pos'])}+/{int(row['n_neg'])}-",
                        va="center", ha="center", fontsize=8, color="#6b7280")
        ok = r[r["interpretable"] == True]  # noqa: E712
        ax.axvline(0.5, ls=":", c="#9ca3af")
        ax.axvline(ok["auc"].mean(), ls="--", c="#1d4ed8", alpha=0.5)
        ax.set_yticks(ypos)
        ax.set_yticklabels(order)
        ax.set_xlim(0.2, 1.0)
        ax.set_xlabel("AUC (95% bootstrap interval)")
        ax.set_title(f"{task}: mean AUC {ok['auc'].mean():.3f} ± {ok['auc'].std():.3f} over {len(ok)} projects",
                     fontsize=10)
        ax.grid(axis="x", alpha=0.25)
    fig.suptitle("Trained on TAWOS only, tested on 10 external projects (gradient boosting)", fontsize=12)
    fig.tight_layout()
    fig.savefig(phase4.FIGURES / "e3_external_auc.png", dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()
    figure()
