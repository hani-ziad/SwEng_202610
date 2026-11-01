"""External validation on MariaDB Jira (MaxScale MXS, ColumnStore MCOL).

Runs the FROZEN Phase 1-3 code on a second data source. No model, label,
feature, split or reconstruction code is edited: this script only points the
existing loaders at a different raw-data folder (data/raw/mariadb, produced by
convert_mariadb.py in the TAWOS CSV layout).

Analysis plan, fixed before any MariaDB outcome was looked at:
  0. Sanity: rerun the paper's headline rows on TAWOS through this harness.
  1. Within MariaDB, the paper's protocol: per-project time-ordered 70/30 split,
     project one-hot, same models and settings. Rows mirror Table II.
  2. Transfer: train on all 8 TAWOS projects, test on all MariaDB sprints, no
     project feature. MariaDB has no story points, so this arm drops the three
     story-point columns from every feature set (on both sides); the same
     point-free sets are also scored on the TAWOS time split for reference.
Uncertainty: two projects cannot support a project-cluster bootstrap, so 95%
intervals resample test sprints (2,000 paired resamples, seed 42).

Usage (from this folder):  python run_external.py
"""
from __future__ import annotations

import functools
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "phase-3-in-sprint-early-warning"))

import phase3  # noqa: E402  (adds Phase 2 and Phase 1 to the path)
import phase2.data as p2d  # noqa: E402
import phase3.data as p3d  # noqa: E402
from phase2.models import (  # noqa: E402
    SCOPE_ONLY, REG_BASELINES, make_classifier, make_regressor, feature_columns,
)
from src.data import load_tawos as lt  # noqa: E402
from sklearn.metrics import roc_auc_score, mean_absolute_error  # noqa: E402

TAWOS_RAW = phase3.RAW_DIR
MDB_RAW = HERE / "data" / "raw" / "mariadb"
RESULTS = HERE / "results"
N_BOOT, SEED = 2000, 42
SP_COLS = {"committed_story_points", "hist_avg_delivered_sp_3", "prior_avg_delivered_sp_3"}
_ORIG = p2d.load_reconstructed_sprints


def use_raw(raw: Path) -> None:
    """Point the frozen loaders at another raw folder (no code is changed)."""
    p2d.load_reconstructed_sprints = functools.partial(_ORIG, raw)
    phase3.RAW_DIR = raw


def frames(raw: Path):
    use_raw(raw)
    base = p2d.load_phase2_frame()
    ck = p3d.load_checkpoint_frame()
    _, audit = lt.load_reconstructed_sprints(raw, return_audit=True)
    return base, ck, audit


# ------------------------------------------------------------------ helpers
def auc(y, s):
    y = np.asarray(y)
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


def boot_ci(y, s, s_ref=None):
    rng = np.random.default_rng(SEED)
    y, s = np.asarray(y), np.asarray(s)
    vals = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            a = roc_auc_score(y[i], s[i])
            if s_ref is not None:
                a -= roc_auc_score(y[i], np.asarray(s_ref)[i])
            vals.append(a)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def fit_clf(name, numeric, train, test, target, with_project):
    cols = feature_columns(numeric, with_project)
    m = make_classifier(name, numeric, with_project).fit(train[cols], train[target])
    return m.predict_proba(test[cols])[:, 1]


def scope_lr(train, test, target, cols):
    return fit_clf("logreg", cols, train, test, target, False)


def project_rate(train, test, target):
    rate = train.groupby("project")[target].mean()
    return test["project"].map(rate).fillna(train[target].mean()).to_numpy(float)


ROWS = []


def row(arm, task, split, test, target, model, s, comp_name=None, s_ref=None):
    y = test[target].to_numpy(int)
    r = {"arm": arm, "task": task, "split": split, "n": len(y), "pos": int(y.sum()),
         "model": model, "auc": auc(y, s)}
    r["auc_lo"], r["auc_hi"] = boot_ci(y, s)
    if s_ref is not None:
        r["comparator"] = comp_name
        r["comp_auc"] = auc(y, s_ref)
        r["diff"] = r["auc"] - r["comp_auc"]
        r["diff_lo"], r["diff_hi"] = boot_ci(y, s, s_ref)
    ROWS.append(r)
    return r


def drop_sp(cols):
    return [c for c in cols if c not in SP_COLS]


# ------------------------------------------------------------------ arms
def within(arm, base, ck):
    strict, legacy = p2d.FEATURE_SETS["strict"], p2d.FEATURE_SETS["legacy"]
    # late closure, all sprints (Table II rows 1-2)
    tr, te = p2d.time_ordered_split(p2d.population(base, "full"))
    gb = fit_clf("gradient_boosting", strict, tr, te, "label_delay", True)
    row(arm, "Late closure", "Time", te, "label_delay", "GB, strict", gb,
        "Project rate", project_rate(tr, te, "label_delay"))
    row(arm, "Late closure", "Time", te, "label_delay", "GB, strict", gb,
        "Scope-only LR", scope_lr(tr, te, "label_delay", SCOPE_ONLY))
    rf = fit_clf("random_forest", strict, tr, te, "label_delay", True)
    row(arm, "Late closure", "Time", te, "label_delay", "RF, strict", rf,
        "Scope-only LR", scope_lr(tr, te, "label_delay", SCOPE_ONLY))
    # spillover, all sprints vs nonempty (row 8)
    rf_all = fit_clf("random_forest", legacy, tr, te, "label_spillover", True)
    row(arm, "Spillover, all sprints", "Time", te, "label_spillover", "RF, legacy", rf_all)
    # spillover, nonempty (row 5)
    tr, te = p2d.time_ordered_split(p2d.population(base, "nonempty"))
    lr = scope_lr(tr, te, "label_spillover", SCOPE_ONLY)
    for name, label in [("gradient_boosting", "GB, strict"), ("random_forest", "RF, strict"),
                        ("logreg", "LR, strict")]:
        s = fit_clf(name, strict, tr, te, "label_spillover", True)
        row(arm, "Spillover, nonempty", "Time", te, "label_spillover", "LR, scope only", lr, label, s)
    rf_ne = fit_clf("random_forest", legacy, tr, te, "label_spillover", True)
    row(arm, "Spillover, nonempty", "Time", te, "label_spillover", "RF, legacy", rf_ne)
    # regression rows 13-15 (MAE)
    for task, pop, target, bl in [("Issues completed", "nonempty", "completed_issue_count", "scope_x_rate"),
                                  ("Carry share", "nonempty", "carry_ratio", "train_median"),
                                  ("Days late", "full", "late_days_capped", "train_median")]:
        tr, te = p2d.time_ordered_split(p2d.population(base, pop))
        cols = feature_columns(strict, True)
        m = make_regressor("gb_median", strict, True).fit(tr[cols], tr[target])
        p = m.predict(te[cols])
        b = np.asarray(REG_BASELINES[target][bl](tr, te, target), float)
        ROWS.append({"arm": arm, "task": task, "split": "Time", "n": len(te), "model": "GB (abs. error)",
                     "mae": float(mean_absolute_error(te[target], p)), "comparator": bl,
                     "comp_mae": float(mean_absolute_error(te[target], b))})
    # in-sprint checkpoints (rows 10-12)
    prog, plan = p3d.CHECKPOINT_NUMERIC, p2d.FEATURE_SETS["strict"]
    for task, pop, target, f in [("Spillover, 25%", "nonempty", "label_spillover", 0.25),
                                 ("Spillover, 50%", "nonempty", "label_spillover", 0.5),
                                 ("Late closure, 50%", "full", "label_delay", 0.5)]:
        d = ck[(ck["checkpoint"] == f) & ck["open_at_checkpoint"] & ck[f"split_{pop}"].notna()]
        if pop == "nonempty":
            d = d[d["nonempty"]]
        tr, te = d[d[f"split_{pop}"] == "train"], d[d[f"split_{pop}"] == "test"]
        a = fit_clf("gradient_boosting", prog, tr, te, target, True)
        b = fit_clf("gradient_boosting", plan, tr, te, target, True)
        row(arm, task, "Time", te, target, "GB, strict + progress", a, "No progress", b)


def transfer(tawos, mdb):
    strict = drop_sp(p2d.FEATURE_SETS["strict"])
    scope = drop_sp(SCOPE_ONLY)
    for task, pop, target in [("Late closure", "full", "label_delay"),
                              ("Spillover, nonempty", "nonempty", "label_spillover")]:
        tr, te = p2d.population(tawos, pop), p2d.population(mdb, pop)
        base_s = scope_lr(tr, te, target, scope)
        for name, label in [("gradient_boosting", "GB, strict, no SP"), ("random_forest", "RF, strict, no SP"),
                            ("logreg", "LR, strict, no SP")]:
            s = fit_clf(name, strict, tr, te, target, False)
            row("Transfer TAWOS->MariaDB", task, "Cross-source", te, target, label, s, "Scope-only LR, no SP", base_s)
        # reference: the same point-free sets on the TAWOS time split
        trt, tet = p2d.time_ordered_split(tr)
        ref_s = scope_lr(trt, tet, target, scope)
        s = fit_clf("gradient_boosting", strict, trt, tet, target, True)
        row("TAWOS, no SP (reference)", task, "Time", tet, target, "GB, strict, no SP", s, "Scope-only LR, no SP", ref_s)
        row("TAWOS, no SP (reference)", task, "Time", tet, target, "Scope-only LR, no SP", ref_s)


def describe(name, base, audit):
    out = {"source": name, "modeled_sprints": int(len(base)), "nonempty": int(base["nonempty"].sum()),
           "projects": {p: int(n) for p, n in base["project"].value_counts().items()},
           "late_rate": float(base["label_delay"].mean()),
           "spillover_rate_nonempty": float(base.loc[base["nonempty"], "label_spillover"].mean()),
           "median_committed_issues_nonempty": float(base.loc[base["nonempty"], "committed_issue_count"].median()),
           "first_start": str(base["start_date"].min().date()), "last_start": str(base["start_date"].max().date())}
    out.update({f"audit_{k}": v for k, v in audit.items()})
    return out


def main():
    RESULTS.mkdir(exist_ok=True)
    t_base, t_ck, t_audit = frames(TAWOS_RAW)
    within("TAWOS (sanity)", t_base, t_ck)
    m_base, m_ck, m_audit = frames(MDB_RAW)
    within("MariaDB", m_base, m_ck)
    transfer(t_base, m_base)
    res = pd.DataFrame(ROWS)
    res.to_csv(RESULTS / "external_validation_results.csv", index=False)
    desc = [describe("TAWOS", t_base, t_audit), describe("MariaDB", m_base, m_audit)]
    (RESULTS / "cohort_description.json").write_text(json.dumps(desc, indent=2))
    import platform, sklearn, scipy
    env = {"python": platform.python_version(), "platform": platform.platform(), "numpy": np.__version__,
           "pandas": pd.__version__, "scikit_learn": sklearn.__version__, "scipy": scipy.__version__}
    (RESULTS / "environment_versions.json").write_text(json.dumps(env, indent=2))
    print("environment:", env)
    m_base.to_csv(RESULTS / "mariadb_modeling_frame.csv", index=False)
    pd.set_option("display.width", 250)
    print(json.dumps(desc, indent=1))
    show = res.copy()
    for c in [c for c in show.columns if c not in ("arm", "task", "split", "model", "comparator", "n", "pos")]:
        show[c] = show[c].map(lambda v: f"{v:.3f}" if isinstance(v, float) and np.isfinite(v) else "")
    print(show.to_string(index=False))


if __name__ == "__main__":
    main()
