"""Check the corrected-label numbers in the CCECE paper against the Phase 4 result files.

The 8 Nov 2026 label audit (Phase 4 experiments 4 and 5) first went into the paper as a
subsection with Table II rows 21-24. When it was merged into the 6-page audit version the
same evening, it shrank to two passages (next-working-day late closure in Section IV-A,
heavy carry-over in Section IV-F) and gained within-project scores (experiment 6). The
checks below cover the experiment 4 and 5 result files, which the merged paper still
draws on, plus the experiment 6 numbers it now quotes. Run after verify_numbers.py:

    python ccece-2027-paper/verify_labels.py
"""
import os
from pathlib import Path

import pandas as pd

R = os.environ.get("SPRINT_RISK_ROOT", str(Path(__file__).resolve().parents[1]))
F4 = f"{R}/phase-4-fixes/results"
fails = []


def check(name, claimed, actual, tol=0.0006):
    ok = abs(float(claimed) - float(actual)) <= tol
    print(("PASS " if ok else "FAIL ") + f"{name}: paper={claimed} artifact={float(actual):.4f}")
    if not ok:
        fails.append(name)


san = pd.read_csv(f"{F4}/e4_sanity_checks.csv").set_index("check").value
check("spillover 93% positive", 93.1, 100 * san["spillover_rate_nonempty"], 0.05)
check("one-issue sprints 75% spill", 75, 100 * san["spillover_rate_1_to_1_issues"], 0.5)
check("over ten issues 99.7% spill", 99.7, 100 * san["spillover_rate_11_to_10000_issues"], 0.05)
check("weekend share of late 32.8%", 32.8, 100 * san["late_share_end_fri_to_sun_closed_within_3_5_days"], 0.05)
check("Cloud finished without date 68%", 68, 100 * san["closed_without_resolution_Atlassian Confluence Cloud"], 0.5)
others = [v for k, v in san.items() if k.startswith("closed_without_resolution_")
          and "Cloud" not in k and k not in ("closed_without_resolution_mariadb", "closed_without_resolution_mongodb")]
check("other TAWOS at most 0.5%", 0.5, 100 * max(others), 0.05)
check("other TAWOS projects counted 7", 7, len(others), 0)

lab = pd.read_csv(f"{F4}/e5_label_definitions.csv").set_index("project")
allp = lab.loc["All 8 projects"]
check("late rate 38.7% (1-day rule)", 38.7, 100 * allp.late_1d, 0.05)
check("late rate 25.4% (next working day)", 25.4, 100 * allp.late_next_working_day, 0.05)
check("Titanium SDK 60%", 60, 100 * lab.loc["The Titanium SDK", "late_1d"], 0.5)
check("Titanium SDK 19%", 19, 100 * lab.loc["The Titanium SDK", "late_next_working_day"], 0.5)
check("heavy carry sprints 1,494", 1494, allp.nonempty, 0)
check("heavy carry 78.0% positive", 78.0, 100 * allp.heavy_carry, 0.05)

fx = pd.read_csv(f"{F4}/e5_fixed_split.csv")


def fxv(task, model, col):
    r = fx[(fx.task == task) & (fx.model == model)]
    assert len(r) == 1, (task, model)
    return r[col].item()


GB = "Gradient boosting"
check("row21 n 786", 786, fxv("late", GB, "n"), 0); check("row21 pos 167", 167, fxv("late", GB, "n_pos"), 0)
check("row21 GB .727", .727, fxv("late", GB, "auc")); check("row21 past lateness .619", .619, fxv("late", "Past lateness (rule)", "auc"))
check("row22 n 451", 451, fxv("carry", GB, "n"), 0); check("row22 pos 387", 387, fxv("carry", GB, "n_pos"), 0)
check("row22 GB .565", .565, fxv("carry", GB, "auc")); check("row22 past carry .621", .621, fxv("carry", "Past carry share (rule)", "auc"))

dl = pd.read_csv(f"{F4}/e5_model_vs_best_rule.csv")


def dv(task, cp):
    r = dl[(dl.task == task) & (dl.checkpoint == cp)]
    assert len(r) == 1, (task, cp)
    return r.iloc[0]


r = dv("late", 0.0); check("row21 gap +.108", .108, r.delta_auc); check("row21 lo .069", .069, r.lo); check("row21 hi .150", .150, r.hi)
assert r.rule == "Past lateness (rule)", r.rule
r = dv("carry", 0.0); check("row22 gap -.056", -.056, r.delta_auc); check("row22 lo -.151", -.151, r.lo); check("row22 hi .040", .040, r.hi)
assert r.rule == "Past carry share (rule)", r.rule
r = dv("carry", 0.5); check("row23 gap +.029", .029, r.delta_auc); check("row23 lo -.040", -.040, r.lo); check("row23 hi .094", .094, r.hi)
check("row23 n 446", 446, r.n, 0); check("row23 pos 383", 383, r.n_pos, 0)
assert r.rule == "Remaining share (rule)", r.rule

ins = pd.read_csv(f"{F4}/e5_insprint.csv")


def iv(task, cp, model):
    x = ins[(ins.task == task) & (ins.checkpoint == cp) & (ins.model == model)]
    assert len(x) == 1, (task, cp, model)
    return x.auc.item()


check("row23 GB .839", .839, iv("carry", 0.5, "Gradient boosting, progress features"))
check("row23 share still open .810", .810, iv("carry", 0.5, "Remaining share (rule)"))
gain = max(iv("late", c, "Gradient boosting, progress features") - iv("late", c, "Gradient boosting, planning features")
           for c in (0.25, 0.5, 0.75))
check("late progress adds at most .015", .015, gain, 0.0006)

s = pd.read_csv(f"{F4}/e5_summary.csv")


def sv(task, model, col):
    x = s[(s.task == task) & s.scheme.str.startswith("C.") & (s.model == model)]
    assert len(x) == 1
    return x[col].item()


check("late time x5 GB .706", .706, sv("late", GB, "auc_mean")); check("late time x5 SD .047", .047, sv("late", GB, "auc_std"))
models = ["Gradient boosting", "Random forest", "Logistic regression"]
lo_, hi_ = min(sv("carry", m, "auc_mean") for m in models), max(sv("carry", m, "auc_mean") for m in models)
check("carry models from .63", .63, lo_, 0.005); check("carry models to .66", .66, hi_, 0.005)
check("carry committed count .567", .567, sv("carry", "Committed count (rule)", "auc_mean"))
sens = pd.read_csv(f"{F4}/e5_label_sensitivity.csv")
check("old label same sprints count .890", .890,
      sens[(sens.label == "label_spillover") & (sens.model == "Committed count (rule)")]["mean"].item())

ext = pd.read_csv(f"{F4}/e5_external_per_project.csv")
for task, mean, sd, above, n in (("late", .546, .139, 2, 9), ("carry", .522, None, 0, 9)):
    g = ext[(ext.task == task) & (ext.model == GB) & ext.interpretable]
    check(f"outside {task} projects {n}", n, len(g), 0)
    check(f"outside {task} mean {mean}", mean, g.auc.mean())
    if sd is not None:
        check(f"outside {task} SD {sd}", sd, g.auc.std(), 0.001)
    check(f"outside {task} CI above .5 in {above}", above, g.ci_above_half.sum(), 0)

e3 = pd.read_csv(f"{F4}/e3_external_per_project.csv")
sg = e3[(e3.task == "Spillover") & (e3.model == GB) & e3.interpretable]
check("outside spillover negatives min 6", 6, sg.n_neg.min(), 0); check("outside spillover negatives max 9", 9, sg.n_neg.max(), 0)
check("outside resolution dates complete (MariaDB <1%)", 0.8, 100 * san["closed_without_resolution_mariadb"], 0.05)

# experiment 6: within-project scores of the corrected labels (quoted in the merged paper)
w = pd.read_csv(f"{F4}/e6_within_project.csv")
gp = pd.read_csv(f"{F4}/e6_within_project_gaps.csv").set_index("task")
wv = lambda task, m: w[(w.task == task) & (w.model == m)]["within_auc"].item()
check("late next-working-day GB within .631", .631, wv("late", GB))
check("late GB within CI low .531", .531, gp.loc["late", "gb_within_lo"]); check("late GB within CI high .689", .689, gp.loc["late", "gb_within_hi"])
check("late past late rate within .482", .482, wv("late", "Past lateness (rule)"))
check("late within: GB minus rule interval above 0", 1, int(gp.loc["late", "gap_lo"] > 0), 0)
check("carry GB within .521", .521, wv("carry", GB))
check("carry committed count within .621", .621, wv("carry", "Committed count (rule)"))
ins = pd.read_csv(f"{F4}/e5_model_vs_best_rule.csv")
r50 = ins[(ins.task == "carry") & (ins.checkpoint == 0.5)].iloc[0]
check("carry at 50%: gap interval includes zero", 1, int(r50.lo < 0 < r50.hi), 0)

print("\nFAILS:", fails if fails else "none")
