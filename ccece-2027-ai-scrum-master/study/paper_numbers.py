"""Step 5 of the study: print the paper's table rows from the result files, and check every number
the paper quotes from the study against them.

Usage (from the ccece-2027-ai-scrum-master folder):
  python -m study.paper_numbers table     # LaTeX rows for the results table
  python -m study.paper_numbers check     # checks CCECE_2027_Grounding_AI_Scrum_Master.tex
"""
from __future__ import annotations

import json
import re
import sys

import pandas as pd

import study

R = study.HERE / "results"


def f2(x):
    """Two decimals with a leading dot, rounded half up (as in the paper)."""
    s = f"{float(x) + 1e-9:.2f}"
    return s.replace("0.", ".", 1) if s.startswith("0.") else s


def table():
    am = pd.read_csv(R / "arm_metrics_by_run.csv")
    m = am.groupby(["arm", "checkpoint"]).mean(numeric_only=True)
    names = {"M": "$M$ model", "rule": "rule", "L": "$L$ agent", "G": "$G$ grounded"}
    rows = []
    for c in (0.0, 0.5):
        for arm in (("M", "rule", "L", "G") if c == 0.0 else ("M", "L", "G")):
            r = m.loc[(arm, c)]
            brier = "--" if arm == "rule" else f2(r.brier)
            kap = "1.00" if arm == "M" else f2(r.kappa_M)
            rows.append(f"{c:g} & {names[arm]} & {f2(r.warn_rate)} & {f2(r.sensitivity)} & {f2(r.specificity)} & "
                        f"{f2(r.auc)} & {f2(r.auc_within_project)} & {kap} & {brier} & {f2(r.unf_spearman)}\\\\")
    return rows


def expected():
    """Every study number the paper text quotes, keyed by the phrase that carries it."""
    s = json.loads((R / "summary.json").read_text())
    b = json.loads((R / "build_info.json").read_text())
    fa = json.loads((R / "factcheck_auto.json").read_text())["study"]
    ra = json.loads((R / "ratings_summary.json").read_text())
    p, c = s["pooled"], s["contrasts"]
    cv = {r["kappa_over_F"]: r for r in s["cost_curve"]}
    d = pd.read_csv(R / "replay_all_arms.csv")
    g0 = d[d.checkpoint == 0.0]
    late_pp = g0.groupby("project").late.sum()
    rule_agree = [int((g0[f"L{r}_warn"] == g0.rule_warn).sum()) for r in (1, 2, 3)]
    am = pd.read_csv(R / "arm_metrics_by_run.csv").groupby(["arm", "checkpoint"]).mean(numeric_only=True)
    const_brier = float(((g0.late.mean() - g0.late) ** 2).mean())
    sc = d[(d.checkpoint == 0.0) & (d.unf_scored == 1)]
    const_mae = float((sc.unfinished - sc.unfinished.median()).abs().mean())
    gdep = p["departures_G"]
    e = {
        "pool sprints": str(b["pool_n"]), "pool late": str(b["pool_late"]),
        "sample sprints": str(b["n_sprints"]), "sample late": str(b["n_late"]),
        "late per project min": str(int(late_pp.min())), "late per project max": str(int(late_pp.max())),
        "A_G percent": f"{round(100 * p['A_G']['mean'])}\\%", "kappa_G": f"{p['kappa_G']['mean']:.2f}",
        "G departures min": str(min(gdep["n"])), "G departures max": str(max(gdep["n"])),
        "G right": str(gdep["correct"][0]),
        "G silences min": str(min(gdep["to_silent"])), "G silences max": str(max(gdep["to_silent"])),
        "G extra warnings": str(gdep["n"][0] - gdep["to_silent"][0]),
        "auc G-M c0": f"{c['auc_G_minus_M_c0']['point']:.2f}",
        "auc G-M c0 lo": f"{c['auc_G_minus_M_c0']['lo']:.2f}", "auc G-M c0 hi": f"{c['auc_G_minus_M_c0']['hi']:.2f}",
        "auc G-M c50 lo": f"{c['auc_G_minus_M_c50']['lo']:.2f}", "auc G-M c50 hi": f"{c['auc_G_minus_M_c50']['hi']:.2f}",
        "cost gap 0.02": f"{-cv[0.02]['M_minus_G']:.3f}", "cost gap 0.02 lo": f"{-cv[0.02]['M_minus_G_hi']:.3f}",
        "cost gap 0.02 hi": f"{-cv[0.02]['M_minus_G_lo']:.3f}",
        "cost gap 0.5": f"{-cv[0.5]['M_minus_G']:.3f}", "cost gap 0.5 lo": f"{-cv[0.5]['M_minus_G_hi']:.3f}",
        "cost gap 0.5 hi": f"{-cv[0.5]['M_minus_G_lo']:.3f}",
        "rule agree runs": ", ".join(map(str, rule_agree[:2])) + " and " + str(rule_agree[2]),
        "L within c0": f"{am.loc[('L', 0.0)].auc_within_project:.2f}",
        "rule within": f"{am.loc[('rule', 0.0)].auc_within_project:.2f}",
        "M-rule": f"{c['auc_M_minus_rule_c0']['point']:.2f}", "M-rule lo": f"{c['auc_M_minus_rule_c0']['lo']:.2f}",
        "M-rule hi": f"{c['auc_M_minus_rule_c0']['hi']:.2f}",
        "L brier c0": f"{am.loc[('L', 0.0)].brier:.3f}", "const brier": f"{const_brier:.3f}",
        "same call L": f"{round(100 * p['same_call_all_runs_L'])}\\%", "same call G": f"{round(100 * p['same_call_all_runs_G'])}\\%",
        "fleiss L": f"{p['fleiss_runs_L']:.2f}", "fleiss G": f"{p['fleiss_runs_G']:.2f}",
        "p sd": f"{(p['p_late_sd_L'] + p['p_late_sd_G']) / 2:.2f}",
        "M changed": str(p["sprints_with_changed_call"]["M"]),
        "agents changed min": str(min(v for k, v in p["sprints_with_changed_call"].items() if k != "M")),
        "agents changed max": str(max(v for k, v in p["sprints_with_changed_call"].items() if k != "M")),
        "const mae": f"{const_mae:.2f}",
        "unf M c50": f2(am.loc[("M", 0.5)].unf_spearman).replace(".", "0.", 1),
        "unf G c50": f2(am.loc[("G", 0.5)].unf_spearman).replace(".", "0.", 1),
        "unf L c50": f2(am.loc[("L", 0.5)].unf_spearman).replace(".", "0.", 1),
        "unf G-M": f"{c['unf_rho_G_minus_M_c50']['point']:.2f}", "unf G-M lo": f"{c['unf_rho_G_minus_M_c50']['lo']:.2f}",
        "unf G-M hi": f"{c['unf_rho_G_minus_M_c50']['hi']:.2f}",
        "unf L-M": f"{c['unf_rho_L_minus_M_c50']['point']:.2f}", "unf L-M lo": f"{c['unf_rho_L_minus_M_c50']['lo']:.2f}",
        "unf L-M hi": f"{c['unf_rho_L_minus_M_c50']['hi']:.2f}",
        "unf hist c50": f2(am.loc[("rule", 0.5)].unf_spearman).replace(".", "0.", 1),
        "warnings": str(s["n_warnings_total"]),
        "L warnings": str(fa["L_warnings"]), "G warnings": str(fa["G_warnings"]),
        "numbers": f"{fa['numbers']:,}", "numbers traced": f"{fa['numbers'] - fa['unsupported_numbers']:,}",
        "unsupported L": str(ra["study"]["L"]["unsupported_both"]),
        "unsupported G": str(ra["study"]["G"]["unsupported_both"]),
        "unsupported L pct": f"{round(100 * ra['study']['L']['unsupported_both'] / ra['study']['L']['n'])}\\%",
        "unsupported G pct": f"{round(100 * ra['study']['G']['unsupported_both'] / ra['study']['G']['n'])}\\%",
        "kappa claims": f"{ra['study']['kappa_claims']:.2f}", "kappa close": f"{ra['study']['kappa_close']:.2f}",
        "kappa vs original": f"{ra['pilot']['kappa_both_vs_original']:.2f}",
        "close L": str(ra["study"]["L"]["close_both"]), "close G": str(ra["study"]["G"]["close_both"]),
        "close L pct": f"{round(100 * ra['study']['L']['close_both'] / ra['study']['L']['n'])}\\%",
        "close G pct": f"{round(100 * ra['study']['G']['close_both'] / ra['study']['G']['n'])}\\%",
    }
    return e


def check():
    tex = (study.HERE.parent / "CCECE_2027_Grounding_AI_Scrum_Master.tex").read_text()
    bad = 0
    for row in table():
        if row not in tex:
            print("TABLE ROW MISSING:", row)
            bad += 1
    for k, v in expected().items():
        if v not in tex:
            print(f"NOT FOUND  {k}: {v}")
            bad += 1
    print(f"{len(expected())} numbers and {len(table())} table rows checked; {bad} problems")
    return bad


if __name__ == "__main__":
    if sys.argv[1] == "table":
        print("\n".join(table()))
    elif sys.argv[1] == "expected":
        print(json.dumps(expected(), indent=1))
    else:
        sys.exit(1 if check() else 0)
