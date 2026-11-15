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
        # first (unmasked) rating, quoted as counts only since 15 Nov 2026 (night)
        "first rating both": f'gave {ra["study"]["L"]["unsupported_both"]} and {ra["study"]["G"]["unsupported_both"]} when both',
        "first rating either": f'{ra["study"]["L"]["unsupported_either"]} and {ra["study"]["G"]["unsupported_either"]} for either',
        "kappa vs original": f"{ra['pilot']['kappa_both_vs_original']:.2f}",
        "first close": f'against {ra["study"]["L"]["close_both"]} and {ra["study"]["G"]["close_both"]} in the first',
    }
    # added 15 Nov 2026 (evening): model-output ablation and cost scenarios
    ab = json.loads((R / "ablation_summary.json").read_text())
    iv = ab["intervals"]
    sc = json.loads((R / "cost_scenarios.json").read_text())
    rows = {(r["scenario"], r["kappa_over_F"]): r for r in sc["rows"]}
    band = sc["beats_trivial_band"]["default"]
    e.update({
        "P agree": f"{round(100 * ab['pooled']['P_agree_true_M'])}\\%",
        "P-G": f"{iv['P_minus_G_agree']['point']:.2f}", "P-G lo": f"{iv['P_minus_G_agree']['lo']:.2f}",
        "P-G hi": f"{iv['P_minus_G_agree']['hi']:.2f}",
        "F follow n": f"{ab['pooled']['F_follow_n']} of 240",
        "F-L": f"{iv['F_follow_minus_L']['point']:.2f} more often", "F-L lo": f"({iv['F_follow_minus_L']['lo']:.2f} to",
        "F-L hi": f"{iv['F_follow_minus_L']['hi']:.2f})",
        "F auc c0": f"(AUC {ab['c0']['F']['auc']:.2f})",
        "F auc gap": f"{-iv['auc_F_minus_L_c0']['point']:.2f} below the agent alone ({-iv['auc_F_minus_L_c0']['hi']:.2f} to {-iv['auc_F_minus_L_c0']['lo']:.2f})",
        "strong gap": f"({rows[('strong', 0.02)]['G_minus_M']:.3f} to {rows[('strong', 0.5)]['G_minus_M']:.3f})",
        "M band": f"between {band['M']['from']:.2f} and {band['M']['to']:.2f}",
        "G band": f"between {band['G']['from']:.2f} and {band['G']['to']:.2f}",
    })
    assert band["M"]["contiguous"] and band["G"]["contiguous"]
    assert rows[("zero", 0.5)]["G_minus_M_lo"] <= 0 <= rows[("zero", 0.5)]["G_minus_M_hi"]

    # added 15 Nov 2026 (night): masked re-rating, sensitivity runs, cost-optimal M, pilot overlap, eligibility
    mk = json.loads((R / "ratings_masked_summary.json").read_text())
    se = json.loads((R / "sensitivity_summary.json").read_text())
    rv = json.loads((R / "review2.json").read_text())
    ts = json.loads((R / "train_side.json").read_text())
    pc = lambda a, b: f"{round(100 * a / b)}\\%"
    L, G = mk["L"], mk["G"]
    sv = se["intervals"]
    e.update({
        "masked both L": f"{L['uns_both']} of $L$'s warnings ({pc(L['uns_both'], L['n'])})",
        "masked both G": f"{G['uns_both']} of $G$'s ({pc(G['uns_both'], G['n'])})",
        "masked either": f"{L['uns_either']} and {G['uns_either']} ({pc(L['uns_either'], L['n'])} and {pc(G['uns_either'], G['n'])})",
        "masked kappa": f"(kappa {mk['kappa_claims']:.2f}, {mk['disagree_claims']} disagreements)",
        "masked ci L": f"{round(100 * L['uns_both_ci'][0])}\\% to {round(100 * L['uns_both_ci'][1])}\\% for $L$",
        "masked ci G": f"{round(100 * G['uns_both_ci'][0])}\\% to {round(100 * G['uns_both_ci'][1])}\\% for $G$",
        "masked close kappa": f"masked kappa {mk['kappa_close']:.2f}, {mk['disagree_close']} disagreements",
        "masked close L": f"{L['close_both']} of $L$'s warnings ({pc(L['close_both'], L['n'])})",
        "masked close G": f"{G['close_both']} of $G$'s ({pc(G['close_both'], G['n'])})",
        "abstract uns": f"{pc(G['uns_both'], G['n'])} to {pc(L['uns_both'], L['n'])} of warnings",
        "abstract close": f"{pc(G['close_both'], G['n'])} to {pc(L['close_both'], L['n'])} advised",
        "F2 follow": f"followed on {se['F2']['follow_n']} points ({sv['F2_follow_minus_L']['point']:.2f} above the agent alone, "
                     f"{sv['F2_follow_minus_L']['lo']:.2f} to {sv['F2_follow_minus_L']['hi']:.2f})",
        "F F2 same": f"same call on {round(100 * se['F_F2_same_call'])}\\% of points",
        "flip split": f"followed it on {rv['flip']['followed_when_model_right']} of the {rv['flip']['points_model_right']} points "
                      f"where it was wrong and {rv['flip']['followed_when_model_wrong']} of the {rv['flip']['points_model_wrong']}",
        "C follow": f"followed the stated call on {se['C']['follow_n']} of 240 points and kept the real call on "
                    f"{round(100 * se['C']['agree_true_M'])}\\%",
        "T agree": f"({round(100 * se['T']['agree_true_M'])}\\%, difference {sv['T_minus_G_agree']['point']:.2f}, "
                   f"{sv['T_minus_G_agree']['lo']:.2f} to {sv['T_minus_G_agree']['hi']:.2f})",
        "no pilot": f"On the {rv['no_pilot']['sprints']} sprints that were not in the pilot ({rv['no_pilot']['late']} late), "
                    f"agreement was again {round(100 * rv['no_pilot']['A_G'])}\\%, all "
                    f"{rv['no_pilot']['G_extra_warnings_on_ontime'][0]} extra warnings",
        "no pilot auc": f"gap at planning was {rv['no_pilot']['auc_G_minus_M_c0']:.2f}",
        "overlap": {13: "Thirteen"}[rv["overlap_with_pilot"]] + " of its sprints",
        "elig": f"its planning AUC is {rv['eligibility']['auc_M_c0_pool']:.2f} on the pool and "
                f"{rv['eligibility']['auc_M_c0_nonempty']:.2f} on all {rv['eligibility']['nonempty']} held-out",
        "projects auc-p": {6: "six"}[rv["within_project_auc_projects"]["c0"]["contributing"]] + " projects with at least two late",
    })
    assert set(rv["no_pilot"]["G_extra_warnings_on_ontime"]) == {16} and rv["no_pilot"]["G_extra_warnings"] == [16, 16, 16]
    assert rv["within_project_auc_projects"]["c50"]["contributing"] == 6
    # G never cheaper than the cost-optimal M; tied only at strong, 0.2
    co = {(r["scenario"], r["kappa_over_F"]): r["G_minus_Mopt"] for r in ts["cost_optimal_M"]}
    assert all(v > -0.0005 for v in co.values()) and [k for k, v in co.items() if v < 0.0005] == [("strong", 0.2)]
    e["tied once"] = "tied once, at $\\kappa/F=0.2$ with stronger warnings"
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
