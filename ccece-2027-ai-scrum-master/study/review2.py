"""Reanalysis for the second external review (15 Nov 2026, night). No new agent calls.

  1. Flipped condition: how often the followed (flipped) call happened to match the outcome, split by
     whether the model's real call was right.
  2. The 13 sprints shared with the 4 Nov pilot: main results on the 107 sprints that were not in it.
  3. How many projects enter each within-project AUC (projects need at least two late and two on-time
     sprints).
  4. Eligibility: the replay pool keeps only sprints still open at mid-sprint, which is not known at
     planning. The model's planning AUC on that pool against all held-out sprints with committed work.

Usage (from the ccece-2027-ai-scrum-master folder):  python -m study.review2
Writes study/results/review2.json.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import study
import pilot.build as pb
from study.ablation import kappa

R = study.HERE / "results"
RUNS = (1, 2, 3)


def main():
    out = {}
    a = pd.read_csv(study.HERE / "ablation" / "ablation_points.csv")
    f = a.F_warn == a.F_shown_warn
    m_right = a.m_warn == a.late
    out["flip"] = {
        "followed": int(f.sum()),
        "followed_and_flipped_call_matched_outcome": int((f & (a.F_shown_warn == a.late)).sum()),
        "followed_and_flipped_call_missed_outcome": int((f & (a.F_shown_warn != a.late)).sum()),
        "points_model_right": int(m_right.sum()), "points_model_wrong": int((~m_right).sum()),
        "followed_when_model_right": int((f & m_right).sum()),
        "followed_when_model_wrong": int((f & ~m_right).sum()),
    }

    d = pd.read_csv(R / "replay_all_arms.csv")
    pilot_ids = set(pd.read_csv(study.HERE.parent / "pilot" / "results" / "replay_points.csv").sprint_id)
    out["overlap_with_pilot"] = int(len(pilot_ids & set(d.sprint_id)))
    sub = {"all": d, "no_pilot": d[~d.sprint_id.isin(pilot_ids)]}
    for name, x in sub.items():
        x0 = x[x.checkpoint == 0.0]
        e = {"sprints": int(x.sprint_id.nunique()), "late": int(x0.late.sum()),
             "A_G": float(np.mean([(x[f"G{r}_warn"] == x.m_warn).mean() for r in RUNS])),
             "kappa_G": float(np.mean([kappa(x[f"G{r}_warn"], x.m_warn) for r in RUNS])),
             "auc_M_c0": float(roc_auc_score(x0.late, x0.m_score)),
             "auc_G_c0": float(np.mean([roc_auc_score(x0.late, x0[f"G{r}_p"]) for r in RUNS])),
             "rule_agree_L_c0": [int((x0[f"L{r}_warn"] == x0.rule_warn).sum()) for r in RUNS],
             "G_extra_warnings_on_ontime": [int(((x[f"G{r}_warn"] == 1) & (x.m_warn == 0) & (x.late == 0)).sum())
                                            for r in RUNS],
             "G_extra_warnings": [int(((x[f"G{r}_warn"] == 1) & (x.m_warn == 0)).sum()) for r in RUNS]}
        e["auc_G_minus_M_c0"] = e["auc_G_c0"] - e["auc_M_c0"]
        out[name] = e
    ab = a[~a.sprint_id.isin(pilot_ids)]
    out["no_pilot"]["P_agree_true_M"] = float((ab.P_warn == ab.m_warn).mean())
    out["no_pilot"]["F_follow_share"] = float((ab.F_warn == ab.F_shown_warn).mean())

    proj = {}
    for c, g in d.groupby("checkpoint"):
        k = g.groupby("project").late.agg(["sum", "count"])
        ok = k[(k["sum"] >= 2) & (k["count"] - k["sum"] >= 2)]
        proj[f"c{int(c * 100)}"] = {"contributing": int(len(ok)), "excluded": sorted(set(k.index) - set(ok.index)),
                                    "late_per_project": {p: int(v) for p, v in k["sum"].items()}}
    out["within_project_auc_projects"] = proj

    base, te0, ck, m = pb.model_arm()
    mid = ck[ck.checkpoint == 0.5].set_index("sprint_id")
    m0 = m[m.checkpoint == 0.0].merge(base[["sprint_id", "label_delay", "committed_issue_count"]], on="sprint_id")
    nonempty = m0[m0.committed_issue_count > 0]
    pool = nonempty[nonempty.sprint_id.map(mid["open_at_checkpoint"]).fillna(False).astype(bool)]
    out["eligibility"] = {
        "test_sprints": int(len(m0)), "auc_M_c0_all_test": float(roc_auc_score(m0.label_delay, m0.m_score)),
        "nonempty": int(len(nonempty)), "auc_M_c0_nonempty": float(roc_auc_score(nonempty.label_delay, nonempty.m_score)),
        "pool": int(len(pool)), "auc_M_c0_pool": float(roc_auc_score(pool.label_delay, pool.m_score)),
        "late_rate_nonempty": float(nonempty.label_delay.mean()), "late_rate_pool": float(pool.label_delay.mean())}
    (R / "review2.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
