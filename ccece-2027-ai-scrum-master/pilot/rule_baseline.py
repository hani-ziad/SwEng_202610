"""Step 3 of the pilot (added 8 Nov 2026): chance-corrected agreement, a past-lateness rule,
and how often each arm changed its call between the two checkpoints.

Reads results/replay_all_arms.csv (written by score.py) and the Phase 2 modelling frame
(for each sprint's late rate over its last three sprints). Writes results/rule_baseline.json.
Every number the paper quotes from this check comes from that file.

The rule is the one a Scrum Master could apply by hand at planning: warn if at least half of
the team's last three sprints closed late (hist_late_rate_3 >= 0.5).

Usage (from the ccece-2027-ai-scrum-master folder):  python -m pilot.rule_baseline
Optional arguments: replay_all_arms.csv phase2_modeling_frame.csv output.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SEED = 42
N_BOOT = 2000

REPLAY = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results" / "replay_all_arms.csv"
FRAME = Path(sys.argv[2]) if len(sys.argv) > 2 else (
    REPO / "phase-2-explainable-sprint-advisor" / "results" / "phase2_modeling_frame.csv")
OUT = Path(sys.argv[3]) if len(sys.argv) > 3 else HERE / "results" / "rule_baseline.json"


def kappa(a, b):
    """Cohen's kappa for two binary decision vectors."""
    a, b = np.asarray(a), np.asarray(b)
    po = (a == b).mean()
    pa, pb = a.mean(), b.mean()
    pe = pa * pb + (1 - pa) * (1 - pb)
    return float((po - pe) / (1 - pe))


def main():
    d = pd.read_csv(REPLAY)
    f = pd.read_csv(FRAME)
    out = {"n_sprints": int(d.sprint_id.nunique()), "n_points": int(len(d))}

    # Agreement with M, raw and chance-corrected, pooled over both checkpoints.
    for arm in ("L", "G"):
        w = d[f"{arm}_warn"]
        dep = d[w != d.m_warn]
        out[arm] = {
            "agreement": float((w == d.m_warn).mean()),
            "cohen_kappa": kappa(w, d.m_warn),
            "departures": int(len(dep)),
            "departures_correct": int((dep[f"{arm}_warn"] == dep.late).sum()),
            "departures_to_silent": int((dep[f"{arm}_warn"] == 0).sum()),
        }

    # Sprints on which each arm changed its call between c = 0 and c = 0.5.
    changes = {}
    for col, name in (("m_warn", "M"), ("L_warn", "L"), ("G_warn", "G")):
        p = d.pivot_table(index="sprint_id", columns="checkpoint", values=col)
        changes[name] = int((p[0.0] != p[0.5]).sum())
    out["sprints_with_changed_call"] = changes

    # Past-lateness rule at planning.
    s = d[d.checkpoint == 0].merge(f[["sprint_id", "hist_late_rate_3", "label_delay"]],
                                   on="sprint_id", how="left", validate="one_to_one")
    assert s.hist_late_rate_3.notna().all(), "history missing for a pilot sprint"
    assert (s.late == s.label_delay).all(), "label mismatch between replay file and frame"
    rule = (s.hist_late_rate_3 >= 0.5).astype(int)
    rng = np.random.default_rng(SEED)
    diffs = []
    for _ in range(N_BOOT):
        b = s.iloc[rng.integers(0, len(s), len(s))]
        if b.late.nunique() < 2:
            continue
        diffs.append(roc_auc_score(b.late, b.m_score) - roc_auc_score(b.late, b.hist_late_rate_3))
    out["rule_c0"] = {
        "definition": "warn if hist_late_rate_3 >= 0.5",
        "warn_rate": float(rule.mean()),
        "sensitivity": float(rule[s.late == 1].mean()),
        "specificity": float(1 - rule[s.late == 0].mean()),
        "auc_hist_late_rate_3": float(roc_auc_score(s.late, s.hist_late_rate_3)),
        "agreement_with_M": float((rule == s.m_warn).mean()),
        "agreement_with_L": float((rule == s.L_warn).mean()),
        "agreement_with_G": float((rule == s.G_warn).mean()),
        "auc_M_minus_rule": float(roc_auc_score(s.late, s.m_score)
                                  - roc_auc_score(s.late, s.hist_late_rate_3)),
        "auc_M_minus_rule_ci": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
    }

    # Project mix of the 40 sprints.
    out["sprints_per_project"] = {k: int(v) for k, v in s.project.value_counts().items()}
    out["late_per_project"] = {k: int(v) for k, v in s.groupby("project").late.sum().items()}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
