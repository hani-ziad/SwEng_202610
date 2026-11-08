"""Phase 4, experiment 4: sanity checks on the labels (added 8 Nov 2026).

No model is trained here. The script asks whether the two Phase 1 labels
measure what their names say, using only the saved Phase 2 and Phase 3 frames
and the raw exports. Its findings are the reason for experiment 5.

    python -m experiments.e4_sanity_checks      (from the phase-4-fixes folder)

Writes results/e4_sanity_checks.csv (check, value, note).
"""
from __future__ import annotations

import pandas as pd
from sklearn.metrics import roc_auc_score

import phase4
from src.data.load_tawos import load_issues

P2 = phase4.ROOT.parent / "phase-2-explainable-sprint-advisor" / "results" / "phase2_modeling_frame.csv"
P3 = phase4.ROOT.parent / "phase-3-in-sprint-early-warning" / "results" / "phase3_checkpoint_frame.csv"
EXT = phase4.ROOT / "external" / "data" / "raw"


def main():
    rows = []

    def add(check, value, note=""):
        rows.append({"check": check, "value": value, "note": note})

    # 1. spillover: base rate and how strongly it follows sprint size
    p2 = pd.read_csv(P2)
    ne = p2[p2["nonempty"]]
    add("spillover_rate_nonempty", round(ne.label_spillover.mean(), 3),
        f"{int(ne.label_spillover.sum())} of {len(ne)} sprints with committed work")
    for lo, hi in [(0, 1), (1, 2), (2, 3), (3, 5), (5, 10), (10, 10_000)]:
        b = ne[(ne.committed_issue_count > lo) & (ne.committed_issue_count <= hi)]
        add(f"spillover_rate_{lo + 1}_to_{hi}_issues", round(b.label_spillover.mean(), 3), f"n={len(b)}")
    add("median_carry_share_nonempty", round(ne.carry_ratio.median(), 3),
        "share of committed issues still unresolved at the planned end, median sprint")

    # 2. Phase 3 spillover: a raw count against the reported model (same open-sprint population)
    p3 = pd.read_csv(P3)
    for cp in (0.25, 0.5, 0.75):
        s = p3[(p3.checkpoint == cp) & (p3.split_nonempty == "test") & p3.open_at_checkpoint]
        add(f"p3_spillover_test_n_cp{cp}", len(s), f"{int((1 - s.label_spillover).sum())} without spillover")
        add(f"p3_spillover_auc_remaining_count_cp{cp}", round(roc_auc_score(s.label_spillover, s.remaining_n), 3),
            "open committed issues at the checkpoint, no model")
        add(f"p3_spillover_auc_committed_count_cp{cp}",
            round(roc_auc_score(s.label_spillover, s.committed_issue_count), 3), "committed issues, no model")
        f = p3[(p3.checkpoint == cp) & (p3.split_full == "test") & p3.open_at_checkpoint]
        add(f"p3_late_auc_remaining_count_cp{cp}", round(roc_auc_score(f.label_delay, f.remaining_n), 3),
            f"n={len(f)}; progress says nothing about late closure")

    # 3. late closure: weekends
    sp = p3[p3.checkpoint == 0.5].drop_duplicates("sprint_id")
    late = sp[sp.label_delay == 1]
    end = pd.to_datetime(late.end_date, errors="coerce")
    add("late_sprints_phase1_label", len(late), f"of {len(sp)}")
    add("late_share_within_3_days", round((late.late_days <= 3).mean(), 3))
    add("late_share_end_fri_to_sun_closed_within_3_5_days",
        round(((end.dt.dayofweek >= 4) & (late.late_days <= 3.5)).mean(), 3),
        "looks like closing on the next working day; dates are UTC")
    add("late_median_days", round(late.late_days.median(), 2))

    # 4. resolution dates: share of finished issues (Closed, Done, Resolved) without one
    done = {"closed", "done", "resolved"}
    tawos = load_issues(phase4.phase3.RAW_DIR)
    closed = tawos[tawos.status.astype(str).str.lower().isin(done)]
    for proj, g in closed.groupby("project"):
        add(f"closed_without_resolution_{proj}", round(g.resolution_date.isna().mean(), 3), f"n={len(g)}")
    for inst in ("mariadb", "mongodb"):
        i = load_issues(EXT / inst)
        c = i[i.status.astype(str).str.lower().isin(done)]
        add(f"closed_without_resolution_{inst}", round(c.resolution_date.isna().mean(), 3), f"n={len(c)}")

    # 5. outside projects: the spillover label is close to constant
    ext = pd.read_csv(phase4.RESULTS / "e3_external_projects.csv")
    for _, r in ext.iterrows():
        add(f"external_spillover_rate_{r.project}", round(r.spill_rate_nonempty, 3), f"nonempty={r.nonempty}")

    out = pd.DataFrame(rows)
    out.to_csv(phase4.RESULTS / "e4_sanity_checks.csv", index=False)
    pd.set_option("display.width", 200)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
