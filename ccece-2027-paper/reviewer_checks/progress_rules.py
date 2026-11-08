"""Learning-free in-sprint rules for spillover, and the resolution-date defect.

Usage: python progress_rules.py <phase3_checkpoint_frame.csv> <out.csv>
Writes <out.csv> (rule AUCs) and <out>_zero_done_by_project.csv.
"""
import sys
import pandas as pd
from sklearn.metrics import roc_auc_score as auc

f = pd.read_csv(sys.argv[1])
rows = []
for c in [0.25, 0.5, 0.75]:
    base = f[(f.checkpoint == c) & f.nonempty & f.open_at_checkpoint]
    for split, x in [("time", base[base.split_nonempty == "test"]), ("all open (LOPO pool)", base)]:
        y = x.label_spillover
        rows.append({"checkpoint": c, "population": split, "n": len(x), "pos": int(y.sum()),
                     "auc_one_minus_share_done": round(auc(y, 1 - x.done_share), 3),
                     "auc_open_committed_count": round(auc(y, x.remaining_n), 3),
                     "auc_committed_count_only": round(auc(y, x.committed_issue_count), 3)})
out = pd.DataFrame(rows)
print(out.to_string(index=False))
out.to_csv(sys.argv[2], index=False)

# Share of nonempty sprints with no committed issue done at the planned end, per project.
end = f[f.nonempty & (f.checkpoint == 1.0)]
z = end.groupby("project").agg(n_nonempty=("done_share", "size"),
                               share_zero_done_at_end=("done_share", lambda v: round((v == 0).mean(), 3))).reset_index()
print(z.to_string(index=False))
z.to_csv(sys.argv[2].replace(".csv", "_zero_done_by_project.csv"), index=False)
