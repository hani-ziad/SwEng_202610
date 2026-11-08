"""When are committed issues resolved, relative to the sprint's planned end and its actual closure?
Tests whether the spillover label measures unfinished work or late resolution bookkeeping.
Usage: python resolution_timing.py <commitment_provenance.csv> <tawos_issue.csv> <tawos_sprint.csv> <phase2_frame.csv> <out.csv>
"""
import sys
import numpy as np
import pandas as pd

cp = pd.read_csv(sys.argv[1])
iss = pd.read_csv(sys.argv[2], na_values=["NULL"], usecols=["ID", "Status", "Resolution_Date"], parse_dates=["Resolution_Date"])
sp = pd.read_csv(sys.argv[3], na_values=["NULL"], usecols=["ID", "End_Date", "Complete_Date"], parse_dates=["End_Date", "Complete_Date"])
fr = pd.read_csv(sys.argv[4], usecols=["sprint_id", "nonempty"])
cp = cp[cp.sprint_id.isin(fr.sprint_id)]
m = cp.merge(iss, left_on="raw_issue_id", right_on="ID", how="left").merge(
    sp.rename(columns={"ID": "sid"}), left_on="raw_sprint_id", right_on="sid", how="left")
m["lag_days"] = (m.Resolution_Date - m.End_Date).dt.total_seconds() / 86400
m["after_close_days"] = (m.Resolution_Date - m.Complete_Date).dt.total_seconds() / 86400
R = []
def add(k, v): R.append({"check": k, "value": v}); print(f"{k}: {v}")
add("commitments", len(m))
add("share_completed_by_end_flag", round(m.completed_by_end.mean(), 3))
add("share_no_resolution_date", round(m.Resolution_Date.isna().mean(), 3))
r = m[m.Resolution_Date.notna()]
add("share_resolved_by_planned_end", round((r.lag_days <= 0).mean(), 3))
add("share_resolved_within_1_day_after_end", round(((r.lag_days > 0) & (r.lag_days <= 1)).mean(), 3))
add("share_resolved_by_actual_closure", round((r.after_close_days <= 0).mean(), 3))
add("share_resolved_more_than_30d_after_end", round((r.lag_days > 30).mean(), 3))
add("median_lag_days_resolved_after_end", round(r[r.lag_days > 0].lag_days.median(), 1))
# sprint-level: label recomputed with 'resolved by actual closure' instead of 'by planned end'
m["done_by_close"] = m.Resolution_Date.notna() & (m.Resolution_Date <= m.Complete_Date)
s = m.groupby("sprint_id").agg(n=("raw_issue_id", "size"), done_end=("completed_by_end", "sum"), done_close=("done_by_close", "sum"))
add("sprints", len(s))
add("spill_rate_planned_end", round((s.done_end < s.n).mean(), 3))
add("spill_rate_by_actual_closure", round((s.done_close < s.n).mean(), 3))
add("share_sprints_nothing_done_by_end", round((s.done_end == 0).mean(), 3))
add("share_sprints_nothing_done_by_closure", round((s.done_close == 0).mean(), 3))
# status of issues never resolved
add("unresolved_status_top", "; ".join(f"{k}={v}" for k, v in m[m.Resolution_Date.isna()].Status.value_counts().head(4).items()))
pd.DataFrame(R).to_csv(sys.argv[5], index=False)
