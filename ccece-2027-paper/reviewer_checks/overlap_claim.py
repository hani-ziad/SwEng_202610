"""Check the introduction's claim: in 71% of sprints the previous sprint had not reached its planned end.
Previous = the project's closed sprint with the latest start before this one.
Usage: python overlap_claim.py <tawos_sprint.csv> <phase2_frame.csv> <out.csv>
"""
import sys
import pandas as pd
s = pd.read_csv(sys.argv[1], na_values=["NULL"], parse_dates=["Start_Date", "End_Date", "Complete_Date"])
fr = pd.read_csv(sys.argv[2], usecols=["sprint_id", "project"])
fr["ID"] = fr.sprint_id.str.split("__").str[1].astype(int)
s = s[(s.State == "CLOSED") & s.Project_Name.isin(fr.project.unique())].sort_values(["Project_Name", "Start_Date", "ID"])
s["prev_end"] = s.groupby("Project_Name").End_Date.shift()
s["prev_close"] = s.groupby("Project_Name").Complete_Date.shift()
m = s[s.ID.isin(fr.ID)]
R = [{"check": "modelled_sprints_with_predecessor", "value": int(m.prev_end.notna().sum())},
     {"check": "share_prev_planned_end_after_start", "value": round((m.prev_end > m.Start_Date).mean(), 3)},
     {"check": "share_prev_closed_after_start", "value": round((m.prev_close > m.Start_Date).mean(), 3)},
     {"check": "share_prev_unfinished_either", "value": round(((m.prev_end > m.Start_Date) | (m.prev_close > m.Start_Date)).mean(), 3)}]
print(pd.DataFrame(R).to_string(index=False)); pd.DataFrame(R).to_csv(sys.argv[3], index=False)
