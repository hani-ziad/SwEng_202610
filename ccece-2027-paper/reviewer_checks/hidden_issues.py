"""Share of each project's issues an anonymous reader of public Jira cannot see (added 14 Nov 2026).

Private, deleted and moved issues keep their numbers, so the share of numbers missing from the export
bounds how much of a project is hidden. A hidden issue that sat in a sprint makes that sprint's committed
scope look smaller than it was. The gap is measured from the first issue number that ever entered a
sprint to the highest number in the project.

Usage: python hidden_issues.py <tawos_issue.csv> <out.csv>
"""
import sys
import pandas as pd

i = pd.read_csv(sys.argv[1], usecols=["Issue_Key", "Project_Name", "Sprint_ID"], na_values=["NULL"], low_memory=False)
i["prefix"] = i.Issue_Key.str.rsplit("-", n=1).str[0]
i["num"] = pd.to_numeric(i.Issue_Key.str.rsplit("-", n=1).str[1], errors="coerce")
rows = []
for (p, pre), g in i.groupby(["Project_Name", "prefix"]):
    if len(g) < 200:  # stray keys of issues moved in from other projects
        continue
    lo, hi = g[g.Sprint_ID.notna()].num.min(), g.num.max()
    era = g[(g.num >= lo) & (g.num <= hi)]
    rows.append({"project": p, "key": pre, "first_sprint_issue": int(lo), "highest": int(hi),
                 "visible": len(era), "gap": round(1 - len(era) / (hi - lo + 1), 4)})
out = pd.DataFrame(rows)
out.to_csv(sys.argv[2], index=False)
print(out.to_string(index=False))
print("median", round(out.gap.median(), 3), "min", out.gap.min(), "max", out.gap.max())
