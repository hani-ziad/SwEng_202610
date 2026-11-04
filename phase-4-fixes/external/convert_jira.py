"""Convert a Jira pull (from pull_jira_browser.js) into the TAWOS CSV layout.

Generalised from external-validation/convert_mariadb.py (same rules) so it works
for any Jira Server instance. One instance per output folder, because issue and
sprint ids are only unique inside one Jira instance.

Output: <out_dir>/{tawos_sprint,tawos_issue,tawos_change_log}.csv, with the
same columns the frozen Phase 1 loader (src/data/load_tawos.py) reads, so the
reconstruction, labels and features run unchanged.

Choices (all fixed before looking at any outcome):
* Sprint metadata comes from the Sprint field on issues (Jira Server serialises
  id, board, state, name and the four dates). A sprint shared by both projects
  is assigned to the project that owns most of its issues; only that project's
  issues can enter its reconstructed scope (TAWOS keeps sprints per project too).
* Only each project's home board is kept (the board that holds most of its
  sprints); sprints on other teams' boards are dropped because their scope is
  mostly outside the project.
* All timestamps are converted to UTC and stored without a zone.
* Sprint ids are Jira ids in every table (Sprint.ID == Sprint.JiraID), so the
  static Issue.Sprint_ID and the Change_Log values share one namespace.
* Story points are copied when the instance has them; the transfer test drops
  the story-point features anyway, because not every source uses them.

Usage: python convert_jira.py <pull.json> <out_dir> KEY="Project name" [KEY="..."]
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

PROJECT_NAMES: dict[str, str] = {}


def to_utc(value):
    if value is None or value == "" or str(value).strip() in {"<null>", "null", "None"}:
        return pd.NaT
    ts = pd.to_datetime(str(value), errors="coerce", utc=True)
    return ts.tz_convert(None) if ts is not pd.NaT and not pd.isna(ts) else pd.NaT


def parse_sprint(v) -> dict:
    if isinstance(v, dict):
        return v
    m = re.search(r"\[(.*)\]\s*$", str(v), re.S)
    out = {}
    if not m:
        return out
    # values never contain '=' ; split on ',key=' boundaries
    for part in re.split(r",(?=[A-Za-z]+=)", m.group(1)):
        if "=" in part:
            k, val = part.split("=", 1)
            out[k.strip()] = val
    return out


def main(src: Path, out_dir: Path) -> None:
    data = json.loads(src.read_text(encoding="utf-8"))
    issues = data["issues"]

    sprints: dict[str, dict] = {}
    owner_votes: dict[str, Counter] = defaultdict(Counter)
    for it in issues:
        for raw in it.get("spr") or []:
            s = parse_sprint(raw)
            sid = str(s.get("id", "")).strip()
            if not sid.isdigit():
                continue
            sprints.setdefault(sid, s)
            owner_votes[sid][it["p"]] += 1

    srows = []
    for sid, s in sprints.items():
        proj = owner_votes[sid].most_common(1)[0][0]
        srows.append({
            "ID": int(sid), "JiraID": int(sid), "Name": s.get("name"),
            "State": str(s.get("state", "")).upper(),
            "Start_Date": to_utc(s.get("startDate")),
            "Activated_Date": to_utc(s.get("activatedDate")),
            "End_Date": to_utc(s.get("endDate")),
            "Complete_Date": to_utc(s.get("completeDate")),
            "Project_Name": PROJECT_NAMES[proj],
            "Board_ID": s.get("rapidViewId") or s.get("boardId"),
            "Shared_Projects": "+".join(sorted(owner_votes[sid])),
        })
    sdf = pd.DataFrame(srows).sort_values(["Project_Name", "Start_Date"])
    # Keep each project's home board only (the board that holds most of its
    # sprints). Stray sprints on other teams' boards (e.g. a server release
    # sprint that a MaxScale issue passed through) have scope we cannot see.
    home = sdf.groupby("Project_Name")["Board_ID"].agg(lambda b: b.value_counts().idxmax())
    stray = sdf["Board_ID"] != sdf["Project_Name"].map(home)
    print("home boards:", home.to_dict(), "| stray sprints dropped:", int(stray.sum()))
    sdf = sdf[~stray]

    irows, crows = [], []
    for it in issues:
        spr_ids = [str(parse_sprint(x).get("id", "")) for x in (it.get("spr") or [])]
        spr_ids = [x for x in spr_ids if x.isdigit()]
        irows.append({
            "ID": int(it["id"]), "Issue_Key": it["k"], "Type": it.get("t"), "Status": it.get("s"),
            "Story_Point": it.get("sp"), "Priority": it.get("pr"),
            "Creation_Date": to_utc(it.get("c")), "Resolution_Date": to_utc(it.get("r")),
            "Sprint_ID": int(spr_ids[-1]) if spr_ids else None,
            "Assignee_ID": it.get("a"), "Project_Name": PROJECT_NAMES[it["p"]],
            "Original_Estimate_Hours": (it["oe"] / 3600.0) if it.get("oe") else None,
        })
        for hid, field, frm, to, frm_s, to_s, created in it.get("ch") or []:
            crows.append({
                "ID": int(hid), "Issue_ID": int(it["id"]), "Field": field,
                "From_Value": frm, "To_Value": to, "From_String": frm_s, "To_String": to_s,
                "Change_Type": "OTHER" if field.lower() == "sprint" else "STORY_POINT",
                "Creation_Date": to_utc(created),
            })
    idf = pd.DataFrame(irows)
    cdf = pd.DataFrame(crows)

    out_dir.mkdir(parents=True, exist_ok=True)
    fmt = "%Y-%m-%d %H:%M:%S"
    sdf.to_csv(out_dir / "tawos_sprint.csv", index=False, date_format=fmt)
    idf.to_csv(out_dir / "tawos_issue.csv", index=False, date_format=fmt)
    cdf.to_csv(out_dir / "tawos_change_log.csv", index=False, date_format=fmt)

    print(f"issues {len(idf)} | change events {len(cdf)} | sprints {len(sdf)}")
    print(sdf.groupby(["Project_Name", "State"]).size().to_string())
    print("shared sprints:", int((sdf["Shared_Projects"].str.contains(r"\+")).sum()))
    trunc = sum(1 for it in issues if it.get("clt") and it.get("clm") and it["clt"] > it["clm"])
    print("issues with truncated changelog:", trunc, "| pull errors:", len(data.get("errors", [])))


if __name__ == "__main__":
    for kv in sys.argv[3:]:
        k, v = kv.split("=", 1)
        PROJECT_NAMES[k] = v
    main(Path(sys.argv[1]), Path(sys.argv[2]))
