"""Step 4a of the study: automatic check of every number an agent quoted in a warning.

A number in a warning counts as supported when it appears on the board the agent saw (for G,
including the model lines), or follows from the board by one simple step: a count behind a
rate (2 of the last 3 sprints), a percentage of a share, its complement, a difference of two
board counts (still open = committed - resolved), a sum of story points listed on the board,
or a count of listed issues by type or estimate. Numbers hedged with "about", "around",
"almost", "nearly", "over", "under", "more than", "less than", "roughly" or "~" may be off by
up to 15% (and by up to one unit). Number words from one to twenty are read as numbers.
Everything else is flagged unsupported. This is deliberately strict: a flag means the number
could not be traced, not that it is wrong; the rater agents (rate.py) judge every claim in full.

Writes study/results/factcheck_auto.csv (one row per warning) and factcheck_auto.json.
Also checks the 81 warnings of the 4 Nov pilot (pilot/results/replay_all_arms.csv with the
pilot prompts) when those files are present.

Usage (from the ccece-2027-ai-scrum-master folder):  python -m study.factcheck
"""
from __future__ import annotations

import itertools
import json
import re

import numpy as np
import pandas as pd

import study

OUT = study.HERE / "results"
WORDS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
    "sixteen seventeen eighteen nineteen twenty".split())}
HEDGE = re.compile(r"(about|around|almost|nearly|over|under|more than|less than|roughly|approximately|~|"
                   r"close to|just over|just under|well over|some)\s*$", re.I)
NUM = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(\s*%)?|\b(" + "|".join(WORDS) + r")\b", re.I)


def board_numbers(text: str) -> set[float]:
    vals = [float(x) for x in re.findall(r"(?<![\w])(\d+(?:\.\d+)?)", text)]
    s = set(vals)
    # shares and rates: percentages and complements
    for v in vals:
        if 0 < v < 1:
            s |= {round(100 * v), round(100 * (1 - v)), round(1 - v, 2)}
    # counts behind "share that closed late x" with n sprints available
    m = re.search(r"(\d+) available;.*?closed more than a day late ([\d.]+)", text)
    if m:
        n, r = int(m.group(1)), float(m.group(2))
        s |= {round(n * r), n}
    # line items
    items = re.findall(r"^- (committed: |added later: )?([^,]+), (no estimate|[\d.]+ SP)(?:, (.*))?$", text, re.M)
    types = [t[1].strip() for t in items]
    for t in set(types):
        s.add(types.count(t))
    s.add(sum(1 for t in items if t[2] == "no estimate"))
    s.add(sum(1 for t in items if t[2] != "no estimate"))
    open_items = [t for t in items if t[3] and "not resolved" in t[3]]
    s.add(len(open_items))
    sps = [float(t[2].split()[0]) for t in items if t[2] != "no estimate"]
    s.add(sum(sps))
    open_sp = [float(t[2].split()[0]) for t in open_items if t[2] != "no estimate"]
    s.add(sum(open_sp))
    for typ in set(types):
        s.add(sum(1 for t in open_items if t[1].strip() == typ))
    # story-point sums of the obvious subsets: committed or added, open or all, by type
    def sp_sum(rows):
        return sum(float(t[2].split()[0]) for t in rows if t[2] != "no estimate")
    groups = {"c": [t for t in items if t[0].startswith("committed")],
              "a": [t for t in items if t[0].startswith("added")]}
    for rows in list(groups.values()) + [items]:
        open_rows = [t for t in rows if t[3] and "not resolved" in t[3]]
        s |= {sp_sum(rows), sp_sum(open_rows), len(rows), len(open_rows)}
        for typ in set(types):
            s |= {sp_sum([t for t in rows if t[1].strip() == typ]),
                  sp_sum([t for t in open_rows if t[1].strip() == typ])}
    # pairwise differences and sums of the headline counts
    head = [float(x) for x in re.findall(r"(?<![\w.])(\d+)(?![\d.])", text.split("Issues:")[0])]
    for a, b in itertools.combinations(head, 2):
        s |= {abs(a - b), a + b}
    return s


def check(message: str, board: str):
    allowed = board_numbers(board)
    out = []
    for m in NUM.finditer(message):
        raw = m.group(0)
        v = float(WORDS[m.group(3).lower()]) if m.group(3) else float(m.group(1))
        if m.group(3) and v in (0, 1) :
            continue                    # "one"/"zero" used as words ("one bug", "no one")
        prev = message[max(0, m.start() - 14):m.start()]
        hedged = bool(HEDGE.search(prev))
        tol = max(1.0, 0.15 * v) if hedged else 0.0
        ok = any(abs(v - a) <= tol + 1e-9 for a in allowed)
        out.append({"value": raw.strip(), "hedged": hedged, "supported": ok})
    return out


def run(msgs: pd.DataFrame, label: str):
    rows = []
    for r in msgs.itertuples(index=False):
        board = r.board_text + ("\n" + r.model_text if r.arm == "G" else "")
        res = check(r.message, board)
        bad = [x["value"] for x in res if not x["supported"]]
        rows.append({**r._asdict(), "n_numbers": len(res), "n_unsupported": len(bad),
                     "unsupported": "; ".join(bad), "all_supported": int(not bad)})
    df = pd.DataFrame(rows)
    summ = {"set": label, "warnings": int(len(df)), "warnings_with_numbers": int((df.n_numbers > 0).sum()),
            "numbers": int(df.n_numbers.sum()), "unsupported_numbers": int(df.n_unsupported.sum()),
            "warnings_all_supported": int(df.all_supported.sum())}
    for arm in ("L", "G"):
        g = df[df.arm == arm]
        summ[f"{arm}_warnings"] = int(len(g))
        summ[f"{arm}_all_supported"] = int(g.all_supported.sum())
        summ[f"{arm}_numbers"] = int(g.n_numbers.sum())
        summ[f"{arm}_unsupported_numbers"] = int(g.n_unsupported.sum())
    return df, summ


def pilot_messages():
    root = study.HERE.parent / "pilot"
    d = pd.read_csv(root / "results" / "replay_all_arms.csv")
    man = json.loads((root / "prompts" / "manifest.json").read_text())
    texts = {}
    for m in man:
        body = (root / "prompts" / f"{m['prompt']}.txt").read_text()
        for blk in body.split("=== Board ")[1:]:
            bid, txt = blk.split(" ===\n", 1)
            texts[(m["arm"], bid.strip(), m["checkpoint"])] = txt.strip()
    rows = []
    for r in d.itertuples(index=False):
        for arm in ("L", "G"):
            if getattr(r, f"{arm}_warn") == 1:
                t = texts[(arm, r.board_id, r.checkpoint)]
                board, _, model = t.partition("\nRisk model:")
                rows.append({"run": 0, "arm": arm, "board_id": r.board_id, "checkpoint": r.checkpoint,
                             "late": r.late, "message": getattr(r, f"{arm}_msg"), "board_text": board,
                             "model_text": ("Risk model:" + model) if model else ""})
    return pd.DataFrame(rows)


def main():
    msgs = pd.read_csv(OUT / "messages.csv").fillna("")
    df, summ = run(msgs, "study")
    df.drop(columns=["board_text", "model_text"]).to_csv(OUT / "factcheck_auto.csv", index=False)
    allsum = {"study": summ}
    try:
        pm = pilot_messages()
        pdf, psum = run(pm, "pilot_4nov")
        pdf.drop(columns=["board_text", "model_text"]).to_csv(OUT / "factcheck_auto_pilot.csv", index=False)
        pm.to_csv(OUT / "pilot_messages.csv", index=False)
        allsum["pilot_4nov"] = psum
    except FileNotFoundError as e:
        print("pilot files not found, skipped:", e)
    (OUT / "factcheck_auto.json").write_text(json.dumps(allsum, indent=1))
    print(json.dumps(allsum, indent=1))


if __name__ == "__main__":
    main()
