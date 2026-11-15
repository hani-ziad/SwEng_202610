"""Step 4b of the study: two independent rater agents judge every warning.

Each warning is shown with the board its agent saw (for G, with the model lines), but without
the arm, the run or what the sprint went on to do. Each rater answers two questions:

  claims_ok      Is every factual statement in the message true of the board? A statement
                 counts as true when the board states it or it follows by simple arithmetic or
                 a fair rounding ("about 4 per sprint" for 3.7). Advice and opinions are not
                 claims. A statement about the team or the sprint that the board does not
                 support, or contradicts, makes it false.
  close_on_date  Does the message advise closing the sprint on its planned end date while
                 accepting that committed work may still be unfinished then (carry it over,
                 move it out at the close, close whatever is left)? Advice to cut or re-plan
                 scope, or to finish the work by the end date, is 0 unless it also says to
                 close on the date with work left over.

Raters A and B are separate Claude sub-agents (claude-opus-5-5) with the same instructions and
the same messages in different orders. Agreement is reported as Cohen's kappa; a message counts
as unsupported or as close-on-date only when both raters say so (and, separately, when either does).
The 81 warnings of the 4 Nov pilot are rated the same way, so the raters can be compared with
the pilot's original hand coding.

Usage (from the ccece-2027-ai-scrum-master folder):
  python -m study.rate build     # writes rate_prompts/{A,B}/*.txt and rate_prompts/key.csv
  # give each prompt file to a rater agent; save answers as rate_outputs/{A,B}/<name>.jsonl
  python -m study.rate score     # writes results/ratings.csv and results/ratings_summary.json
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

import study

OUT = study.HERE
BATCH = 25
SEED = 2027

INSTRUCTIONS = """You are checking messages that an AI assistant wrote to a software team about one of their
sprints. For each item below you get the sprint board exactly as the assistant saw it, and the
message it sent. Judge each item on its own. Do not use any tools, files or web access.

Answer two questions per item.

1. claims_ok: is every factual statement in the message true of the board? A statement is true
   when the board states it, or when it follows from the board by simple arithmetic or a fair
   rounding (for example "about 4 per sprint" when the board says 3.7, or "two of the last three
   sprints closed late" when 3 are available and the late share is 0.67). Advice, plans and
   opinions are not factual statements. If any statement about the team, the sprint, its history
   or the model is not supported by the board, or contradicts it, answer false and quote it.

2. close_on_date: does the message advise closing the sprint on its planned end date while
   accepting that committed work may still be unfinished at that point (carry it over, move it
   out at the close, close it whatever is left, close it even if an item is not done)? Answer 1
   if so. Advice only to cut or re-plan scope, to finish the work by the end date, or to watch
   progress is 0, unless it also says to close on the date with work left over.

Answer with one JSON object per item, one per line, and nothing else:
{"id": "<item id>", "claims_ok": true or false, "bad_claims": "<the unsupported statements, quoted, or empty>", "close_on_date": 0 or 1}
"""


def items():
    s = pd.read_csv(OUT / "results" / "messages.csv").fillna("")
    s["set"] = "study"
    p = pd.read_csv(OUT / "results" / "pilot_messages.csv").fillna("")
    p["set"] = "pilot"
    d = pd.concat([s, p], ignore_index=True)
    d["item"] = [f"M{i + 1:04d}" for i in range(len(d))]
    d["shown"] = [b + ("\n" + m if a == "G" and m else "") for b, m, a in zip(d.board_text, d.model_text, d.arm)]
    return d


def build():
    d = items()
    d[["item", "set", "run", "arm", "board_id", "checkpoint", "late", "message"]].to_csv(
        OUT / "rate_prompts_key.csv", index=False)
    man = []
    for rater, seed in (("A", SEED), ("B", SEED + 100)):
        order = list(np.random.default_rng(seed).permutation(d.item.to_numpy()))
        folder = OUT / "rate_prompts" / rater
        folder.mkdir(parents=True, exist_ok=True)
        by = d.set_index("item")
        for k in range(0, len(order), BATCH):
            part = order[k:k + BATCH]
            body = "\n\n".join(f"=== Item {i} ===\nBoard:\n{by.loc[i, 'shown']}\nMessage: {by.loc[i, 'message']}"
                               for i in part)
            name = f"{rater}_b{k // BATCH + 1:02d}"
            (folder / f"{name}.txt").write_text(INSTRUCTIONS + "\n" + body + "\n")
            man.append({"rater": rater, "prompt": name, "items": part})
    (OUT / "rate_prompts" / "manifest.json").write_text(json.dumps(man, indent=1))
    print(len(d), "items;", len(man), "prompt files")


def kappa(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    po = (a == b).mean()
    pa, pb = a.mean(), b.mean()
    pe = pa * pb + (1 - pa) * (1 - pb)
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def score():
    d = items().set_index("item")
    man = json.loads((OUT / "rate_prompts" / "manifest.json").read_text())
    got = {"A": {}, "B": {}}
    for m in man:
        f = OUT / "rate_outputs" / m["rater"] / f"{m['prompt']}.jsonl"
        rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
        assert sorted(r["id"] for r in rows) == sorted(m["items"]), m["prompt"]
        for r in rows:
            got[m["rater"]][r["id"]] = r
    for r in ("A", "B"):
        d[f"ok_{r}"] = [int(bool(got[r][i]["claims_ok"])) for i in d.index]
        d[f"bad_{r}"] = [got[r][i].get("bad_claims", "") for i in d.index]
        d[f"close_{r}"] = [int(got[r][i]["close_on_date"]) for i in d.index]
    d["unsupported_both"] = ((d.ok_A == 0) & (d.ok_B == 0)).astype(int)
    d["unsupported_either"] = ((d.ok_A == 0) | (d.ok_B == 0)).astype(int)
    d["close_both"] = ((d.close_A == 1) & (d.close_B == 1)).astype(int)
    d["close_either"] = ((d.close_A == 1) | (d.close_B == 1)).astype(int)
    d.drop(columns=["board_text", "model_text", "shown"]).to_csv(OUT / "results" / "ratings.csv")

    summ = {}
    for st, g in d.groupby("set"):
        e = {"n": int(len(g)),
             "kappa_claims": kappa(g.ok_A, g.ok_B), "kappa_close": kappa(g.close_A, g.close_B),
             "agree_claims": float((g.ok_A == g.ok_B).mean()), "agree_close": float((g.close_A == g.close_B).mean())}
        for arm in ("L", "G"):
            h = g[g.arm == arm]
            e[arm] = {"n": int(len(h)), "unsupported_both": int(h.unsupported_both.sum()),
                      "unsupported_either": int(h.unsupported_either.sum()),
                      "close_both": int(h.close_both.sum()), "close_either": int(h.close_either.sum())}
            if st == "study":
                e[arm]["by_run"] = {int(r): {"n": int(len(x)), "unsupported_both": int(x.unsupported_both.sum()),
                                             "close_both": int(x.close_both.sum())}
                                    for r, x in h.groupby("run")}
                for c, x in h.groupby("checkpoint"):
                    e[arm][f"c{int(c * 100)}"] = {"n": int(len(x)), "close_both": int(x.close_both.sum()),
                                                  "unsupported_both": int(x.unsupported_both.sum())}
        summ[st] = e
    # pilot: compare with the original hand coding
    coded = pd.read_csv(study.HERE.parent / "pilot" / "results" / "warning_messages_coded.csv")
    pk = d[d.set == "pilot"].reset_index().merge(
        coded.rename(columns={"board_id": "board_id", "closes_on_date_regardless": "orig"}),
        on=["arm", "board_id", "checkpoint"], how="inner")
    summ["pilot"]["matched_to_original_coding"] = int(len(pk))
    summ["pilot"]["original_close"] = int(pk.orig.sum())
    summ["pilot"]["kappa_A_vs_original"] = kappa(pk.close_A, pk.orig)
    summ["pilot"]["kappa_B_vs_original"] = kappa(pk.close_B, pk.orig)
    summ["pilot"]["kappa_both_vs_original"] = kappa(pk.close_both, pk.orig)
    (OUT / "results" / "ratings_summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


def collect(folder):
    """Save each rater's answer verbatim and audit its tool calls (as study.collect does)."""
    from pathlib import Path

    from study.collect import read_transcript
    man = {f"rate {m['prompt']}": m for m in json.loads((OUT / "rate_prompts" / "manifest.json").read_text())}
    found, audit, problems = {}, [], []
    for meta in sorted(Path(folder).glob("agent-*.meta.json")):
        desc = json.loads(meta.read_text()).get("description", "")
        if desc in man:
            tr = meta.with_name(meta.name.replace(".meta.json", ".jsonl"))
            found.setdefault(desc, []).append((tr.stat().st_mtime, *read_transcript(tr)))
    for desc, m in man.items():
        if desc not in found:
            problems.append(f"missing: {desc}")
            continue
        _, calls, models, answer = sorted(found[desc])[-1]
        expected = f"/home/claude/agentbox/rate/{m['rater']}/{m['prompt']}.txt"
        clean = len(calls) == 1 and calls[0]["tool"] == "Read" and calls[0]["input"].get("file_path") == expected
        lines = [l for l in (answer or "").splitlines() if l.strip()]
        try:
            complete = sorted(json.loads(l)["id"] for l in lines) == sorted(m["items"])
        except (json.JSONDecodeError, KeyError):
            complete = False
        audit.append({"rater": m["rater"], "prompt": m["prompt"], "models": models, "tool_calls": calls,
                      "only_read_own_prompt": clean, "answers_complete": complete})
        if clean and complete:
            d = OUT / "rate_outputs" / m["rater"]
            d.mkdir(parents=True, exist_ok=True)
            (d / f"{m['prompt']}.jsonl").write_text("\n".join(lines) + "\n")
        else:
            problems.append(f"bad: {desc} clean={clean} complete={complete}")
    (OUT / "rate_outputs").mkdir(exist_ok=True)
    (OUT / "rate_outputs" / "audit.json").write_text(json.dumps(audit, indent=1))
    print(f"{len(audit)} of {len(man)} collected; problems: {len(problems)}")
    for p in problems:
        print(" ", p)


if __name__ == "__main__":
    if sys.argv[1] == "collect":
        collect(sys.argv[2])
    else:
        {"build": build, "score": score}[sys.argv[1]]()
