"""Arm-masked re-rating (added 15 Nov 2026, night, after the second external review).

In study/rate.py the model lines were shown only with the grounded agent's warnings, so a rater could
tell the arm. Here every one of the 395 study warnings is shown with the same evidence: the board and
the model lines for that board, whichever agent wrote it. Raters are told that the writer may or may
not have seen the model lines. Same two questions, same JSON answer, two raters (A2, B2) with the
warnings in different orders, 25 per prompt.

Usage (from the ccece-2027-ai-scrum-master folder):
  python -m study.rate_masked build
  python -m study.rate_masked collect <transcript folder>
  python -m study.rate_masked score       # writes results/ratings_masked.csv and ratings_masked_summary.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import study
from study.collect import read_transcript
from study.rate import INSTRUCTIONS, kappa

OUT = study.HERE
D = OUT / "rate_masked"
BATCH = 25
SEED = 2027 + 300
NOTE = ("Every board below ends with the output of two prediction models for that sprint. The message's "
        "writer may or may not have seen those lines; judge the message only against what is shown.\n")


def items():
    s = pd.read_csv(OUT / "results" / "messages.csv").fillna("")
    s["item"] = [f"W{i + 1:04d}" for i in range(len(s))]
    s["shown"] = s.board_text + "\n" + s.model_text
    return s


def build():
    d = items()
    d[["item", "run", "arm", "board_id", "checkpoint", "late"]].to_csv(D.with_name("rate_masked_key.csv"), index=False)
    man = []
    by = d.set_index("item")
    for rater, seed in (("A2", SEED), ("B2", SEED + 100)):
        order = list(np.random.default_rng(seed).permutation(d.item.to_numpy()))
        (D / "prompts" / rater).mkdir(parents=True, exist_ok=True)
        for k in range(0, len(order), BATCH):
            part = order[k:k + BATCH]
            body = "\n\n".join(f"=== Item {i} ===\nBoard:\n{by.loc[i, 'shown']}\nMessage: {by.loc[i, 'message']}"
                               for i in part)
            name = f"{rater}_b{k // BATCH + 1:02d}"
            (D / "prompts" / rater / f"{name}.txt").write_text(INSTRUCTIONS + "\n" + NOTE + "\n" + body + "\n")
            man.append({"rater": rater, "prompt": name, "items": part})
    (D / "prompts" / "manifest.json").write_text(json.dumps(man, indent=1))
    print(len(d), "items;", len(man), "prompt files")


def collect(folder):
    man = {f"rate2 {m['prompt']}": m for m in json.loads((D / "prompts" / "manifest.json").read_text())}
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
        expected = f"/home/claude/agentbox/rate2/{m['rater']}/{m['prompt']}.txt"
        clean = len(calls) == 1 and calls[0]["tool"] == "Read" and calls[0]["input"].get("file_path") == expected
        ls = [l for l in (answer or "").splitlines() if l.strip()]
        try:
            complete = sorted(json.loads(l)["id"] for l in ls) == sorted(m["items"])
        except (json.JSONDecodeError, KeyError):
            complete = False
        audit.append({"rater": m["rater"], "prompt": m["prompt"], "models": models, "tool_calls": calls,
                      "only_read_own_prompt": clean, "answers_complete": complete})
        if clean and complete:
            (D / "outputs" / m["rater"]).mkdir(parents=True, exist_ok=True)
            (D / "outputs" / m["rater"] / f"{m['prompt']}.jsonl").write_text("\n".join(ls) + "\n")
        else:
            problems.append(f"bad: {desc} clean={clean} complete={complete}")
    (D / "outputs").mkdir(parents=True, exist_ok=True)
    (D / "outputs" / "audit.json").write_text(json.dumps(audit, indent=1))
    print(f"{len(audit)} of {len(man)} collected; problems: {len(problems)}")
    for p in problems:
        print(" ", p)


def boot_rate(g, col, rng, n=2000):
    """Share of warnings flagged, with a bootstrap over boards (warnings on the same board move together)."""
    by = {b: x[col].to_numpy() for b, x in g.groupby("board_id")}
    keys = list(by)
    vals = []
    for _ in range(n):
        pick = rng.choice(len(keys), len(keys), replace=True)
        v = np.concatenate([by[keys[i]] for i in pick])
        vals.append(v.mean())
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def score():
    d = items().set_index("item")
    man = json.loads((D / "prompts" / "manifest.json").read_text())
    got = {"A2": {}, "B2": {}}
    for m in man:
        for l in (D / "outputs" / m["rater"] / f"{m['prompt']}.jsonl").read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                got[m["rater"]][r["id"]] = r
    for r in ("A2", "B2"):
        d[f"ok_{r}"] = [int(bool(got[r][i]["claims_ok"])) for i in d.index]
        d[f"bad_{r}"] = [got[r][i].get("bad_claims", "") for i in d.index]
        d[f"close_{r}"] = [int(got[r][i]["close_on_date"]) for i in d.index]
    d["uns_both"] = ((d.ok_A2 == 0) & (d.ok_B2 == 0)).astype(int)
    d["uns_either"] = ((d.ok_A2 == 0) | (d.ok_B2 == 0)).astype(int)
    d["close_both"] = ((d.close_A2 == 1) & (d.close_B2 == 1)).astype(int)
    d["close_either"] = ((d.close_A2 == 1) | (d.close_B2 == 1)).astype(int)
    d.drop(columns=["board_text", "model_text", "shown"]).to_csv(OUT / "results" / "ratings_masked.csv")
    rng = np.random.default_rng(SEED)
    out = {"n": int(len(d)), "kappa_claims": kappa(d.ok_A2, d.ok_B2), "kappa_close": kappa(d.close_A2, d.close_B2),
           "disagree_claims": int((d.ok_A2 != d.ok_B2).sum()), "disagree_close": int((d.close_A2 != d.close_B2).sum())}
    for arm in ("L", "G"):
        g = d[d.arm == arm]
        e = {"n": int(len(g))}
        for col in ("uns_both", "uns_either", "close_both", "close_either"):
            lo, hi = boot_rate(g, col, rng)
            e[col] = int(g[col].sum())
            e[col + "_ci"] = [lo, hi]
        out[arm] = e
    # the original (unmasked) ratings, with the same board-level intervals, for the side-by-side table
    o = pd.read_csv(OUT / "results" / "ratings.csv")
    o = o[o.set == "study"]
    orig = {"kappa_claims": kappa(o.ok_A, o.ok_B), "disagree_claims": int((o.ok_A != o.ok_B).sum()),
            "disagree_close": int((o.close_A != o.close_B).sum())}
    for arm in ("L", "G"):
        g = o[o.arm == arm]
        e = {"n": int(len(g))}
        for col in ("unsupported_both", "unsupported_either", "close_both", "close_either"):
            lo, hi = boot_rate(g, col, rng)
            e[col] = int(g[col].sum())
            e[col + "_ci"] = [lo, hi]
        orig[arm] = e
    out["original"] = orig
    (OUT / "results" / "ratings_masked_summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "collect":
        collect(sys.argv[2])
    else:
        {"build": build, "score": score}[sys.argv[1]]()
