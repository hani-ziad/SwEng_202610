"""Model-access ablation (added 15 Nov 2026 after the external review).

The main replay compares the agent with the model's output hidden (L) and shown (G). A reviewer
asked whether G's agreement comes from the model's call itself. This script holds the boards, the
instructions and the agent fixed and changes only the risk-model line:

  P  probability only: the model's probability and its held-out AUC, but no threshold and no
     warn/silent call.
  F  flipped call: the probability is mirrored around the model's threshold, p' = 2t - p (clipped
     to 0.01..0.99), so the stated call is the opposite of the real one; threshold and AUC as in G.

The delivery-model line is the same as in G. One run per arm, 24 prompt files (2 arms x 2
checkpoints x 6 batches of 20), boards in a fresh random order.

Usage (from the ccece-2027-ai-scrum-master folder):
  python -m study.ablation build              # writes ablation/prompts/*.txt and manifest.json
  python -m study.ablation collect <folder>   # saves answers, audits tool calls (as study.collect)
  python -m study.ablation score              # writes results/ablation_summary.json
  python -m study.ablation intervals          # adds bootstrap intervals to it (about 2 minutes)
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import study
from study.build import BATCH, CHECKPOINTS, GROUNDED_NOTE, INSTRUCTIONS
from study.collect import read_transcript

HERE = study.HERE
AB = HERE / "ablation"
SEED = 2027 + 50
RUNS = (1, 2, 3)


def mirror(p, t):
    return float(np.clip(2 * t - p, 0.01, 0.99))


def risk_line(model_text: str, arm: str, p: float, t: float) -> str:
    risk, delivery = model_text.split("\n", 1)
    auc = re.search(r"On held-out sprints.*$", risk).group(0)
    if arm == "P":
        risk = f"Risk model: probability of closing late {p:.2f}. {auc}"
    else:
        q = mirror(p, t)
        call = "WARN" if q >= t else "SILENT"
        risk = (f"Risk model: probability of closing late {q:.2f}; its alarm threshold is {t:.2f}, "
                f"so the model says {call}. {auc}")
    return risk + "\n" + delivery


def build():
    rp = pd.read_csv(HERE / "results" / "replay_points.csv").set_index(["board_id", "checkpoint"])
    boards = {(b["board_id"], b["checkpoint"]): b for b in map(json.loads, open(HERE / "results" / "boards.jsonl"))}
    ids = sorted(rp.index.get_level_values(0).unique())
    rng = np.random.default_rng(SEED)
    man = []
    (AB / "prompts").mkdir(parents=True, exist_ok=True)
    for arm in ("P", "F"):
        for c in CHECKPOINTS:
            order = list(rng.permutation(ids))
            for k in range(0, len(order), BATCH):
                part = order[k:k + BATCH]
                body = []
                for i in part:
                    b, r = boards[(i, c)], rp.loc[(i, c)]
                    body.append(f"=== Board {i} ===\n{b['text']}\n"
                                + risk_line(b["model_text"], arm, r.m_score, r.m_threshold))
                name = f"{arm}_c{int(c * 100):02d}_b{k // BATCH + 1}"
                (AB / "prompts" / f"{name}.txt").write_text(
                    INSTRUCTIONS + "\n" + GROUNDED_NOTE + "\n\n" + "\n\n".join(body) + "\n")
                man.append({"prompt": name, "arm": arm, "checkpoint": c, "boards": part})
    (AB / "prompts" / "manifest.json").write_text(json.dumps(man, indent=1))
    print(len(man), "prompt files")


def collect(folder):
    man = {f"ablation {m['prompt']}": m for m in json.loads((AB / "prompts" / "manifest.json").read_text())}
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
        expected = f"/home/claude/agentbox/ablation/{m['prompt']}.txt"
        clean = len(calls) == 1 and calls[0]["tool"] == "Read" and calls[0]["input"].get("file_path") == expected
        lines = [l for l in (answer or "").splitlines() if l.strip()]
        try:
            complete = sorted(json.loads(l)["id"] for l in lines) == sorted(m["boards"])
        except (json.JSONDecodeError, KeyError):
            complete = False
        audit.append({"prompt": m["prompt"], "models": models, "tool_calls": calls,
                      "only_read_own_prompt": clean, "answers_complete": complete})
        if clean and complete:
            (AB / "outputs").mkdir(parents=True, exist_ok=True)
            (AB / "outputs" / f"{m['prompt']}.jsonl").write_text("\n".join(lines) + "\n")
        else:
            problems.append(f"bad: {desc} clean={clean} complete={complete}")
    (AB / "outputs").mkdir(parents=True, exist_ok=True)
    (AB / "outputs" / "audit.json").write_text(json.dumps(audit, indent=1))
    print(f"{len(audit)} of {len(man)} collected; problems: {len(problems)}")
    for p in problems:
        print(" ", p)


def kappa(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    po, pe = (a == b).mean(), a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def within_auc(g, col):
    v = [roc_auc_score(x.late, x[col]) for _, x in g.groupby("project") if x.late.nunique() == 2]
    return float(np.mean(v))


def score():
    d = pd.read_csv(HERE / "results" / "replay_all_arms.csv")
    man = json.loads((AB / "prompts" / "manifest.json").read_text())
    ans = {}
    for m in man:
        for l in (AB / "outputs" / f"{m['prompt']}.jsonl").read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                ans[(m["arm"], r["id"], m["checkpoint"])] = r
    for arm in ("P", "F"):
        d[f"{arm}_warn"] = [int(bool(ans[(arm, b, c)]["warn"])) for b, c in zip(d.board_id, d.checkpoint)]
        d[f"{arm}_p"] = [float(ans[(arm, b, c)]["p_late"]) for b, c in zip(d.board_id, d.checkpoint)]
    d["F_shown_warn"] = [int(mirror(p, t) >= t) for p, t in zip(d.m_score, d.m_threshold)]
    d.to_csv(AB / "ablation_points.csv", index=False)

    out = {"n_points": int(len(d))}
    for c, g in d.groupby("checkpoint"):
        e = {}
        for arm in ("P", "F"):
            e[arm] = {"warn_rate": float(g[f"{arm}_warn"].mean()),
                      "agree_true_M": float((g[f"{arm}_warn"] == g.m_warn).mean()),
                      "kappa_true_M": kappa(g[f"{arm}_warn"], g.m_warn),
                      "auc": float(roc_auc_score(g.late, g[f"{arm}_p"])),
                      "auc_within_project": within_auc(g, f"{arm}_p"),
                      "sensitivity": float(g.loc[g.late == 1, f"{arm}_warn"].mean()),
                      "specificity": float(1 - g.loc[g.late == 0, f"{arm}_warn"].mean())}
        e["F"]["agree_shown_call"] = float((g.F_warn == g.F_shown_warn).mean())
        e["F"]["follows_flip_when_M_warns"] = float((g.loc[g.m_warn == 1, "F_warn"] == 0).mean())
        e["F"]["follows_flip_when_M_silent"] = float((g.loc[g.m_warn == 0, "F_warn"] == 1).mean())
        e["G_mean_runs"] = {"warn_rate": float(np.mean([g[f"G{r}_warn"].mean() for r in RUNS])),
                            "agree_true_M": float(np.mean([(g[f"G{r}_warn"] == g.m_warn).mean() for r in RUNS])),
                            "auc": float(np.mean([roc_auc_score(g.late, g[f"G{r}_p"]) for r in RUNS]))}
        e["L_mean_runs"] = {"warn_rate": float(np.mean([g[f"L{r}_warn"].mean() for r in RUNS])),
                            "agree_true_M": float(np.mean([(g[f"L{r}_warn"] == g.m_warn).mean() for r in RUNS]))}
        e["M"] = {"warn_rate": float(g.m_warn.mean()), "auc": float(roc_auc_score(g.late, g.m_score))}
        out[f"c{int(c * 100)}"] = e
    # pooled over both checkpoints
    out["pooled"] = {"P_agree_true_M": float((d.P_warn == d.m_warn).mean()),
                     "F_agree_shown_call": float((d.F_warn == d.F_shown_warn).mean()),
                     "F_agree_true_M": float((d.F_warn == d.m_warn).mean()),
                     "G_agree_true_M": float(np.mean([(d[f"G{r}_warn"] == d.m_warn).mean() for r in RUNS])),
                     "L_agree_true_M": float(np.mean([(d[f"L{r}_warn"] == d.m_warn).mean() for r in RUNS])),
                     "F_follow_n": int((d.F_warn == d.F_shown_warn).sum())}
    (HERE / "results" / "ablation_summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))



def intervals(n_boot=2000):
    """Bootstrap (sprints resampled within project) for the contrasts quoted in the paper."""
    d = pd.read_csv(AB / "ablation_points.csv")
    d["L_agree_shown"] = np.mean([(d[f"L{r}_warn"] == d.F_shown_warn) for r in RUNS], axis=0)
    d["G_agree_true"] = np.mean([(d[f"G{r}_warn"] == d.m_warn) for r in RUNS], axis=0)
    d["L_p"] = d[[f"L{r}_p" for r in RUNS]].mean(axis=1)
    sp = d.drop_duplicates("sprint_id")[["sprint_id", "project"]]
    groups = {p: g.sprint_id.to_numpy() for p, g in sp.groupby("project")}
    by = {s: g for s, g in d.groupby("sprint_id")}
    rng = np.random.default_rng(SEED)

    def stats(x):
        c0 = x[x.checkpoint == 0.0]
        return {"F_follow_minus_L": float((x.F_warn == x.F_shown_warn).mean() - x.L_agree_shown.mean()),
                "P_minus_G_agree": float((x.P_warn == x.m_warn).mean() - x.G_agree_true.mean()),
                "auc_F_minus_L_c0": float(roc_auc_score(c0.late, c0.F_p) - roc_auc_score(c0.late, c0.L_p))}
    point = stats(d)
    bs = []
    for _ in range(n_boot):
        ids = np.concatenate([rng.choice(v, len(v), replace=True) for v in groups.values()])
        bs.append(stats(pd.concat([by[i] for i in ids], ignore_index=True)))
    out = {k: {"point": v, "lo": float(np.percentile([b[k] for b in bs], 2.5)),
               "hi": float(np.percentile([b[k] for b in bs], 97.5))} for k, v in point.items()}
    out["L_agree_shown_call"] = float(d.L_agree_shown.mean())
    s = json.loads((HERE / "results" / "ablation_summary.json").read_text())
    s["intervals"] = out
    (HERE / "results" / "ablation_summary.json").write_text(json.dumps(s, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "collect":
        collect(sys.argv[2])
    else:
        {"build": build, "score": score, "intervals": intervals}[sys.argv[1]]()
