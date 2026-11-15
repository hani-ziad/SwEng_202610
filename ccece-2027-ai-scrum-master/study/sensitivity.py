"""Sensitivity runs (added 15 Nov 2026, night, after the second external review). One run each,
same boards and instructions as G; only the model lines change.

  T   training-only figures: the model lines quote the risk model's AUC and the delivery model's error
      from 5-fold out-of-fold predictions on the training sprints (study/train_side.py), not from the
      held-out sprints that include the replay sample.
  C   verdict-only flip: the real probability and threshold are shown, but the stated verdict is the
      opposite of what they imply (an internally inconsistent signal, by design). Separates copying
      the verbal call from anchoring on the number.
  F2  a second run of the flipped condition of study/ablation.py (probability mirrored around the
      threshold), in a fresh board order.

Usage (from the ccece-2027-ai-scrum-master folder):
  python -m study.sensitivity build
  python -m study.sensitivity collect <transcript folder>
  python -m study.sensitivity score
Writes study/sensitivity/{prompts,outputs}, study/sensitivity/points.csv, results/sensitivity_summary.json.
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
from study.ablation import mirror, kappa, within_auc
from study.build import BATCH, CHECKPOINTS, GROUNDED_NOTE, INSTRUCTIONS
from study.collect import read_transcript

HERE = study.HERE
SD = HERE / "sensitivity"
SEED = 2027 + 70
RUNS = (1, 2, 3)
ARMS = ("T", "C", "F2")


def lines(model_text, arm, p, t):
    risk, delivery = model_text.split("\n", 1)
    if arm == "C":
        call = "SILENT" if p >= t else "WARN"
        auc = re.search(r"On held-out sprints.*$", risk).group(0)
        return (f"Risk model: probability of closing late {p:.2f}; its alarm threshold is {t:.2f}, "
                f"so the model says {call}. {auc}\n{delivery}")
    q = mirror(p, t)
    call = "WARN" if q >= t else "SILENT"
    auc = re.search(r"On held-out sprints.*$", risk).group(0)
    return (f"Risk model: probability of closing late {q:.2f}; its alarm threshold is {t:.2f}, "
            f"so the model says {call}. {auc}\n{delivery}")


def train_lines(model_text, c, train):
    auc = train["train_oof_auc"][str(c)]
    mae = train["train_oof_mae_unfinished"][str(c)]
    risk, delivery = model_text.split("\n", 1)
    risk = re.sub(r"On held-out sprints.*$",
                  f"On its training sprints, scored by cross-validation, this model ranked late sprints above "
                  f"on-time ones {auc * 100:.0f}% of the time (AUC {auc:.2f}).", risk)
    delivery = re.sub(r"On held-out sprints its typical error was [\d.]+",
                      f"On its training sprints, scored by cross-validation, its typical error was {mae:.2f}",
                      delivery)
    return risk + "\n" + delivery


def build():
    rp = pd.read_csv(HERE / "results" / "replay_points.csv").set_index(["board_id", "checkpoint"])
    boards = {(b["board_id"], b["checkpoint"]): b for b in map(json.loads, open(HERE / "results" / "boards.jsonl"))}
    train = json.loads((HERE / "results" / "train_side.json").read_text())
    ids = sorted(rp.index.get_level_values(0).unique())
    rng = np.random.default_rng(SEED)
    man = []
    (SD / "prompts").mkdir(parents=True, exist_ok=True)
    for arm in ARMS:
        for c in CHECKPOINTS:
            order = list(rng.permutation(ids))
            for k in range(0, len(order), BATCH):
                part = order[k:k + BATCH]
                body = []
                for i in part:
                    b, r = boards[(i, c)], rp.loc[(i, c)]
                    ml = (train_lines(b["model_text"], c, train) if arm == "T"
                          else lines(b["model_text"], arm, r.m_score, r.m_threshold))
                    assert ml and ml != b["model_text"], (arm, i, c)
                    body.append(f"=== Board {i} ===\n{b['text']}\n{ml}")
                name = f"{arm}_c{int(c * 100):02d}_b{k // BATCH + 1}"
                (SD / "prompts" / f"{name}.txt").write_text(
                    INSTRUCTIONS + "\n" + GROUNDED_NOTE + "\n\n" + "\n\n".join(body) + "\n")
                man.append({"prompt": name, "arm": arm, "checkpoint": c, "boards": part})
    (SD / "prompts" / "manifest.json").write_text(json.dumps(man, indent=1))
    print(len(man), "prompt files")


def collect(folder):
    man = {f"sens {m['prompt']}": m for m in json.loads((SD / "prompts" / "manifest.json").read_text())}
    found, audit, problems = {}, [], []
    for meta in sorted(Path(folder).glob("agent-*.meta.json")):
        desc = json.loads(meta.read_text()).get("description", "")
        if desc in man:
            tr = meta.with_name(meta.name.replace(".meta.json", ".jsonl"))
            found.setdefault(desc, []).append((tr.stat().st_mtime, *read_transcript(tr)))
    (SD / "outputs").mkdir(parents=True, exist_ok=True)
    for desc, m in man.items():
        if desc not in found:
            problems.append(f"missing: {desc}")
            continue
        _, calls, models, answer = sorted(found[desc])[-1]
        expected = f"/home/claude/agentbox/sens/{m['prompt']}.txt"
        clean = len(calls) == 1 and calls[0]["tool"] == "Read" and calls[0]["input"].get("file_path") == expected
        ls = [l for l in (answer or "").splitlines() if l.strip()]
        try:
            complete = sorted(json.loads(l)["id"] for l in ls) == sorted(m["boards"])
        except (json.JSONDecodeError, KeyError):
            complete = False
        audit.append({"prompt": m["prompt"], "models": models, "tool_calls": calls,
                      "only_read_own_prompt": clean, "answers_complete": complete})
        if clean and complete:
            (SD / "outputs" / f"{m['prompt']}.jsonl").write_text("\n".join(ls) + "\n")
        else:
            problems.append(f"bad: {desc} clean={clean} complete={complete}")
    (SD / "outputs" / "audit.json").write_text(json.dumps(audit, indent=1))
    print(f"{len(audit)} of {len(man)} collected; problems: {len(problems)}")
    for p in problems:
        print(" ", p)


def score(n_boot=2000):
    d = pd.read_csv(HERE / "ablation" / "ablation_points.csv")
    man = json.loads((SD / "prompts" / "manifest.json").read_text())
    ans = {}
    for m in man:
        for l in (SD / "outputs" / f"{m['prompt']}.jsonl").read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                ans[(m["arm"], r["id"], m["checkpoint"])] = r
    for arm in ARMS:
        d[f"{arm}_warn"] = [int(bool(ans[(arm, b, c)]["warn"])) for b, c in zip(d.board_id, d.checkpoint)]
        d[f"{arm}_p"] = [float(ans[(arm, b, c)]["p_late"]) for b, c in zip(d.board_id, d.checkpoint)]
    d["C_shown_warn"] = 1 - d.m_warn
    d.to_csv(SD / "points.csv", index=False)
    G_agree = np.mean([(d[f"G{r}_warn"] == d.m_warn).mean() for r in RUNS])
    L_shown = np.mean([(d[f"L{r}_warn"] == d.F_shown_warn).mean() for r in RUNS])
    out = {"n_points": int(len(d)), "G_agree_true_M": float(G_agree), "L_agree_flipped_call": float(L_shown)}
    for arm, shown in (("T", "m_warn"), ("C", "C_shown_warn"), ("F2", "F_shown_warn")):
        e = {"agree_true_M": float((d[f"{arm}_warn"] == d.m_warn).mean()),
             "agree_shown_call": float((d[f"{arm}_warn"] == d[shown]).mean()),
             "follow_n": int((d[f"{arm}_warn"] == d[shown]).sum()),
             "kappa_true_M": kappa(d[f"{arm}_warn"], d.m_warn),
             "warn_rate": float(d[f"{arm}_warn"].mean())}
        for c, g in d.groupby("checkpoint"):
            e[f"auc_c{int(c * 100)}"] = float(roc_auc_score(g.late, g[f"{arm}_p"]))
            e[f"auc_within_c{int(c * 100)}"] = within_auc(g, f"{arm}_p")
        out[arm] = e
    # F and F2 together: how stable is the flipped result
    out["F_both_runs_follow"] = int(((d.F_warn == d.F_shown_warn) & (d.F2_warn == d.F_shown_warn)).sum())
    out["F_F2_same_call"] = float((d.F_warn == d.F2_warn).mean())
    # bootstrap within project for the contrasts quoted in the paper
    d["L_p"] = d[[f"L{r}_p" for r in RUNS]].mean(axis=1)
    d["G_agree"] = np.mean([(d[f"G{r}_warn"] == d.m_warn) for r in RUNS], axis=0)
    d["L_shown"] = np.mean([(d[f"L{r}_warn"] == d.F_shown_warn) for r in RUNS], axis=0)
    d["L_agree_C"] = np.mean([(d[f"L{r}_warn"] == d.C_shown_warn) for r in RUNS], axis=0)
    sp =d.drop_duplicates("sprint_id")[["sprint_id", "project"]]
    groups = {p: g.sprint_id.to_numpy() for p, g in sp.groupby("project")}
    by = {s: g for s, g in d.groupby("sprint_id")}
    rng = np.random.default_rng(SEED)

    def stats(x):
        return {"T_minus_G_agree": float((x.T_warn == x.m_warn).mean() - x.G_agree.mean()),
                "C_follow_minus_L": float((x.C_warn == x.C_shown_warn).mean() - x.L_agree_C.mean()),
                "F2_follow_minus_L": float((x.F2_warn == x.F_shown_warn).mean() - x.L_shown.mean())}
    d["L_agree_C"] = np.mean([(d[f"L{r}_warn"] == d.C_shown_warn) for r in RUNS], axis=0)
    point = stats(d)
    bs = []
    for _ in range(n_boot):
        ids = np.concatenate([rng.choice(v, len(v), replace=True) for v in groups.values()])
        bs.append(stats(pd.concat([by[i] for i in ids], ignore_index=True)))
    out["intervals"] = {k: {"point": v, "lo": float(np.percentile([b[k] for b in bs], 2.5)),
                            "hi": float(np.percentile([b[k] for b in bs], 97.5))} for k, v in point.items()}
    out["L_agree_C_call"] = float(d.L_agree_C.mean())
    (HERE / "results" / "sensitivity_summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "collect":
        collect(sys.argv[2])
    else:
        {"build": build, "score": score}[sys.argv[1]]()
