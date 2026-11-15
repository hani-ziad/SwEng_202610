"""Step 1 of the study: pick 120 sprints, rebuild their boards, score the model arms, write the prompts.

Design (fixed before any agent ran, 15 Nov 2026):
  Pool      the same 472 held-out sprints as the pilot: last 30% of each project's history, with
            committed work, still open at mid-sprint.
  Sample    15 sprints from each of the 8 projects (120), drawn at random within project with
            seed 2027. No balancing on the outcome, so the late rate is the projects' own.
            Results are reweighted to the pool by project (pool size / 15).
  Points    sprint start (c = 0) and halfway (c = 0.5).
  Targets   late closure (closed in Jira more than a day after the planned end), as in the pilot;
            and the share of committed issues still unresolved at the planned end. The second
            target is not scored for Atlassian Confluence Cloud, where most issues have no
            resolution date (known TAWOS defect), but its boards are still shown to the agents.
  Arms      M  Phase 2 gradient boosting at c = 0, Phase 3 gradient boosting at c = 0.5 (as in the
               pilot), plus a gradient-boosting regressor for the unfinished share, trained on the
               same training side without Confluence Cloud.
            L  agent, board only.          G  agent, board plus both model outputs.
  Runs      three independent runs per arm; board order is reshuffled for every run.

Outputs (in study/): results/replay_points.csv, results/boards.jsonl, results/build_info.json,
prompts/run{1,2,3}/*.txt and prompts/manifest.json.

Usage (from the ccece-2027-ai-scrum-master folder):  python -m study.build
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, roc_auc_score

import study
from pilot import build as pb
from phase2.data import STRICT_NUMERIC
from phase2.models import feature_columns, make_regressor
from phase3.data import CHECKPOINT_NUMERIC

SEED = 2027
PER_PROJECT = 15
RUNS = 3
BATCH = 20
CHECKPOINTS = (0.0, 0.5)
NO_RESOLUTION = "Atlassian Confluence Cloud"
OUT = study.HERE

INSTRUCTIONS = """You are acting as an AI assistant to a Scrum Master. For each sprint board below, decide
whether to warn the team NOW that the sprint is likely to close late, meaning it will be closed in Jira more
than one day after its planned end date. A warning interrupts the team, so do not warn without reason; a
missed late sprint is costly too. Also estimate how much of the committed work will still be unresolved
at the planned end date.

You see the sprint exactly as it stood at the replay point. Judge every board on its own; boards come from
different teams and are in random order. Do not use any tools, files or web access: answer from the text only.

Answer with one JSON object per board, one per line, and nothing else:
{"id": "<board id>", "warn": true or false, "p_late": <your probability from 0 to 1 that it closes late>, "unfinished": <your estimate from 0 to 1 of the share of the committed issues that will still be unresolved at the planned end date>, "message": "<if warn is true, the one or two sentences you would send the team; otherwise empty>"}
"""

GROUNDED_NOTE = """Each board also shows the output of two models trained on earlier sprints of these
teams. Use them as you see fit: you may follow them or overrule them."""


def fit_unfinished(train, test, numeric):
    est = make_regressor("gradient_boosting", numeric, True)
    cols = feature_columns(numeric, True)
    tr = train[train["project"] != NO_RESOLUTION]
    est.fit(tr[cols], tr["carry_ratio"].to_numpy(float))
    return np.clip(est.predict(test[cols]), 0, 1)


def unfinished_arm(base, te0, ck):
    tr0 = base[~base["sprint_id"].isin(te0["sprint_id"]) & base["nonempty"]]
    te0n = te0[te0["nonempty"]]
    u0 = pd.DataFrame({"sprint_id": te0n["sprint_id"].values, "checkpoint": 0.0,
                       "m_unf": fit_unfinished(tr0, te0n, STRICT_NUMERIC)})
    h = ck[(ck.checkpoint == 0.5) & ck.open_at_checkpoint & ck.nonempty]
    tr5, te5 = h[h.split_full == "train"], h[h.split_full == "test"]
    u5 = pd.DataFrame({"sprint_id": te5["sprint_id"].values, "checkpoint": 0.5,
                       "m_unf": fit_unfinished(tr5, te5, CHECKPOINT_NUMERIC)})
    u = pd.concat([u0, u5], ignore_index=True)
    # held-out error of the regressor on the scored projects, for the line shown to G
    lab = base.set_index("sprint_id")
    mae = {}
    for c, g in u.groupby("checkpoint"):
        g = g[g.sprint_id.map(lab["project"]) != NO_RESOLUTION]
        mae[c] = mean_absolute_error(g.sprint_id.map(lab["carry_ratio"]), g.m_unf)
    return u, mae


def model_lines(m, c, mae):
    call = "WARN" if m["m_warn"] else "SILENT"
    return (f"Risk model: probability of closing late {m['m_score']:.2f}; its alarm threshold is "
            f"{m['m_threshold']:.2f}, so the model says {call}. On held-out sprints this model ranked late sprints "
            f"above on-time ones {pb.MODEL_AUC[c] * 100:.0f}% of the time (AUC {pb.MODEL_AUC[c]:.2f}).\n"
            f"Delivery model: expected share of the committed issues still unresolved at the planned end "
            f"{m['m_unf']:.2f}. On held-out sprints its typical error was {mae[c]:.2f} (mean absolute error).")


def main():
    base, te0, ck, m = pb.model_arm()
    u, mae = unfinished_arm(base, te0, ck)
    m = m.merge(u, on=["sprint_id", "checkpoint"], how="left")
    mid = ck[ck.checkpoint == 0.5].set_index("sprint_id")
    elig = te0[te0["committed_issue_count"] > 0].copy()
    elig = elig[elig["sprint_id"].map(mid["open_at_checkpoint"]).fillna(False).astype(bool)]
    pool = elig.groupby("project").size()

    rng = np.random.default_rng(SEED)
    pick = []
    for p in sorted(pool.index):
        ids = sorted(elig.loc[elig.project == p, "sprint_id"])
        pick += list(rng.choice(ids, PER_PROJECT, replace=False))
    sample = base.set_index("sprint_id").loc[pick].reset_index()
    letters = {p: chr(65 + i) for i, p in enumerate(sorted(base.project.unique()))}
    codes = rng.permutation(len(sample))
    sample["board_id"] = [f"B{c + 1:03d}" for c in codes]

    start = {r.sprint_id: mid.loc[r.sprint_id, "start_date"].value for r in sample.itertuples()}
    end = {r.sprint_id: mid.loc[r.sprint_id, "end_date"].value for r in sample.itertuples()}
    boards = pb.issue_boards(sample["sprint_id"], start, end)
    for sid in sample["sprint_id"]:
        b, r = boards[sid], mid.loc[sid]
        assert len(b["committed"]) == r["committed_n"], sid
        assert sum(x["done_day_by_mid"] is not None for x in b["committed"]) == r["done_n"], sid
        assert len(b["added"]) == r["added_n"], sid

    mm = m.set_index(["sprint_id", "checkpoint"])
    recs, texts, board_rows = [], {}, []
    for r in sample.to_dict("records"):
        for c in CHECKPOINTS:
            mr = mm.loc[(r["sprint_id"], c)]
            assert not np.isnan(mr["m_unf"]), r["sprint_id"]
            t = pb.board_text(r, boards[r["sprint_id"]], c, letters[r["project"]])
            texts[(r["board_id"], c, "L")] = t
            texts[(r["board_id"], c, "G")] = t + "\n" + model_lines(mr, c, mae)
            board_rows.append({"board_id": r["board_id"], "checkpoint": c, "text": t,
                               "model_text": model_lines(mr, c, mae)})
            recs.append({"sprint_id": r["sprint_id"], "board_id": r["board_id"], "project": r["project"],
                         "project_letter": letters[r["project"]], "checkpoint": c,
                         "weight": pool[r["project"]] / PER_PROJECT,
                         "late": int(r["label_delay"]), "late_days": r["late_days"],
                         "unfinished": r["carry_ratio"], "unf_scored": int(r["project"] != NO_RESOLUTION),
                         "hist_late_rate_3": r["hist_late_rate_3"],
                         "hist_unf": (1 - r["hist_avg_completion_ratio_3"]
                                      if not pd.isna(r["hist_avg_completion_ratio_3"]) else np.nan),
                         "m_score": mr["m_score"], "m_threshold": mr["m_threshold"],
                         "m_warn": int(mr["m_warn"]), "m_unf": mr["m_unf"]})
    rp = pd.DataFrame(recs)
    (OUT / "results").mkdir(parents=True, exist_ok=True)
    rp.to_csv(OUT / "results" / "replay_points.csv", index=False)
    with open(OUT / "results" / "boards.jsonl", "w") as fh:
        for b in board_rows:
            fh.write(json.dumps(b) + "\n")

    manifest = []
    for run in range(1, RUNS + 1):
        rr = np.random.default_rng(SEED + run)
        d = OUT / "prompts" / f"run{run}"
        d.mkdir(parents=True, exist_ok=True)
        for arm in ("L", "G"):
            for c in CHECKPOINTS:
                order = list(rr.permutation(list(sample["board_id"])))
                for k in range(0, len(order), BATCH):
                    part = order[k:k + BATCH]
                    head = INSTRUCTIONS + ("\n" + GROUNDED_NOTE + "\n" if arm == "G" else "")
                    body = "\n\n".join(f"=== Board {i} ===\n{texts[(i, c, arm)]}" for i in part)
                    name = f"{arm}_c{int(c * 100):02d}_b{k // BATCH + 1}"
                    (d / f"{name}.txt").write_text(head + "\n" + body + "\n")
                    manifest.append({"run": run, "prompt": name, "arm": arm, "checkpoint": c, "boards": part})
    (OUT / "prompts" / "manifest.json").write_text(json.dumps(manifest, indent=1))

    s0 = rp[rp.checkpoint == 0.0]
    info = {"seed": SEED, "per_project": PER_PROJECT, "n_sprints": int(len(sample)),
            "n_late": int(s0.late.sum()), "pool": {k: int(v) for k, v in pool.items()},
            "pool_n": int(pool.sum()), "pool_late": int(elig.label_delay.sum()),
            "overlap_with_pilot": None,
            "unfinished_model_mae_heldout": {str(k): float(v) for k, v in mae.items()},
            "unfinished_scored_sprints": int(s0.unf_scored.sum()),
            "n_prompts": len(manifest)}
    try:
        pilot_ids = set(pd.read_csv(OUT.parent / "pilot" / "results" / "replay_points.csv").sprint_id)
        info["overlap_with_pilot"] = int(len(pilot_ids & set(sample.sprint_id)))
    except FileNotFoundError:
        pass
    for c in CHECKPOINTS:
        g = rp[rp.checkpoint == c]
        info[f"sample_auc_M_c{int(c * 100)}"] = float(roc_auc_score(g.late, g.m_score))
    (OUT / "results" / "build_info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
