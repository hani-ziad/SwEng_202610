"""Step 1 of the pilot: pick the sprints, rebuild their boards, score the model arm, write the prompts.

Replay points: 40 test sprints of the Phase 2 time-ordered split (20 that closed late and
20 that did not, drawn with seed 42 from the nonempty test sprints still open at mid-sprint),
each seen at two checkpoints: sprint start (c = 0) and halfway (c = 0.5).

Arms
  M  the risk model alone: Phase 2 gradient boosting (strict planning features) at c = 0,
     Phase 3 gradient boosting (strict + progress features) at c = 0.5. Both trained on the
     training side of the time split only; the warn threshold is Youden's J on 5-fold
     out-of-fold training scores (the Phase 4 rule), never tuned on test sprints.
  L  a language-model agent that sees the board only.
  G  the same agent that also sees M's probability and M's warn/silent call.

Every board uses only events time-stamped at or before the replay point. Project names,
issue keys and calendar dates are hidden (projects become letters, time becomes day numbers),
so the agent cannot look a sprint up from memory.

Usage (from the ccece-2027-ai-scrum-master folder):  python -m pilot.build
"""
from __future__ import annotations

import json
from bisect import bisect_right
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict

import pilot  # noqa: F401  (path setup)
import phase3
from phase2.data import STRICT_NUMERIC, load_phase2_frame, population, time_ordered_split
from phase2.models import feature_columns, make_classifier
from phase3.data import CHECKPOINT_NUMERIC
from src.data.load_tawos import load_change_log, load_issues, load_sprints
from src.data.planning_snapshot import (
    _field_kind,
    _parse_change_sprint_set,
    _project_maps,
    _story_after,
    _story_before,
)

SEED = 42
N_PER_CLASS = 20
CHECKPOINTS = (0.0, 0.5)
BATCH = 20
OUT = pilot.HERE
MODEL_AUC = {0.0: 0.751, 0.5: 0.760}   # test AUCs of these two models reported in Phases 2 and 3


def youden(y, s):
    best, thr = -1.0, 0.5
    for t in np.unique(s):
        p = s >= t
        j = (p & (y == 1)).sum() / max((y == 1).sum(), 1) + (~p & (y == 0)).sum() / max((y == 0).sum(), 1) - 1
        if j > best:
            best, thr = j, t
    return float(thr)


def fit_m(train, test, numeric):
    est = make_classifier("gradient_boosting", numeric, True)
    cols = feature_columns(numeric, True)
    y = train["label_delay"].to_numpy(int)
    oof = cross_val_predict(est, train[cols], y, cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                            method="predict_proba")[:, 1]
    thr = youden(y, oof)
    est.fit(train[cols], y)
    return est.predict_proba(test[cols])[:, 1], thr


def model_arm():
    base = population(load_phase2_frame(), "full")
    tr0, te0 = time_ordered_split(base)
    s0, t0 = fit_m(tr0, te0, STRICT_NUMERIC)
    m0 = pd.DataFrame({"sprint_id": te0["sprint_id"].values, "checkpoint": 0.0, "m_score": s0, "m_threshold": t0})

    ck = pd.read_csv(phase3.RESULTS_DIR / "phase3_checkpoint_frame.csv",
                     parse_dates=["start_date", "end_date", "complete_date", "checkpoint_time"])
    h = ck[(ck.checkpoint == 0.5) & ck.open_at_checkpoint]
    tr5, te5 = h[h.split_full == "train"], h[h.split_full == "test"]
    s5, t5 = fit_m(tr5, te5, CHECKPOINT_NUMERIC)
    m5 = pd.DataFrame({"sprint_id": te5["sprint_id"].values, "checkpoint": 0.5, "m_score": s5, "m_threshold": t5})
    m = pd.concat([m0, m5], ignore_index=True)
    m["m_warn"] = (m["m_score"] >= m["m_threshold"]).astype(int)
    return base, te0, ck, m


def issue_boards(sprint_ids, start, end):
    """Per sprint: committed issues at start with their state at mid-sprint, and issues added by then.
    Same membership / story-point rules as Phase 3 (reverse every change after the query time)."""
    sprints = load_sprints(phase3.RAW_DIR, modeling_only=True)
    issues = load_issues(phase3.RAW_DIR)
    cl = load_change_log(phase3.RAW_DIR, required=True)
    jira_maps, raw_maps, name_maps = _project_maps(sprints)
    rel = cl[cl["field"].map(_field_kind).notna()].dropna(subset=["raw_issue_id", "change_date"])
    rel = rel.sort_values(["raw_issue_id", "change_date", "change_id"])
    events = defaultdict(list)
    for r in rel.itertuples(index=False):
        events[r.raw_issue_id].append(r)
    wanted = set(sprint_ids)
    out = {sid: {"committed": [], "added": []} for sid in wanted}
    for issue in issues.itertuples(index=False):
        p = issue.project
        if p not in jira_maps:
            continue
        ev = events.get(issue.raw_issue_id, [])
        jm, rm, nm = jira_maps[p], raw_maps[p], name_maps[p]
        sp_ev = [e for e in ev if _field_kind(e.field) == "sprint"]
        st_ev = [e for e in ev if _field_kind(e.field) == "story_point"]
        sp_times = [e.change_date.value for e in sp_ev]
        sp_from = [_parse_change_sprint_set(e.from_value, e.from_string, jira_map=jm, name_map=nm)[0] for e in sp_ev]
        if sp_ev:
            terminal = _parse_change_sprint_set(sp_ev[-1].to_value, sp_ev[-1].to_string, jira_map=jm, name_map=nm)[0]
            cand = set(terminal)
            for e in sp_ev:
                for val, sv in ((e.from_value, e.from_string), (e.to_value, e.to_string)):
                    cand |= _parse_change_sprint_set(val, sv, jira_map=jm, name_map=nm)[0]
        else:
            terminal = set()
            if not pd.isna(issue.raw_sprint_id):
                k = str(int(float(issue.raw_sprint_id)))
                if k in rm:
                    terminal = {rm[k]}
            cand = set(terminal)
        cand &= wanted
        if not cand:
            continue
        st_times = [e.change_date.value for e in st_ev]
        static = None if pd.isna(issue.story_point) else float(issue.story_point)
        st_term = _story_after(st_ev[-1]) if st_ev else static

        def members(t):
            i = bisect_right(sp_times, t)
            return sp_from[i] if i < len(sp_times) else terminal

        def story(t):
            i = bisect_right(st_times, t)
            return _story_before(st_ev[i]) if i < len(st_times) else st_term

        created = None if pd.isna(issue.creation_date) else issue.creation_date.value
        resolved = None if pd.isna(issue.resolution_date) else issue.resolution_date.value
        for sid in cand:
            t0 = start[sid]
            tk = t0 + 0.5 * (end[sid] - t0)
            in0 = sid in members(t0) and (created is None or created <= t0) and not (resolved is not None and resolved < t0)
            ink = sid in members(tk) and (created is None or created <= tk)
            done_day = (resolved - t0) / 86400e9 if resolved is not None and resolved <= tk else None
            row = {"type": issue.issue_type, "sp_start": story(t0) if in0 else None, "sp_mid": story(tk),
                   "done_day_by_mid": done_day, "in_at_mid": bool(ink),
                   "created_day": None if created is None else (created - t0) / 86400e9}
            if in0:
                out[sid]["committed"].append(row)
            elif ink and not (resolved is not None and resolved < t0):
                out[sid]["added"].append(row)
    return out


def fmt_sp(v):
    return "no estimate" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:g} SP"


def board_text(r, b, c, letter):
    L = r["sprint_length_days"]
    lines = [f"Team: Project {letter} (name hidden). Planned sprint length: {L:.1f} days.",
             f"Other sprints of this project open at the same time: {int(r['n_concurrent_sprints'])}."]
    if r["hist_n_available"] > 0:
        lines.append(
            "Team history (finished sprints before this one started, up to the last three): "
            f"{int(r['hist_n_available'])} available; average share of committed issues completed "
            f"{r['hist_avg_completion_ratio_3']:.2f}; issues carried over from the last one {r['hist_last_carryover']:g}; "
            f"share that closed more than a day late {r['hist_late_rate_3']:.2f}; average days late "
            f"{r['hist_avg_late_days_3']:.1f}; average issues completed {r['hist_avg_completed_issues_3']:.1f}.")
    else:
        lines.append("Team history: no finished earlier sprint was available when this sprint started.")
    com = b["committed"]
    sp_known = [x["sp_start"] for x in com if x["sp_start"] is not None]
    lines.append(f"Committed at the start: {len(com)} issues; story points recorded on {len(sp_known)} of them "
                 f"(total {sum(sp_known):g}).")
    if c == 0.0:
        lines.append("Now: day 0, planning has just finished.")
        rows = [f"- {x['type']}, {fmt_sp(x['sp_start'])}" for x in com]
    else:
        done = [x for x in com if x["done_day_by_mid"] is not None]
        out_now = [x for x in com if not x["in_at_mid"]]
        add = b["added"]
        lines.append(f"Now: day {0.5 * L:.1f} of {L:.1f} (halfway). Committed issues resolved so far: {len(done)} of "
                     f"{len(com)}. Committed issues moved out of the sprint: {len(out_now)}. Issues added since the "
                     f"start: {len(add)}, of which resolved: {sum(x['done_day_by_mid'] is not None for x in add)}.")
        rows = []
        for x in com:
            st = (f"resolved on day {x['done_day_by_mid']:.1f}" if x["done_day_by_mid"] is not None
                  else ("moved out of the sprint" if not x["in_at_mid"] else "not resolved yet"))
            rows.append(f"- committed: {x['type']}, {fmt_sp(x['sp_mid'])}, {st}")
        for x in add:
            st = f"resolved on day {x['done_day_by_mid']:.1f}" if x["done_day_by_mid"] is not None else "not resolved yet"
            rows.append(f"- added later: {x['type']}, {fmt_sp(x['sp_mid'])}, {st}")
    if len(rows) > 40:
        rows = rows[:40] + [f"- ... and {len(rows) - 40} more issues of the same kinds"]
    return "\n".join(lines + ["Issues:"] + rows)


INSTRUCTIONS = """You are acting as an AI assistant to a Scrum Master. For each sprint board below, decide
whether to warn the team NOW that the sprint is likely to close late, meaning it will be closed in Jira more
than one day after its planned end date. A warning interrupts the team, so do not warn without reason; a
missed late sprint is costly too.

You see the sprint exactly as it stood at the replay point. Judge every board on its own; boards come from
different teams and are in random order. Do not use any tools, files or web access: answer from the text only.

Answer with one JSON object per board, one per line, and nothing else:
{"id": "<board id>", "warn": true or false, "p_late": <your probability from 0 to 1 that it closes late>, "message": "<if warn is true, the one or two sentences you would send the team; otherwise empty>"}
"""

GROUNDED_NOTE = """Each board also shows the output of a risk model trained on earlier sprints of these
teams. Use it as you see fit: you may follow it or overrule it."""


def model_lines(m, c):
    call = "WARN" if m["m_warn"] else "SILENT"
    return (f"Risk model: probability of closing late {m['m_score']:.2f}; its alarm threshold is "
            f"{m['m_threshold']:.2f}, so the model says {call}. On held-out sprints this model ranked late sprints "
            f"above on-time ones {MODEL_AUC[c] * 100:.0f}% of the time (AUC {MODEL_AUC[c]:.2f}).")


def main():
    base, te0, ck, m = model_arm()
    mid = ck[ck.checkpoint == 0.5].set_index("sprint_id")
    elig = te0[te0["committed_issue_count"] > 0].copy()
    elig = elig[elig["sprint_id"].map(mid["open_at_checkpoint"]).fillna(False).astype(bool)]
    rng = np.random.default_rng(SEED)
    pick = []
    for lab in (1, 0):
        ids = sorted(elig.loc[elig.label_delay == lab, "sprint_id"])
        pick += list(rng.choice(ids, N_PER_CLASS, replace=False))
    sample = base.set_index("sprint_id").loc[pick].reset_index()
    letters = {p: chr(65 + i) for i, p in enumerate(sorted(base.project.unique()))}
    codes = rng.permutation(len(sample))
    sample["board_id"] = [f"S{c + 1:02d}" for c in codes]

    start = {r.sprint_id: mid.loc[r.sprint_id, "start_date"].value for r in sample.itertuples()}
    end = {r.sprint_id: mid.loc[r.sprint_id, "end_date"].value for r in sample.itertuples()}
    boards = issue_boards(sample["sprint_id"], start, end)

    # check the rebuilt boards against Phase 3's aggregates
    for sid in sample["sprint_id"]:
        b, r = boards[sid], mid.loc[sid]
        assert len(b["committed"]) == r["committed_n"], (sid, len(b["committed"]), r["committed_n"])
        assert sum(x["done_day_by_mid"] is not None for x in b["committed"]) == r["done_n"], sid
        assert len(b["added"]) == r["added_n"], sid

    mm = m.set_index(["sprint_id", "checkpoint"])
    recs, texts = [], {}
    for r in sample.to_dict("records"):
        for c in CHECKPOINTS:
            mr = mm.loc[(r["sprint_id"], c)]
            t = board_text(r, boards[r["sprint_id"]], c, letters[r["project"]])
            texts[(r["board_id"], c, "L")] = t
            texts[(r["board_id"], c, "G")] = t + "\n" + model_lines(mr, c)
            recs.append({"sprint_id": r["sprint_id"], "board_id": r["board_id"], "project": r["project"],
                         "project_letter": letters[r["project"]], "checkpoint": c, "late": int(r["label_delay"]),
                         "late_days": r["late_days"], "m_score": mr["m_score"], "m_threshold": mr["m_threshold"],
                         "m_warn": int(mr["m_warn"])})
    rp = pd.DataFrame(recs)
    rp.to_csv(OUT / "results" / "replay_points.csv", index=False)

    manifest = []
    for arm in ("L", "G"):
        for c in CHECKPOINTS:
            ids = list(sample["board_id"])
            order = list(rng.permutation(ids))
            for k in range(0, len(order), BATCH):
                part = order[k:k + BATCH]
                head = INSTRUCTIONS + ("\n" + GROUNDED_NOTE + "\n" if arm == "G" else "")
                body = "\n\n".join(f"=== Board {i} ===\n{texts[(i, c, arm)]}" for i in part)
                name = f"{arm}_c{int(c * 100):02d}_b{k // BATCH + 1}"
                (OUT / "prompts" / f"{name}.txt").write_text(head + "\n" + body + "\n")
                manifest.append({"prompt": name, "arm": arm, "checkpoint": c, "boards": part})
    (OUT / "prompts" / "manifest.json").write_text(json.dumps(manifest, indent=1))

    # sanity: the two model arms on the whole test split, as in Phases 2 and 3
    from sklearn.metrics import roc_auc_score
    y0 = te0.set_index("sprint_id").loc[m[m.checkpoint == 0.0].sprint_id, "label_delay"]
    y5 = mid.loc[m[m.checkpoint == 0.5].sprint_id, "label_delay"]
    info = {"test_auc_c0": roc_auc_score(y0, m[m.checkpoint == 0.0].m_score),
            "test_auc_c50": roc_auc_score(y5, m[m.checkpoint == 0.5].m_score),
            "n_test_c0": int(len(y0)), "n_test_c50": int(len(y5)),
            "threshold_c0": float(m[m.checkpoint == 0.0].m_threshold.iloc[0]),
            "threshold_c50": float(m[m.checkpoint == 0.5].m_threshold.iloc[0]),
            "eligible_test_sprints": int(len(elig)), "eligible_late": int(elig.label_delay.sum()),
            "sample_projects": sample.project.value_counts().to_dict()}
    (OUT / "results" / "build_info.json").write_text(json.dumps(info, indent=1, default=str))
    print(json.dumps(info, indent=1, default=str))


if __name__ == "__main__":
    main()
