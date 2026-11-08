# Grounding the AI Scrum Master (CCECE 2027)

Position paper for IEEE CCECE 2027, with a small pilot of the sprint-replay test it proposes. This is a separate paper from `ccece-2027-paper` (the sprint-risk paper): different question, different results. It started as the Canadian AI 2027 short paper in `canadian-ai-2027-paper`, now in IEEE format and with the pilot added.

| File | What it is |
|---|---|
| `CCECE_2027_Grounding_AI_Scrum_Master.tex` | The paper (IEEEtran). Same text as the Overleaf project "Grounding the AI Scrum Master - CCECE 2027". |
| `CCECE_2027_Grounding_AI_Scrum_Master.pdf` | Compiled PDF from Overleaf. |
| `pilot/build.py` | Picks the 40 sprints, rebuilds their boards, scores the model arm, writes the prompts. |
| `pilot/prompts/` | The exact prompts the agent arms read (8 files of 20 boards each). |
| `pilot/agent_outputs/` | The agents' answers, saved word for word, plus `audit.json` (every tool call each agent made). |
| `pilot/score.py` | Scores the three arms: agreement, AUC, sensitivity, specificity, F1, cost curve, bootstrap intervals. |
| `pilot/rule_baseline.py` | Added 8 Nov 2026. Cohen's kappa, a past-lateness rule, and how often each arm changed its call between checkpoints. |
| `pilot/results/` | Every number in the paper's pilot section. |
| `guide/` | `Rebuild_Manual.docx` (every step to rebuild the pilot by hand) and `Plain_Explanation.docx` (the paper in plain words). |

## The pilot in short

- 40 held-out TAWOS sprints (20 late, 20 on time), drawn with seed 42 from the 472 test sprints that had committed work and were still open at mid-sprint. 37 of the 40 come from five projects.
- Two replay points each: sprint start and halfway. 80 decisions per arm, but calls rarely change between the two points (see below), so they behave like about 40.
- Arm M: the Phase 2 gradient boosting model at the start and the Phase 3 one at halfway, trained on the first 70% of each project only. The threshold comes from training data only.
- Arms L and G: Claude (claude-opus-5-5), run on 4 Nov 2026 as sub-agents in a Claude session, so sampling settings were not under our control. L sees the board; G also sees the model's score and call. Project names, issue keys and dates are hidden. Each agent read one prompt file and nothing else (see `audit.json`). One run.

Main results:

| | Value |
|---|---|
| G agrees with M | 91% of decisions [84, 98], Cohen's kappa 0.82 |
| L agrees with M | 73% [60, 85], kappa 0.47; 19 of its 22 departures are silences |
| G overrules M | 7 times in 80, right 4 times |
| Past-lateness rule (warn if half or more of the last 3 sprints were late) | Same call as L on 39 of 40 sprints at the start; AUC 0.63; M beats it by 0.09 [-0.08, 0.25] |
| Sprints whose call changed between start and halfway | M 3, G 2, L 1 (of 40) |
| AUC (M / L / G) at start | .72 / .69 / .74 |
| AUC (M / L / G) at halfway | .71 / .70 / .72 |
| Cost, G vs M | no detectable difference at any warning price (gap at most 0.02 F) |
| Cost, Delta at kappa/F = 0.5 | -0.086 F [-0.161, -0.019] (L is cheaper) |
| Warnings that say "close on the date, carry work over" | 16 of 81 (8 per arm), coded by hand in `results/warning_messages_coded.csv`. This is what the Scrum Guide asks for, so it shows the late-closure label rewards timebox discipline, not delivery. |

What the companion audit adds (8 Nov 2026, from `ccece-2027-paper`): the model arm's 0.751 is pooled over projects and falls to 0.634 within a project; the recent late rate behind the past-lateness rule ranks a project's own sprints at chance (0.51); 40% of late closures are one to three days late, and sprints planned to end on a Friday or weekend close late 57% of the time against 28% otherwise. That backs the paper's point that late closure measures timebox discipline. A yes-or-no spillover label is no fix, since counting committed issues alone ranks it at 0.88, so the next replay should score the share of committed work left unfinished. The paper now says all of this, and it no longer cites the shuffled-split gap (0.70 to 0.77), which mixes leakage with drift; it cites the two leakage case studies instead.

What the pilot cannot show: it is one run of one model with one prompt on 40 sprints, the boards have no issue text (the TAWOS extract has no titles), the numbers the agents quote were not checked against the boards (so the ungrounded failure is not tested), and the effect of a warning e(c) is assumed.

## How to run

Run from this folder. The pilot reuses Phases 1 to 3, so the folders must stay side by side.

```
python -m pilot.build            # boards, model arm, prompts (about 20 seconds)
# give each file in pilot/prompts to a language-model agent with no tools;
# save each answer as pilot/agent_outputs/<prompt name>.jsonl
python -m pilot.score            # all pilot numbers (about 1 minute)
python -m pilot.rule_baseline    # kappa, rule baseline, checkpoint stability (a few seconds)
```

Our run used Python 3.13, scikit-learn 1.9.1 and pandas 3.0.
