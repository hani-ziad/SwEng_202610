# Grounding the AI Scrum Master (CCECE 2027)

Position paper for IEEE CCECE 2027, with a replay study of the sprint-replay test it proposes. This is a separate paper from `ccece-2027-paper` (the sprint-risk paper): different question, different results. It started as the Canadian AI 2027 short paper in `canadian-ai-2027-paper`, now in IEEE format with the study added.

| File | What it is |
|---|---|
| `CCECE_2027_Grounding_AI_Scrum_Master.tex` | The paper (IEEEtran). Same text as the Overleaf project "Grounding the AI Scrum Master - CCECE 2027". |
| `CCECE_2027_Grounding_AI_Scrum_Master.pdf` | Compiled PDF from Overleaf. |
| `study/` | The 120-sprint replay (15 Nov 2026), the study the paper reports. Code, prompts, raw agent and rater answers, audits and results. |
| `study/build.py` | Draws 15 held-out sprints per project, rebuilds their boards, scores the late-closure and unfinished-share models, writes 72 prompt files (3 runs). |
| `study/collect.py` | Saves each agent answer word for word and audits every tool call (`agent_outputs/audit.json`). |
| `study/score.py` | All replay numbers: per run, arm and checkpoint, run-to-run agreement, bootstrap intervals, cost curve. |
| `study/factcheck.py` | Traces every number in every warning to its board. |
| `study/rate.py` | Two independent rater agents judge every warning for unsupported claims and close-on-date advice. |
| `study/paper_numbers.py` | Prints the paper's table rows and checks all study numbers in the paper against the result files. |
| `study/write_guides.py` | Rebuilds the two guide documents from the result files. |
| `pilot/` | The first 40-sprint pilot (4 Nov 2026), kept as it was. |
| `guide/` | `Rebuild_Manual.docx` (every step to rebuild the study by hand) and `Plain_Explanation.docx` (the paper in plain words). |

## The study in short

- 120 held-out TAWOS sprints, 15 from each of the eight projects (seed 2027), drawn from the 472 test sprints that had committed work and were still open at mid-sprint. 33 closed late (1 to 10 per project). No balancing on the outcome.
- Two replay points each: sprint start and halfway.
- Arm M: the Phase 2 gradient boosting model at the start and the Phase 3 one at halfway, trained on the first 70% of each project only, threshold from training data only. A second model of the same kind predicts the share of committed issues left unresolved at the planned end.
- Arms L and G: Claude (claude-opus-5-5) sub-agents, three independent runs, boards in a new random order each run. L sees the board; G also sees both models' outputs. Project names, issue keys and dates are hidden. Every agent read only its own prompt file (`study/agent_outputs/audit.json`).
- Baseline rule: warn if at least half of the last three sprints closed late.

Main results (agent numbers are means of three runs; intervals are 95% bootstrap over sprints within project):

| | Value |
|---|---|
| G agrees with M | 89% of decisions, kappa 0.75 |
| G's departures from M | 24 to 26 of 240 per run: 16 extra warnings, all on sprints that closed on time; 8 to 10 silences, 5 right |
| AUC, G minus M | -0.04 at both checkpoints (intervals about -0.10 to 0.02) |
| Cost, G minus M | G costs more at every warning price: 0.010 F [0.002, 0.024] at kappa/F = 0.02 up to 0.031 F [0.006, 0.059] at 0.5 |
| L vs the past-lateness rule at planning | same call on 119, 120 and 116 of 120 sprints; L's within-project AUC 0.47, the rule's 0.46 |
| M minus rule, AUC at planning | 0.14 [0.03, 0.24] |
| Calibration (Brier) at planning | M 0.163, G 0.177, L 0.206; always quoting the late rate 0.199 |
| Run-to-run | same call in all three runs on 93% (L) and 94% (G) of points |
| Unfinished share at planning | nothing predicts it (all correlations under 0.11; no arm beats a constant guess) |
| Unfinished share at halfway (Spearman) | M 0.66, G 0.64, L 0.56; team's own record 0.10 |
| Numbers in warnings traced to the board | 1,129 of 1,131 (one advice, one real error) |
| Unsupported claims (both raters) | L 16 of 168 (10%), G 13 of 227 (6%); rater kappa 0.49 |
| "Close on the date, carry work over" (both raters) | L 49 of 168 (29%), G 34 of 227 (15%); rater kappa 0.66; on the 81 pilot warnings the raters matched the original hand coding at kappa 0.81 |

What it means: the grounded agent repeats its model and its own calls are false alarms; the agent alone is the past-lateness rule; its messages are mostly faithful to the board, with overreach rather than invented numbers; and its close-on-date advice is what the Scrum Guide asks, so late closure measures timebox discipline more than delivery.

What the study cannot show: one model with one prompt; the raters are the same model family as the agents; the boards have no issue text (the TAWOS extract has no titles); e(c) is assumed; the unfinished share leaves out Confluence Cloud, where most issues have no resolution date.

**Hidden issues (noted 14 Nov 2026).** Public Jira hides private and deleted issues. Across the eight TAWOS projects a median of 26% of issue numbers since the first sprint issue are missing (`../ccece-2027-paper/reviewer_checks/hidden_issues.csv`), so the boards all arms saw may be incomplete. The paper's Limits of the replay say so.

**From the companion audit (8 Nov 2026).** The model arm's 0.751 is pooled over projects and falls to 0.634 within a project; 40% of late closures are one to three days late, and sprints planned to end on a Friday or weekend close late 57% of the time against 28% otherwise.

## How to run

Run from this folder. The study reuses Phases 1 to 3, so the folders must stay side by side.

```
python -m study.build                       # boards, models, 72 prompts (about 15 seconds)
# give each prompt file to an agent that may read only that file (see guide/Rebuild_Manual.docx)
python -m study.collect <transcript folder> # save answers, audit tool calls
python -m study.score                       # all replay numbers (about 3 minutes)
python -m study.factcheck                   # number check of every warning
python -m study.rate build                  # 40 rater prompts; give each to a rater agent
python -m study.rate collect <transcript folder>
python -m study.rate score                  # rater results
python -m study.paper_numbers check         # every study number in the paper (must say 0 problems)
python -m study.write_guides                # rebuild the two guide documents
```

The 4 Nov pilot reruns with `python -m pilot.build`, `python -m pilot.score` and `python -m pilot.rule_baseline`.

Our run used Python 3.13, scikit-learn 1.9.1, pandas 3.0 and scipy.
