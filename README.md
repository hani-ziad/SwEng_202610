# Sprint-risk prediction on TAWOS, Phases 1 to 4, with external checks on MariaDB and MongoDB

One project in four phases, built on the TAWOS Jira dataset: 8 open-source projects, 2,602 sprints we can model, 1,691 of them with committed scope. The two things we predict never change. Spillover means a committed issue is unfinished at the planned end. Late closure means the sprint closes more than a day after its planned end. Data: https://rdr.ucl.ac.uk/articles/dataset/The_TAWOS_dataset/21308124 (Apache 2.0, research use).

| Phase | Question | Folder | Main result |
|---|---|---|---|
| 1 | Which sprints are risky, using only what is known at planning time? Four algorithms, baselines and an ablation. | `week-12-method-preliminary-results/sprint-risk-prediction` | Spillover: scope-only logistic regression wins (AUC .885, random forest .852). Late closure: random forest wins (AUC .705, scope-only .530). |
| 2 | Can we say how much, explain it, and build an advisor? Leak-free history features, Shapley explanations. | `phase-2-explainable-sprint-advisor` | Same split: scope drives spillover, team habits drive late closure. The what-if advice has no causal basis. |
| 3 | What if we predict while the sprint is running (25, 50, 75, 100% checkpoints)? | `phase-3-in-sprint-early-warning` | Spillover AUC rises to 0.93 to 0.96 with progress. Late closure does not improve. |
| Check | Does it hold outside TAWOS? The frozen pipeline run once on MariaDB Jira (MaxScale and ColumnStore, 391 sprints). | `external-validation` | Late closure transfers: a TAWOS-trained model reaches AUC 0.71 (scope only 0.58). Spillover can't be tested there (99% of sprints spill). |
| 4 | Fixing the weak spots after expert feedback: repeated splits with mean ± SD, ROC curves with uncertainty bands, and a test on 10 external projects (MariaDB and MongoDB). | `phase-4-fixes` | Shuffled splits flatter late closure (.765 ± .017 against .696 ± .054 in time order). On 10 new projects, late closure still ranks above chance in 7 of 8 testable ones (mean AUC .62). |

All phases use the same setup: sprint scope rebuilt from the change log, a per-project time-ordered 70/30 split (786 test sprints, 511 with scope), leave-one-project-out, bootstrap over projects (2,000 resamples) and seed 42. Phases 2 and 3 import Phase 1 without changing it, so keep the three folders side by side.

## What is where

- `week-12-method-preliminary-results/sprint-risk-prediction`: Phase 1 code, data and results. Its `README.md` is also the repo README for CSBP711 Assignment 1.
- `phase-2-explainable-sprint-advisor` and `phase-3-in-sprint-early-warning`: each has a README, requirements, tests and results.
- `week-08-proposal`, `week-10-baseline-artifact-plan`, `week-14-full-results-analysis`: the proposal, checkpoints and analysis.
- `week-16-article-artifact-presentation`: revised conference paper (LaTeX source). Compile with `pdflatex` or upload to Overleaf; requires `IEEEtran.cls`.
- `ccece-2027-paper`: the IEEE CCECE 2027 submission, "Reconstructing Sprint-Start Jira State for Leakage-Aware Outcome Prediction" (LaTeX, PDF, Word copy, both figures). `verify_numbers.py` checks every number in the paper against the result files in this repo, including the MariaDB, MongoDB and Phase 4 figures (359 checks).
- `phase-4-fixes`: Phase 4. Three experiments, each with its own script, results and figure, plus the MongoDB pull and a general Jira converter. See its README.
- `external-validation`: the MariaDB external check. It holds the browser pull script, the raw pull (`data/pull`), the same data in the TAWOS CSV layout (`data/raw/mariadb`), the converter, the runner and the results. See its README.
- `ccece-2027-ai-scrum-master`: the second CCECE 2027 submission, "Grounding the AI Scrum Master: A Sprint-Replay Test and Pilot" (LaTeX, PDF), with the sprint-replay pilot: code, prompts, the agents' raw answers and results. See its README.
- `canadian-ai-2027-paper`: the earlier Canadian AI 2027 version of that position paper (6-page PMLR format, plus an 8-page draft and an IEEE-format copy), kept as a backup venue.
- `Claude outputs`: the CCECE submission sheet (title, authors, abstract, track).
- `guides`: learning and reproduction guides. `Sprint_Risk_Prediction_Walkthrough.html` walks through all four phases and the outside checks; `AI-Basics-and-Reproduction-Guide.docx` covers the first three.
- `CSBP711 Datasets Algorithm Comparison.pptx` and `.pdf`: the 5-slide Assignment 1 deck.

## Running it

```bash
# Phase 1 (this is also the assignment)
cd week-12-method-preliminary-results/sprint-risk-prediction
pip install -r requirements.txt
python -m scripts.run_tests
python -m scripts.run_assignment_comparison

# Phases 2 and 3
cd ../../phase-2-explainable-sprint-advisor && python -m experiments.run_tests && python -m experiments.run_all
cd ../phase-3-in-sprint-early-warning && python -m experiments.run_tests && python -m experiments.run_all

# External check on MariaDB (about 5 minutes)
cd ../external-validation && python run_external.py

# Phase 4 (about 15 minutes)
cd ../phase-4-fixes && python -m experiments.run_all

# Check every number in the CCECE paper
cd .. && python ccece-2027-paper/verify_numbers.py
```

More detail is in `PHASE1_FREEZE_CHANGELOG.md` and the per-phase READMEs.
