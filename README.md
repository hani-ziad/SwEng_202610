# Sprint-risk prediction on TAWOS, Phases 1 to 3

One project in three phases, built on the TAWOS Jira dataset: 8 open-source projects, 2,602 sprints we can model, 1,691 of them with committed scope. The two things we predict never change. Spillover means a committed issue is unfinished at the planned end. Late closure means the sprint closes more than a day after its planned end. Data: https://rdr.ucl.ac.uk/articles/dataset/The_TAWOS_dataset/21308124 (Apache 2.0, research use).

| Phase | Question | Folder | Main result |
|---|---|---|---|
| 1 | Which sprints are risky, using only what is known at planning time? Four algorithms, baselines and an ablation. | `week-12-method-preliminary-results/sprint-risk-prediction` | Spillover: scope-only logistic regression wins (AUC .885, random forest .852). Late closure: random forest wins (AUC .705, scope-only .530). |
| 2 | Can we say how much, explain it, and build an advisor? Leak-free history features, Shapley explanations. | `phase-2-explainable-sprint-advisor` | Same split: scope drives spillover, team habits drive late closure. The what-if advice has no causal basis. |
| 3 | What if we predict while the sprint is running (25, 50, 75, 100% checkpoints)? | `phase-3-in-sprint-early-warning` | Spillover AUC rises to 0.93 to 0.96 with progress. Late closure does not improve. |

All phases use the same setup: sprint scope rebuilt from the change log, a per-project time-ordered 70/30 split (786 test sprints, 511 with scope), leave-one-project-out, bootstrap over projects (2,000 resamples) and seed 42. Phases 2 and 3 import Phase 1 without changing it, so keep the three folders side by side.

## What is where

- `week-12-method-preliminary-results/sprint-risk-prediction`: Phase 1 code, data and results. Its `README.md` is also the repo README for CSBP711 Assignment 1.
- `phase-2-explainable-sprint-advisor` and `phase-3-in-sprint-early-warning`: each has a README, requirements, tests and results.
- `week-08-proposal`, `week-10-baseline-artifact-plan`, `week-14-full-results-analysis`: the proposal, checkpoints and analysis.
- `week-16-article-artifact-presentation`: revised conference paper (LaTeX source). Compile with `pdflatex` or upload to Overleaf; requires `IEEEtran.cls`.
- `guides`: learning and reproduction guides. `AI-Basics-and-Reproduction-Guide.docx` covers all three phases.
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
```

More detail is in `PHASE1_FREEZE_CHANGELOG.md` and the per-phase READMEs.
