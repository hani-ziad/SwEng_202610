# Sprint-risk prediction on TAWOS, Phases 1 to 4, with external checks on MariaDB and MongoDB

One project in four phases, built on the TAWOS Jira dataset: 8 open-source projects, 2,602 sprints we can model, 1,691 of them with committed scope. Phases 1 to 3 predict two things. Spillover means a committed issue is unfinished at the planned end. Late closure means the sprint closes more than a day after its planned end. Phase 4 found that both labels partly measure something else (sprint size, and closing on the next working day after a weekend) and reran everything with corrected labels. Data: https://rdr.ucl.ac.uk/articles/dataset/The_TAWOS_dataset/21308124 (Apache 2.0, research use).

| Phase | Question | Folder | Main result |
|---|---|---|---|
| 1 | Which sprints are risky, using only what is known at planning time? Four algorithms, baselines and an ablation. | `week-12-method-preliminary-results/sprint-risk-prediction` | Spillover: scope-only logistic regression wins (AUC .885, random forest .852). Late closure: random forest wins (AUC .705, scope-only .530). Checked 8 Nov: counting committed issues alone gives .881 for spillover, so the .885 says the label is easy to rank, not that risk is predictable. |
| 2 | Can we say how much, explain it, and build an advisor? Leak-free history features, Shapley explanations. | `phase-2-explainable-sprint-advisor` | Same split: scope drives spillover, team habits drive late closure. Gradient boosting reaches .751 for late closure pooled over projects, but .634 within a project, and 40% of late sprints are only one to three days late. The what-if advice has no causal basis. |
| 3 | What if we predict while the sprint is running (25, 50, 75, 100% checkpoints)? | `phase-3-in-sprint-early-warning` | Spillover AUC rises to 0.93 to 0.96 with progress, but the number of committed issues still open, with no model, gets 0.925 to 0.954 and beats the model across projects. Late closure does not improve. |
| Check | Does it hold outside TAWOS? The frozen pipeline run once on MariaDB Jira (MaxScale and ColumnStore, 391 sprints). | `external-validation` | Late closure transfers weakly: a TAWOS-trained model reaches AUC 0.71 pooled (scope only 0.58), but 0.70 and 0.59 per project, and on MariaDB's own time split it scores below the project's late rate (0.69 against 0.74). Spillover can't be tested there (99% of sprints spill). |
| 4 | Fixing the weak spots after expert feedback: repeated splits with mean ± SD, ROC curves with uncertainty bands, a test on 10 external projects (MariaDB and MongoDB), and (8 Nov) a check of the labels themselves with a rerun on corrected labels. | `phase-4-fixes` | Shuffled splits flatter late closure (.765 ± .017 against .696 ± .054 in time order). On 10 new projects only 2 of 8 late-closure intervals exclude 0.5, so transfer is weak. With a next-working-day grace, late closure ranks at .727 pooled and .631 within a project (past late rate .482 within), but outside TAWOS it falls to chance (mean .546). Heavy carry-over (half the commitment unfinished) is not predictable at planning time (within a project the committed count .621 beats the model .521); mid-sprint, the share still open (.810) does nearly as well as the model (.839). |

All phases use the same setup: sprint scope rebuilt from the change log, a per-project time-ordered 70/30 split (786 test sprints, 511 with scope), leave-one-project-out, bootstrap over projects (2,000 resamples) and seed 42. Phases 2 and 3 import Phase 1 without changing it, so keep the three folders side by side.

**Known data defect.** In Atlassian Confluence Cloud, 68% of finished issues have no resolution date, and the export has no status history to recover them. Anything that reads "finished by the planned end" (spillover, carry share, progress features) is wrong for that project. Phase 4 experiment 5 leaves it out of those tasks. Late closure is not affected.

## What is where

- `week-12-method-preliminary-results/sprint-risk-prediction`: Phase 1 code, data and results. Its `README.md` is also the repo README for CSBP711 Assignment 1.
- `phase-2-explainable-sprint-advisor` and `phase-3-in-sprint-early-warning`: each has a README, requirements, tests and results.
- `week-08-proposal`, `week-10-baseline-artifact-plan`, `week-14-full-results-analysis`: the proposal, checkpoints and analysis.
- `week-16-article-artifact-presentation`: revised conference paper (LaTeX source). Compile with `pdflatex` or upload to Overleaf; requires `IEEEtran.cls`.
- `ccece-2027-paper`: the IEEE CCECE 2027 submission. On 8 Nov 2026 it was retitled "Predicting Sprint Outcomes from Reconstructed Jira State: Count Baselines and a Leakage Audit" and reframed as an audit, since rebuilding sprint state from the change log was already done by Choetkiertikul et al. (IEEE TSE, 2018). It holds the LaTeX, PDF, Word copy, both figures, `REVIEW_8Nov2026.md` (every review finding and what was done) and `reviewer_checks/` (the extra analyses behind the new numbers). `verify_numbers.py` checks every number in the paper against the result files in this repo (the original 359 checks plus about 140 added on 8 Nov). `verify_labels.py` checks the corrected-label numbers (Phase 4 experiments 4 to 6).
- `phase-4-fixes`: Phase 4. Six experiments, each with its own script and results, plus the MongoDB pull and a general Jira converter. See its README.
- `external-validation`: the MariaDB external check. It holds the browser pull script, the raw pull (`data/pull`), the same data in the TAWOS CSV layout (`data/raw/mariadb`), the converter, the runner and the results. See its README.
- `ccece-2027-ai-scrum-master`: the second CCECE 2027 submission, "Grounding the AI Scrum Master: A Sprint-Replay Test and Pilot" (LaTeX, PDF), with the sprint-replay pilot: code, prompts, the agents' raw answers and results. See its README.
- `canadian-ai-2027-paper`: the earlier Canadian AI 2027 version of that position paper (6-page PMLR format, plus an 8-page draft and an IEEE-format copy), kept as a backup venue.
- `Claude outputs`: the CCECE submission sheet (title, authors, abstract, track).
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

# Phase 4 (about 30 minutes)
cd ../phase-4-fixes && python -m experiments.run_all

# Check every number in the CCECE paper
cd .. && python ccece-2027-paper/verify_numbers.py
```

More detail is in `PHASE1_FREEZE_CHANGELOG.md` and the per-phase READMEs.
