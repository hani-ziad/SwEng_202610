# Phase 4: fixing the weak spots of Phases 1-3

After Phases 1-3, a professor reviewed the work and suggested changes to how we test and report it. Phase 4 does all of them, one experiment at a time. Phases 1-3 are not changed; Phase 4 imports them, so the folders have to stay side by side.

| Feedback | What Phase 4 does | Where |
|---|---|---|
| Shuffle sprints, 70/30, repeat 5 times, report mean ± SD | Experiment 1, schemes A and B | `results/e1_summary_pretty.csv` |
| Random cross-validation | Experiment 1, scheme B (5-fold CV, repeated 5 times) | same |
| Keep the result honest about time | Experiment 1, scheme C (time-ordered, 5 repeats) | same |
| ROC curves instead of the bar chart, one per problem | Experiment 2 | `figures/e2_roc_repeats.png`, `figures/e2_roc_bootstrap.png` |
| ROC as a range, with uncertainty | Bands of ± 1 SD over repeats, and 95% bootstrap bands | same |
| 10 external projects, per project sensitivity, specificity, F1 | Experiment 3 | `results/e3_external_per_project.csv`, `figures/e3_external_auc.png` |
| One experiment at a time, documented | One script, one section and one set of outputs per experiment | this file |

Everything uses the same models and features as Phase 2 (strict planning-time features plus project): gradient boosting, random forest and logistic regression, with scope-only logistic regression as the baseline. Seed 42.

**Decision threshold.** Sensitivity, specificity and F1 need a cut-off. In every experiment it is fixed on the training data only, as the point that maximises sensitivity + specificity (Youden's J) on 5-fold out-of-fold predictions inside the training set. It is never tuned on test data.

## Experiment 1: repeated splits, mean ± SD

Three ways to split, 5 repeats each:

- **A. Random 70/30.** Sprints are shuffled with a different seed each time. This is what the professor asked for.
- **B. Random 5-fold CV,** repeated 5 times with different shuffles.
- **C. Time-ordered.** For each project the sprints stay in time order; repeat k trains on the first 40/50/60/70/80% and tests on the next 20%. The model never sees a sprint from after the ones it is tested on.

Gradient boosting (mean ± SD over 5 repeats):

| Problem | Scheme | AUC | Sensitivity | Specificity | F1 |
|---|---|---|---|---|---|
| Late closure | A. Random 70/30 | .765 ± .017 | .683 ± .045 | .706 ± .048 | .635 ± .013 |
| Late closure | B. Random 5-fold CV | .754 ± .005 | .634 ± .014 | .739 ± .016 | .619 ± .005 |
| Late closure | C. Time-ordered | .696 ± .054 | .509 ± .080 | .790 ± .047 | .529 ± .064 |
| Spillover | A. Random 70/30 | .813 ± .016 | .687 ± .065 | .789 ± .077 | .806 ± .044 |
| Spillover | B. Random 5-fold CV | .836 ± .007 | .680 ± .024 | .852 ± .019 | .804 ± .017 |
| Spillover | C. Time-ordered | .806 ± .028 | .707 ± .061 | .747 ± .113 | .821 ± .041 |

The scope-only baseline gets .557 ± .018 for late closure (scheme A) and .831 ± .026 for spillover. The full table, with all four models, is in `results/e1_summary_pretty.csv`; every single repeat is in `results/e1_repeats.csv`.

What it shows:

- **Shuffling makes late closure look better and steadier than it is.** Random splits give .75 to .77 with a tiny SD. Keeping time order gives .70 ± .05. A shuffled split lets the model learn a team's later habits and test on its earlier sprints, which can never happen in real use.
- **Part of the time-ordered drop is just less training data.** The time-ordered repeats train on 40% up to 80% of each project's history, and AUC climbs with it: .62, .66, .71, .75, .74. With 70% or more, it matches the paper's .751.
- **Spillover holds up under every scheme** (.81 to .84), and the scope-only baseline roughly matches the learned models, as in the paper. Spillover is mostly about how much was committed.
- **Use the time-ordered numbers as the headline,** and the shuffled ones as the optimistic comparison the professor asked for.
- **Not all of the shuffled gap is leakage** (added 8 Nov 2026). The late rate of the five time-ordered test windows runs from 27% to 47%, so part of the gap fits drift over time. All of these AUCs are also pooled over projects. Within a single project, gradient boosting on the 70% split scores 0.634, against 0.751 pooled (`ccece-2027-paper/reviewer_checks/`).

## Experiment 2: ROC curves with uncertainty

Two panels in each figure: late closure on the left, spillover on the right (on the 1,691 sprints with committed work, since empty sprints make spillover trivially easy).

- `figures/e2_roc_repeats.png`: gradient boosting, mean ROC curve with a ± 1 SD band over the 5 repeats, for shuffled 70/30 and for time-ordered splits.
- `figures/e2_roc_bootstrap.png`: the paper's own time split (last 30% of each project), with a 95% band from 2,000 bootstrap resamples, and the scope-only baseline for comparison.

| Paper's time split | Gradient boosting | Scope-only LR |
|---|---|---|
| Late closure (786 test sprints, 246 late) | .751 [.711, .791] | .530 [.484, .574] |
| Spillover (511 test sprints, 29 without spillover) | .817 [.746, .886] | .885 [.841, .924] |

The spillover bands are wide because only 29 test sprints finish everything. That is worth saying in the paper.

## Experiment 3: 10 external projects

Trained once on all 8 TAWOS projects and tested on 10 projects that were never used in any earlier phase:

- **MariaDB Jira (2):** MaxScale and ColumnStore (the same data as the earlier external check).
- **MongoDB Jira (8):** Compass, WiredTiger, Shell, Java Driver, C Driver, C++ Driver, PHP Driver and Database Tools.

The MongoDB data was pulled from its public Jira with `external/pull_jira_browser.js` in a browser (the cloud sandbox cannot reach it), on 4 Nov 2026: 47,038 issues scanned, 17,773 kept (those that were ever in a sprint), no errors and no cut-off change logs. `external/convert_jira.py` turns a pull into the TAWOS layout; it is the MariaDB converter made general, and it reproduces the MariaDB tables byte for byte.

Rules, fixed before looking at any external result: no project feature (the projects are new) and no story-point features (MariaDB has none), on both sides; the threshold comes from TAWOS only; late closure is tested on all sprints and spillover on sprints with committed work; a metric is marked not interpretable when a project has fewer than 5 sprints in either class.

Gradient boosting, per project (95% bootstrap intervals are in the CSV):

| Project | Late closure: n (late/on time) | AUC | Sens. | Spec. | F1 | Spillover: n (spill/no spill) | AUC | Sens. | Spec. | F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| MariaDB ColumnStore | 122 (66/56) | .699 | .636 | .661 | .661 | 106 (106/0) | not interpretable | | | |
| MariaDB MaxScale | 269 (40/229) | .592 | .300 | .873 | .296 | 245 (242/3) | not interpretable | | | |
| MongoDB C Driver | 36 (33/3) | not interpretable | | | | 28 (27/1) | not interpretable | | | |
| MongoDB C++ Driver | 29 (17/12) | .623 | .824 | .333 | .718 | 25 (19/6) | .816 | .526 | .833 | .667 |
| MongoDB Compass | 127 (9/118) | .481 | .111 | .949 | .125 | 125 (123/2) | not interpretable | | | |
| MongoDB Database Tools | 43 (21/22) | .567 | .857 | .500 | .720 | 23 (15/8) | .663 | .200 | 1.000 | .333 |
| MongoDB Java Driver | 27 (24/3) | not interpretable | | | | 22 (20/2) | not interpretable | | | |
| MongoDB PHP Driver | 32 (13/19) | .781 | .615 | .789 | .640 | 25 (16/9) | .674 | .312 | 1.000 | .476 |
| MongoDB Shell | 21 (7/14) | .582 | .429 | .786 | .462 | 20 (14/6) | .786 | .214 | .833 | .333 |
| WiredTiger | 211 (24/187) | .635 | .208 | .925 | .233 | 205 (203/2) | not interpretable | | | |

Averages over the projects where the metric can be measured (mean ± SD):

| Problem | Projects | AUC | Sensitivity | Specificity | F1 |
|---|---|---|---|---|---|
| Late closure, gradient boosting | 8 | .620 ± .090 | .498 ± .279 | .727 ± .217 | .482 ± .237 |
| Late closure, scope-only baseline | 8 | .362 ± .118 | .147 ± .203 | .808 ± .211 | .158 ± .219 |
| Spillover, gradient boosting | 4 | .734 ± .078 | .313 ± .151 | .917 ± .096 | .452 ± .158 |
| Spillover, scope-only baseline | 4 | .742 ± .133 | .498 ± .261 | .802 ± .318 | .599 ± .215 |

What it shows:

- **Late closure carries over, but weakly.** A model that never saw these teams ranks their late sprints above their on-time ones in 7 of 8 projects (mean AUC .62). It beats the scope-only baseline in every project; that baseline is on average below chance here. Within a single new project it is clearly weaker than on TAWOS (.75), which is expected: each team has its own habits about when to close a sprint.
- **The threshold does not travel well.** Sensitivity swings from .11 to .86 between projects, because late-closure rates differ so much (7% in Compass, 54% in ColumnStore). A threshold set on TAWOS fits some teams and not others. Any real tool would need a short calibration on the new team.
- **Spillover cannot be tested on most of these projects.** In 6 of 10, almost every sprint leaves something unfinished, which leaves too few counter-examples. Where it can be tested (4 projects), AUC is .73 and the scope-only baseline does as well, as on TAWOS.

**Limits.** Some MongoDB teams share a board (Compass and Shell, for example), and a few sprints hold issues from projects we did not pull. Each sprint is given to the project that owns most of its issues, and only that project's issues count towards its scope, the same rule as for TAWOS and MariaDB. That can under-count the scope of shared sprints. The small projects (20 to 40 sprints) give wide intervals, so the per-project numbers should be read together with their counts.

## How to run

```
pip install -r requirements.txt
python -m experiments.run_all      # about 15 minutes; or run e1, e2 and e3 one by one
```

Run from the `phase-4-fixes` folder. Experiment 2 needs the scores saved by experiment 1. To rebuild the external tables from the raw pulls:

```
cd external
python convert_jira.py data/pull/mariadb_mxs_mcol.json data/raw/mariadb MXS="MariaDB MaxScale" MCOL="MariaDB ColumnStore"
python convert_jira.py data/pull/mongodb_8projects.json data/raw/mongodb COMPASS="MongoDB Compass" WT="WiredTiger" MONGOSH="MongoDB Shell" JAVA="MongoDB Java Driver" CDRIVER="MongoDB C Driver" CXX="MongoDB C++ Driver" PHPC="MongoDB PHP Driver" TOOLS="MongoDB Database Tools"
```

Our run used Python 3.13, scikit-learn 1.9.1 and pandas 3.0. Phase 2's numbers were made with scikit-learn 1.8, so a few values can differ in the third decimal (gradient boosting spillover on the paper's split is .817 here and .812 in the paper).

## Files

- `phase4/common.py`: tasks, models, the threshold rule and the metrics.
- `experiments/e1_repeated_splits.py`, `e2_roc.py`, `e3_external.py`, `run_all.py`.
- `external/pull_jira_browser.js`, `external/convert_jira.py`, `external/data/pull/` (raw pulls), `external/data/raw/` (TAWOS-layout tables).
- `results/`: every number above, per repeat and per project. `figures/`: the three figures.

Data sources: TAWOS (Tawosi et al., MSR 2022, Apache 2.0); the public MariaDB and MongoDB Jira instances (read-only, no login).
