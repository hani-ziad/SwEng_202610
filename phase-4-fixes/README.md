# Phase 4: fixing the weak spots of Phases 1-3

After Phases 1-3, a professor reviewed the work and suggested changes to how we test and report it. Experiments 1 to 3 do all of them, one at a time. On 8 Nov 2026 we added experiments 4 and 5, which check the two labels themselves. They found that both Phase 1 labels partly measure something other than their names, and they rerun everything with corrected labels. Experiment 6, added the same evening, scores the corrected labels within each project. Phases 1-3 are not changed; Phase 4 imports them, so the folders have to stay side by side.

| Feedback or finding | What Phase 4 does | Where |
|---|---|---|
| Shuffle sprints, 70/30, repeat 5 times, report mean ± SD | Experiment 1, schemes A and B | `results/e1_summary_pretty.csv` |
| Random cross-validation | Experiment 1, scheme B (5-fold CV, repeated 5 times) | same |
| Keep the result honest about time | Experiment 1, scheme C (time-ordered, 5 repeats) | same |
| ROC curves instead of the bar chart, one per problem | Experiment 2 | `figures/e2_roc_repeats.png`, `figures/e2_roc_bootstrap.png` |
| ROC as a range, with uncertainty | Bands of ± 1 SD over repeats, and 95% bootstrap bands | same |
| 10 external projects, per project sensitivity, specificity, F1 | Experiment 3 | `results/e3_external_per_project.csv`, `figures/e3_external_auc.png` |
| Do the labels measure what they say? | Experiment 4, sanity checks, no model | `results/e4_sanity_checks.csv` |
| Rerun with corrected labels, against rules that need no training | Experiment 5 | `results/e5_*.csv`, `figures/e5_insprint_vs_rules.png` |
| Pooled AUC also rewards telling projects apart | Experiment 6, the corrected labels scored within each project | `results/e6_within_project*.csv` |
| One experiment at a time, documented | One script, one section and one set of outputs per experiment | this file |

Everything uses the same models and features as Phase 2 (strict planning-time features plus project): gradient boosting, random forest and logistic regression, with scope-only logistic regression as the baseline. Seed 42.

**Decision threshold.** Sensitivity, specificity and F1 need a cut-off. In every experiment it is fixed on the training data only, as the point that maximises sensitivity + specificity (Youden's J) on 5-fold out-of-fold predictions inside the training set. It is never tuned on test data. Rules get their threshold the same way, from their training-side scores.

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
- **Part of the time-ordered drop is just less training data.** The time-ordered repeats train on 40% up to 80% of each project's history, and AUC climbs with it: .62, .66, .71, .75, .74. The 70% cut used in the paper is the best of the five, so .70 is the fairer single figure.
- **The ± .054 is not a confidence interval.** The five cuts overlap and train on growing amounts of history, so their spread mixes training size with chance.
- **Spillover holds up under every scheme** (.81 to .84), and the scope-only baseline roughly matches the learned models. Experiment 4 shows why: the label is mostly sprint size.
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

The spillover bands are wide because only 29 test sprints finish everything.

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

- **Late closure carried over weakly, on the old label.** The point estimate is above .5 in 7 of 8 projects (mean .62), but only 2 of the 8 intervals (ColumnStore, PHP Driver) clear .5. With the corrected label of experiment 5 the mean falls to .55, so most of this transfer was the weekend effect, not team behaviour.
- **The ranking itself flips, not just the threshold.** The scope-only baseline averages .362, below chance, and .05 on the C Driver: outside TAWOS, bigger sprints are less often late. Beating a below-chance baseline says little.
- **The threshold does not travel either.** Sensitivity swings from .11 to .86 between projects, because late-closure rates differ so much (7% in Compass, 54% in ColumnStore).
- **The spillover average should not be quoted.** In 6 of 10 projects almost every sprint leaves something unfinished. These projects have complete resolution dates (experiment 4), so this is real, not the Confluence Cloud defect. The 4-project mean rests on 6 to 9 sprints without spillover per project.

**Limits.** Some MongoDB teams share a board (Compass and Shell, for example), and a few sprints hold issues from projects we did not pull. Each sprint is given to the project that owns most of its issues, and only that project's issues count towards its scope, the same rule as for TAWOS and MariaDB. That can under-count the scope of shared sprints. The small projects (20 to 40 sprints) give wide intervals.

## Experiment 4: do the labels measure what they say? (added 8 Nov 2026)

No model. `experiments/e4_sanity_checks.py` reads the saved Phase 2 and Phase 3 frames and the raw exports. All numbers are in `results/e4_sanity_checks.csv`.

| Check | Result |
|---|---|
| Spillover rate, sprints with committed work | 93.1% (1,575 of 1,691) |
| Spillover rate by committed issues: 1 / 2 / 3 / 4-5 / 6-10 / 11+ | .75 / .86 / .91 / .96 / .97 / .997 |
| Phase 3 spillover, open committed issues alone (no model), 25/50/75% | AUC .925 / .945 / .954 (the model: .926 / .957 / .953) |
| Phase 3 spillover test sprints without spillover | 29 at every checkpoint |
| Late sprints closed within 3 days of the planned end | 40% |
| Late sprints that end Friday to Sunday and close within 3.5 days | 32.8% |
| Finished issues (Closed, Done, Resolved) with no resolution date | Confluence Cloud 68.4%; every other TAWOS project 0 to 0.5%; MariaDB 0.8%; MongoDB 0% |

What it shows:

- **Spillover is a sprint-size detector.** "At least one of n issues unfinished" is almost certain for large n. The Phase 3 headline (.957) is a count of open issues; the share-done rule Phase 3 used as its baseline threw that count away.
- **A third of late closures look like closing on the next working day.** The "more than 1 day" rule counts a Friday end closed on Monday as late.
- **The resolution-date defect is Confluence Cloud only.** Confluence Server's many sprints with nothing done at the planned end are real late resolutions, not missing data. The export has no status history, so Cloud's finish times cannot be recovered.

## Experiment 5: rerun with corrected labels (added 8 Nov 2026)

`phase4/relabel.py` defines two corrected labels, fixed before any model was run on them:

- **Late closure:** closed after the end of the next working day (Mon-Fri) after the planned end. Dates are UTC. All 2,602 sprints. The late rate drops from 38.7% to 25.4%; The Titanium SDK drops from 60% to 19%.
- **Heavy carry-over** (replaces spillover): half or more of the committed issues unresolved at the planned end. Sprints with committed work, Confluence Cloud excluded: 1,494 sprints, 78.0% positive. In Cloud, the history features that count finished issues are set to missing.

History features are rebuilt with the new late label. Every model is compared with rules that need no training: committed count, the team's past lateness (share of its last 3 known sprints that were late), its past carry share, and in-sprint the remaining count and remaining share. On the fixed split, the best rule is picked by its training-side AUC, never the test side, and the gap is a paired bootstrap (2,000 resamples of test sprints).

**Planning time, time-ordered cuts (mean ± SD over 5):**

| Model or rule | Late closure (next working day) | Heavy carry-over |
|---|---|---|
| Gradient boosting | .706 ± .047 | .634 ± .035 |
| Random forest | .696 ± .033 | .662 ± .068 |
| Logistic regression | .687 ± .019 | .664 ± .098 |
| Scope-only LR | .556 ± .013 | .590 ± .043 |
| Committed count (rule) | .466 ± .029 | .567 ± .037 |
| Past lateness or past carry share (rule) | .627 ± .015 | .625 ± .028 |

**Fixed time split, gradient boosting against the best rule:**

| Task | n (pos.) | Gradient boosting | Best rule | Gap [95% CI] |
|---|---|---|---|---|
| Late closure | 786 (167) | .727 [.681, .771] | past lateness .619 | +.108 [.069, .150] |
| Heavy carry-over | 451 (387) | .565 [.488, .644] | past carry share .621 | −.056 [−.151, .040] |

**In-sprint (Phase 3 split, sprints still open, Confluence Cloud excluded):**

| Task | Checkpoint | GB, planning | GB, progress | Remaining share (rule) | GB minus best rule [95% CI] |
|---|---|---|---|---|---|
| Late closure | 25 / 50 / 75% | .709 / .700 / .702 | .715 / .715 / .710 | .526 / .522 / .532 | +.090 / +.089 / +.085 vs past lateness, all intervals above 0 |
| Heavy carry-over | 25% | .574 | .782 | .716 | +.067 [−.005, .138] |
| Heavy carry-over | 50% | .593 | .839 | .810 | +.029 [−.040, .094] |
| Heavy carry-over | 75% | .594 | .900 | .872 | +.029 [−.022, .085] |

**Outside projects (trained on TAWOS only, gradient boosting):** late closure mean AUC .546 ± .139 over 9 measurable projects, interval above .5 in 2 (ColumnStore .678, WiredTiger .675), Java Driver .244. Heavy carry-over .522 ± .107 over 9, no interval above .5. Per project in `results/e5_external_per_project.csv`.

**Label sensitivity** (`results/e5_label_sensitivity.csv`, time-ordered, GB): late closure .689 (old 1-day rule), .701 (flat 3 days), .706 (next working day). On the same 1,494 sprints, the old any-spillover label gives committed count .890 against GB .813; carry share ≥ 20% gives .791 against .701; ≥ 30% gives .690 against .644; ≥ 50% gives .567 against .634. Only at 50% does size stop explaining the pooled label; within a project it still does (experiment 6).

What it shows:

- **Late closure survives the weekend fix inside TAWOS.** The fixed-split AUC drops from .751 to .727, and the model beats the team's own past-lateness rate by .108 pooled. Experiment 6 shows this is not only project separation: within a project the model scores .631 against .482 for past lateness. Progress during the sprint still adds nothing (at most +.015).
- **Late-closure transfer to outside projects does not survive.** Mean .546 with 2 of 9 intervals above .5. The old .620 was partly teams that close on Monday.
- **Heavy carry-over cannot be predicted at planning time.** Models land at .63 to .66 over the time cuts, and on the fixed split gradient boosting is no better than the team's past carry share (gap −.056, interval −.151 to .040).
- **In-sprint, heavy carry-over becomes predictable, mostly by arithmetic.** At 50% of the sprint, the share still open (no model) gives .810 and gradient boosting .839; the gap's interval includes zero at 50% and 75%, and barely at 25%. A tool would do as well with "how much is still open".
- **Every interval here is a sprint-level bootstrap.** Projects are not resampled, so the true uncertainty is wider.

## Experiment 6: the corrected labels within each project (added 8 Nov 2026)

Pooled AUC also rewards telling projects apart: the CCECE audit found the old late-closure label at .751 pooled but .634 within a project. Experiment 6 reruns experiment 5's fixed time split and scores every model and rule inside each project, then averages over the projects with both classes in the test period (8 for late closure, 7 for heavy carry-over). Intervals resample sprints inside each project, 2,000 times.

| Task | Model or rule | Pooled | Within a project |
|---|---|---|---|
| Late closure (next working day) | Gradient boosting | .727 | .631 [.531, .689] |
| | Random forest | .726 | .556 |
| | Past lateness (rule) | .619 | .482 |
| | Committed count (rule) | .456 | .473 |
| Heavy carry-over | Gradient boosting | .565 | .521 [.442, .599] |
| | Past carry share (rule) | .621 | .520 |
| | Committed count (rule) | .582 | .621 |

What it shows:

- **The late-closure signal is not only project separation.** Within a project gradient boosting beats past lateness by .149 (interval .053 to .195), and past lateness on its own is at chance.
- **Heavy carry-over is at chance within a project for every model.** The only thing that ranks it there is sprint size: bigger sprints leave half their work unfinished more often (.621). So "size no longer explains the label" holds only for the pooled score.

## How to run

```
pip install -r requirements.txt
python -m experiments.run_all      # about 30 minutes; or run e1 to e6 one by one
```

Run from the `phase-4-fixes` folder. Experiment 2 needs the scores saved by experiment 1, and experiment 4 reads `results/e3_external_projects.csv`. To rebuild the external tables from the raw pulls:

```
cd external
python convert_jira.py data/pull/mariadb_mxs_mcol.json data/raw/mariadb MXS="MariaDB MaxScale" MCOL="MariaDB ColumnStore"
python convert_jira.py data/pull/mongodb_8projects.json data/raw/mongodb COMPASS="MongoDB Compass" WT="WiredTiger" MONGOSH="MongoDB Shell" JAVA="MongoDB Java Driver" CDRIVER="MongoDB C Driver" CXX="MongoDB C++ Driver" PHPC="MongoDB PHP Driver" TOOLS="MongoDB Database Tools"
```

Our run used Python 3.13, scikit-learn 1.9.1 and pandas 3.0. Phase 2's numbers were made with scikit-learn 1.8, so a few values can differ in the third decimal (gradient boosting spillover on the paper's split is .817 here and .812 in the paper). Experiments 4 and 5 were run on 8 Nov 2026 in a cloud copy of the repo whose Phase 2 frame matched the saved one exactly (2,602 sprints, zero differences on labels and features).

## Files

- `phase4/common.py`: tasks, models, the threshold rule and the metrics.
- `phase4/relabel.py`: the two corrected labels and the frame built with them.
- `experiments/e1_repeated_splits.py`, `e2_roc.py`, `e3_external.py`, `e4_sanity_checks.py`, `e5_relabelled.py`, `e6_within_project.py`, `run_all.py`.
- `external/pull_jira_browser.js`, `external/convert_jira.py`, `external/data/pull/` (raw pulls), `external/data/raw/` (TAWOS-layout tables).
- `results/`: every number above, per repeat and per project. `figures/`: the four figures.

Data sources: TAWOS (Tawosi et al., MSR 2022, Apache 2.0); the public MariaDB and MongoDB Jira instances (read-only, no login).
