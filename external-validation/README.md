# External validation on MariaDB Jira (MaxScale + ColumnStore)

The frozen Phase 1-3 pipeline, run once on a second Jira source that is not in TAWOS. No model, label, feature, split or reconstruction code was edited. `run_external.py` just points the existing loaders at `data/raw/mariadb`, which `convert_mariadb.py` writes in the TAWOS CSV layout.

## Data

- Source: the public MariaDB Jira REST API (jira.mariadb.org, no login). Pulled 2026-11-01 with `pull_mariadb_browser.js`, which ran in Chrome because the cloud sandbox can't reach the site. The pull returned 11,484 issues with no errors and no truncated change logs.
- Projects: MaxScale (MXS, board 34) and ColumnStore (MCOL, board 37). Only each project's home board is kept. Five stray sprints on other teams' boards were dropped before any outcome was looked at.
- Cohort after the frozen rules: 391 modeled sprints (269 MXS, 122 MCOL), Jan 2016 to Oct 2026. 351 are nonempty; only 41 of 393 (10%) have zero reconstructed scope, against 35% in TAWOS.
- Story points: MariaDB doesn't use them; the field is empty on every issue. The frozen code therefore reads committed and delivered points as 0. The transfer arm drops the three point columns on both sides.

## Plan (fixed before results)

1. **Sanity:** rerun the paper's headline rows on TAWOS through the same harness.
2. **Within MariaDB:** the paper's protocol (per-project time-ordered 70/30 split, project one-hot, same models and settings).
3. **Transfer:** train on all 8 TAWOS projects and test on every MariaDB sprint, with no project feature and no story points.

Intervals are 95% sprint-level bootstraps (2,000 paired resamples, seed 42). Two projects are too few for the paper's project-cluster bootstrap.

Environment: the recorded results come from a run on Hani's laptop (Windows 11, Python 3.14, scikit-learn 1.9.0, pandas 3.0; see `results/environment_versions.json`). A separate Linux run with scikit-learn 1.9.1 gave identical numbers. The TAWOS phases were produced with scikit-learn 1.8; under 1.9 their random-forest and logistic-regression results are unchanged, and gradient-boosting results move by at most about .01.

## Results

| Arm | Task | n (pos.) | Model | AUC [95% CI] | vs comparator |
|---|---|---|---|---|---|
| Transfer TAWOS→MariaDB | Late closure | 391 (106) | RF, strict, no SP | .716 [.649, .776] | +.140 [.054, .218] over scope-only LR .576 |
| Transfer TAWOS→MariaDB | Late closure | 391 (106) | GB, strict, no SP | .710 [.637, .773] | +.134 [.034, .223] over scope-only LR .576 |
| Within MariaDB, time split | Late closure | 118 (39) | RF, strict | .750 [.652, .843] | +.233 [.105, .369] over scope-only LR .517 |
| Within MariaDB, time split | Late closure | 118 (39) | GB, strict | .690 [.588, .792] | −.055 [−.125, .019] vs project rate .745 |
| Within MariaDB, 50% checkpoint | Late closure | 117 (39) | GB + progress | .707 | +.019 [−.050, .089] vs no progress |
| Within MariaDB, time split | Spillover, nonempty | 106 (104) | LR, scope only | .897 | **not interpretable: 2 negatives** |

Regression (MAE, time split): issues completed 2.64 vs scope × rate 3.18; carry share .169 vs train median .200; days late 3.22 vs 3.31.

TAWOS sanity rows reproduce the paper: late closure GB .751 (vs project rate .640, scope-only .530), spillover scope-only LR .885 and all-sprint RF .986. A few gradient-boosting rows drift by up to .01 (for example GB strict spillover .817 vs .812) because scikit-learn 1.9 replaced the 1.8 used for the paper's TAWOS results.

## What it means

- **Late closure carries over.** A model trained only on TAWOS ranks MariaDB sprints at AUC .71, with no project identity and no story points. That falls between the paper's LOPO (.642) and time-split (.751) numbers. Scope alone still does little (.58), which matches the paper's claim that late closure comes from team habits rather than sprint size.
- **The project-rate caveat is sharper here.** MCOL closes late 54% of the time and MXS 15%, so knowing the project alone scores .745 within MariaDB. The learned GB doesn't beat that; the RF does slightly (.750). With only two projects this comparator is very strong, so report it plainly.
- **Progress doesn't help late closure:** +.019 [−.05, .09]. This replicates paper row 12.
- **Spillover can't be tested on this source.** 99% of nonempty MariaDB sprints leave something unfinished (median 17 committed issues, mean carry share .60). That gives 2 negatives in the time-split test set and 3 in the whole cohort, so the AUC means nothing. Part of this may be that MariaDB resolves issues when they close for release rather than when the work is done.
- **The zero-scope inflation is source-dependent.** Only 10% of MariaDB sprints are empty, against 35% in TAWOS, so the "all sprints" AUC is inflated less. This supports the paper's point that it depends on the population.

## Draft text for the paper (Threats or a short Results paragraph)

> To check that the late-closure result is not a TAWOS artifact, we ran the frozen pipeline once on two MariaDB projects outside TAWOS (MaxScale and ColumnStore; 391 sprints, 2016 to 2026). MariaDB records no story points, so this run drops the point features on both sides. Trained on TAWOS alone, with no project identity, gradient boosting ranks MariaDB late closures at AUC .710 [.637, .773] and the random forest at .716, against .576 for the scope-only baseline. Within MariaDB, under our time split, the random forest reaches .750, but the two projects' late rates differ so much (54% and 15%) that the project rate alone scores .745. Spillover could not be tested: 99% of nonempty MariaDB sprints spill over, which leaves three negatives.

## Files

- `pull_mariadb_browser.js`: the browser pull.
- `convert_mariadb.py`: converts the pull to the TAWOS CSV layout.
- `run_external.py`: runs the analysis.
- `data/pull/mariadb_mxs_mcol.json`: the raw pull.
- `data/raw/mariadb/*.csv`: the converted tables.
- `results/external_validation_results.csv`: every row, with intervals.
- `results/cohort_description.json`: cohort and reconstruction audit for both sources.
- `results/mariadb_modeling_frame.csv`: the modeled MariaDB sprints.
- `results/environment_versions.json`: Python and library versions of the recorded run.
- `results_log.txt`: console output from the Linux run (same numbers).

Rerun with `python convert_mariadb.py data/pull/mariadb_mxs_mcol.json data/raw/mariadb`, then `python run_external.py` from this folder. This folder must sit next to the week-12, phase-2 and phase-3 folders. It takes about 5 minutes.
