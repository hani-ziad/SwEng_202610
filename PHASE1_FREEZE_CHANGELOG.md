# Phase 1 freeze cleanup log

This cleaned package prepares the sprint-risk prediction project for a Phase 1 freeze before Phase 2 work.

## Documentation fixes
- Restored `guides/AI-Basics-and-Reproduction-Guide.docx` from the tracked project history.
- Corrected the final article description from 7 pages to 6 pages.
- Removed references to missing `long paper` and originality-report folders from the root README.
- Corrected the Beginner Learning Guide statement about XGBoost and conditional spillover: the nonempty conditional-spillover result is strongest for scope-only logistic regression by ROC-AUC.
- Updated Week 14 analysis and presentation slides to use the standardized 2,000-resample project-cluster intervals.

## Code/comment cleanup
- Updated stale scope-decomposition comments from the old 2,002-sprint / 363-zero-scope version to the final 2,602-sprint / 911-zero-scope / 1,691-nonempty version.
- Updated reproducibility-manifest comments to the final 2,602 modeled-sprint cohort.
- Reworded XGBoost-exclusion comments in supplementary uncertainty/decomposition scripts so they no longer imply XGBoost is unavailable in the final environment.
- Standardized the validation-sensitivity cluster uncertainty call to 2,000 project-cluster resamples.

## Result synchronization
- Regenerated `cluster_uncertainty.csv` and updated the compact JSON summary using 2,000 project-cluster resamples.
- Updated the conference paper source and rebuilt the final 6-page PDF.
- Regenerated `PACKAGE_FILE_LIST.txt` from the cleaned package contents.

## Verification performed
- Rebuilt `sprint_risk_paper.pdf`; confirmed the final paper has 6 pages and rendered all pages for visual QA.
- Rendered the modified DOCX files to PNGs for layout QA.
- Exported the modified PowerPoint deck to PDF and rendered all slides for visual QA.
- Compiled the edited Python scripts successfully.
- Ran core unit tests: 14 pytest tests passed, and 10 additional leakage/load/scope tests were executed directly and passed.

## Added after the freeze (no result changes)
- Phase 1 now has a `README.md` for the CSBP711 Assignment 1 submission: data source, licence, data problems, results, explanation and ablation, how to run it, contributions and AI use.
- New script `scripts/run_assignment_comparison.py` and three new files in `results/` (`assignment_comparison.csv`, `assignment_ablation.csv`, `assignment_summary.json`). They add training time, model size and a feature-group ablation. The metrics match the saved Phase 1 numbers exactly, and no existing result file was changed.
- Phase 2 and Phase 3 were added next to Phase 1. They import it and do not change it.
- Root README, the reproduction guide (new Phase 2 and 3 sections) and the assignment deck (training time and model size on slide 4, the ablation on slide 5, the data source on slide 1) were updated.
