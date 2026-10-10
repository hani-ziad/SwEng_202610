# Week 16 — Revised Conference Paper

## Paper

**Title:** Predicting, Quantifying, and Explaining Sprint Outcomes from Reconstructed Jira Histories

**Authors:** Hani AbuSharkh (HCT), Maryam Hanif (UAEU), Rafat Damseh (UAEU)

**Format:** IEEE conference (6 pages), LaTeX source using `IEEEtran.cls`

**File:** `sprint_risk_paper_revised.tex`

This is the revised version of the paper, incorporating Phase 2 results (Shapley explanations, numeric outcomes, sprint advisor) alongside Phase 1's binary classification. It addresses all reviewer feedback from the first submission.

## What changed from the original paper

1. **Expanded scope.** The original paper covered only two binary outcomes (spillover and late closure). The revised paper adds three numeric outcomes (issues done, carry share, late days), exact Shapley explanations, and the sprint advisor — all from Phase 2.
2. **Stricter history features.** The parallel-sprint timing fix is now the default throughout, with a dedicated subsection explaining why it matters.
3. **Clearer research questions.** Three explicit RQs replace the original implicit framing.
4. **Expanded related work.** Added paragraphs on Jira-mining studies, Shapley-based ML explanations, and sprint-scope advising.
5. **Bootstrap table reworked.** Table II now reports delta-AUC against the scope-only baseline (strict features only), not absolute AUC ranges mixing legacy and strict.
6. **Tightened numbers.** Days-late MAE range corrected to 3.7–4.1 (strict features only).
7. **Repository link.** Added a GitHub URL in the conclusion.

## How to compile

The paper requires `IEEEtran.cls`, which is standard in TeX Live and MiKTeX. Either:

```bash
pdflatex sprint_risk_paper_revised.tex
pdflatex sprint_risk_paper_revised.tex   # run twice for references
```

Or upload the `.tex` file to [Overleaf](https://www.overleaf.com) and compile there.

## Previous files (removed)

The following files from the original submission were removed because they are superseded by the revised paper:

- `sprint_risk_paper.tex` / `sprint_risk_paper.pdf` (original paper)
- `sprint_risk_presentation.pptx` (original presentation)
- `README.docx` / `README.txt` (original READMEs)
- `calibration_reliability.pdf/.png`, `dataflow_timing.pdf/.png`, `lopo_auc_by_project.pdf/.png` (figures that were embedded in the original paper — the revised paper has no external figure files)
