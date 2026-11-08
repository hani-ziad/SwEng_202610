# CCECE 2027 Submission Sheet (paste-ready, updated 8 Nov 2026 to match the rewritten paper)

40th IEEE Canadian Conference on Electrical and Computer Engineering (CCECE 2027), Toronto, May 2027.
Portal: **Sparcly** (conferences.sparcly.ai). Confirm it's the *2027* instance before entering anything; the site's "Submit Your Work" button currently still points at a 2026 path.

---

## Track
**Track 4: Computer and Software Engineering** (Regular / full paper, 6 pages)

## Title
Predicting Sprint Outcomes from Reconstructed Jira State: Count Baselines and a Leakage Audit

(New title from the 8 Nov rewrite. Maryam and Rafat must agree to it before submission. Old title: "Reconstructing Sprint-Start Jira State for Leakage-Aware Outcome Prediction".)

## Authors
1. **Hani AbuSharkh**, Higher Colleges of Technology, Al Ain, UAE, habusharkh@hct.ac.ae (corresponding and presenting author)
2. **Maryam Hanif**, United Arab Emirates University, Al Ain, UAE, 700052696@uaeu.ac.ae
3. **Rafat Damseh**, United Arab Emirates University, Al Ain, UAE, rdamseh@uaeu.ac.ae

## Keywords / Index Terms
Scrum; sprint analytics; Jira; machine learning; data leakage; baselines

## File to upload
`CCECE_2027_Sprint_Risk_Prediction.pdf` (6-page IEEE two-column).
If the portal says **double-blind**, upload the anonymized version instead (author block removed); tell me and I'll generate it in a minute.

## Reproducibility / artifact link (if a field asks)
https://github.com/hani-ziad/SwEng_202610

---

## Abstract (full, same as the paper)
A present-day Jira export shows how a sprint ended, not what the team knew when it began, so sprint-risk models trained on it can learn from hindsight. We replay the change log of eight TAWOS projects to each sprint's start (2,602 sprints, 1,691 with committed work) and log where missing history forces a static fallback. We then test planning-time and in-sprint classifiers for late closure and spillover against learning-free baselines. Gradient boosting (GB) ranks late closure at a pooled ROC-AUC of 0.751 on one time-ordered split and 0.696 ± 0.054 over five. Much of this separates projects. Within a project GB averages 0.634 (95% interval 0.570 to 0.695), and the project's own recent late rate ranks its next sprints at chance (0.510). The remaining signal sits in short administrative lags: 40% of late sprints close one to three days late, and with a seven-day threshold the within-project AUC falls to 0.502. Transfer to ten MariaDB and MongoDB projects is weak. Spillover, meaning any committed issue unfinished at the planned end, is close to mechanical. Only 32% of commitments are resolved by the planned end, the number of committed issues ranks spillover at 0.881, and at mid-sprint the number still open (0.945) cannot be distinguished from the best model (0.957). We offer the results as an audit protocol for sprint analytics.

## Abstract (short, use only if the portal caps the length, about 135 words)
A present-day Jira export shows how a sprint ended, not what the team knew when it began. We replay the change log of eight TAWOS projects to each sprint's start (2,602 sprints) and test sprint-risk classifiers against learning-free baselines. Gradient boosting ranks late closure at 0.751 pooled, but 0.634 within a project, where the project's own recent late rate is at chance. Most of the signal is short administrative lag: 40% of late sprints close one to three days late, and at a seven-day threshold the within-project AUC is 0.502. Spillover is close to mechanical: counting committed issues ranks it at 0.881, and at mid-sprint the count still open (0.945) cannot be told apart from the best model (0.957). Transfer to ten outside projects is weak. We offer an audit protocol, not a risk tool.

---

## Before you click submit
- Compile the `.tex` on Overleaf (or your TeX) to confirm it lands at **6 pages**.
- Run the final PDF through **IEEE PDF eXpress** (the conference will give the ID) for the camera-ready.
- Verify the track list and the blind-review model on the live 2027 portal, since both may differ from the 2026 leftovers.
