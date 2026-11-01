# CCECE 2027 Submission Sheet (paste-ready, updated 1 Nov 2026)

40th IEEE Canadian Conference on Electrical and Computer Engineering (CCECE 2027), Toronto, May 2027.
Portal: **Sparcly** (conferences.sparcly.ai). Confirm it's the *2027* instance before entering anything; the site's "Submit Your Work" button currently still points at a 2026 path.

---

## Track
**Track 4: Computer and Software Engineering** (Regular / full paper, 6 pages)

## Title
Reconstructing Sprint-Start Jira State for Leakage-Aware Outcome Prediction

## Authors
1. **Hani AbuSharkh**, Higher Colleges of Technology, Al Ain, UAE, habusharkh@hct.ac.ae (corresponding and presenting author)
2. **Maryam Hanif**, United Arab Emirates University, Al Ain, UAE, 700052696@uaeu.ac.ae
3. **Rafat Damseh**, United Arab Emirates University, Al Ain, UAE, rdamseh@uaeu.ac.ae

## Keywords / Index Terms
Scrum; sprint analytics; Jira; machine learning; explainability; Shapley values; data leakage

## File to upload
`CCECE_2027_Sprint_Risk_Prediction.pdf` (6-page IEEE two-column).
If the portal says **double-blind**, upload the anonymized version instead (author block removed); tell me and I'll generate it in a minute.

## Reproducibility / artifact link (if a field asks)
https://github.com/hani-ziad/SwEng_202610

---

## Abstract (full, same as the paper)
To predict how a sprint will end, you first have to know what the team knew when it began, and a present-day Jira export does not tell you. For eight open-source TAWOS projects we replay the change log back to the first day of each sprint and rebuild the board as it stood: 2,602 sprints, 1,691 of them with committed work. On a time-ordered split, a scope-only logistic regression ranks spillover among those 1,691 at ROC-AUC 0.885; on the 110 test sprints (94 positive) whose committed scope has fully recorded history it reaches 0.772, so the main estimate is not yet established for data free of static fallbacks. Late closure is harder: gradient boosting reaches 0.751, and trained on TAWOS alone it still ranks late closure on two MariaDB projects outside TAWOS at 0.710. Dropping unfinished prior sprints from the history features lifts leave-one-project-out late-closure AUC for logistic regression from 0.542 to 0.629. Once a sprint is live, progress features lift spillover AUC from 0.830 to 0.926 at the 25% checkpoint and 0.957 at 50%, while late closure barely changes. Two reproduced leakage mistakes inflate F1 by up to 0.24. What we offer is a change-log reconstruction and a leakage-aware way of evaluating on it, not a deployable risk tool: results vary across projects, the prediction intervals under-cover large sprints, and model-based what-if tables say nothing about cause.

## Abstract (short, use only if the portal caps the length, about 135 words)
A present-day Jira export does not show what a team knew when a sprint began. For eight open-source TAWOS projects we replay the change log to rebuild each sprint's first-day state (2,602 sprints, 1,691 with committed work). On a time-ordered split, a scope-only logistic regression ranks spillover at ROC-AUC 0.885, and 0.772 on the 110 test sprints whose scope has fully recorded history. Gradient boosting ranks late closure at 0.751, and trained on TAWOS alone it reaches 0.710 on two MariaDB projects outside TAWOS. Excluding unfinished prior sprints from history features lifts cross-project late-closure AUC from 0.542 to 0.629. Mid-sprint progress lifts spillover AUC to 0.957 but leaves late closure flat, and two reproduced leakage mistakes inflate F1 by up to 0.24. We offer a reconstruction and a leakage-aware evaluation, not a deployable risk tool.

---

## Before you click submit
- Compile the `.tex` on Overleaf (or your TeX) to confirm it lands at **6 pages**.
- Run the final PDF through **IEEE PDF eXpress** (the conference will give the ID) for the camera-ready.
- Verify the track list and the blind-review model on the live 2027 portal, since both may differ from the 2026 leftovers.
