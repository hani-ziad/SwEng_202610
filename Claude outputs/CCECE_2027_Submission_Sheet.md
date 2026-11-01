# CCECE 2027 — Submission Sheet (paste-ready)

40th IEEE Canadian Conference on Electrical and Computer Engineering (CCECE 2027), Toronto, May 2027.
Portal: **Sparcly** (conferences.sparcly.ai) — confirm it's the *2027* instance before entering anything; the site's "Submit Your Work" button currently still points at a 2026 path.

---

## Track
**Track 4 — Computer and Software Engineering** (Regular / full paper, 6 pages)

## Title
Sprint Risk Prediction from Reconstructed Jira Histories Using Explainable Machine Learning

## Authors (current)
1. **Hani AbuSharkh** — Higher Colleges of Technology, Abu Dhabi, UAE — habusharkh@hct.ac.ae — *corresponding & presenting author*
2. **Maryam Hanif** — United Arab Emirates University, Al Ain, UAE — 700052696@uaeu.ac.ae

*Possible addition you mentioned:* Rafat Damseh — United Arab Emirates University, Al Ain, UAE — rdamseh@uaeu.ac.ae  (he was a co-author on the earlier versions; add in the portal if you want him on this one).

## Keywords / Index Terms
Scrum; sprint analytics; Jira; machine learning; explainability; Shapley values; data leakage

## File to upload
`CCECE_2027_Sprint_Risk_Prediction.pdf` (6-page IEEE two-column).
If the portal says **double-blind**, upload the anonymized version instead (author block removed) — tell me and I'll generate it in a minute.

## Reproducibility / artifact link (if a field asks)
https://github.com/hani-ziad/SwEng_202610

---

## Abstract (full)
Scrum teams often end a sprint with committed work still open, or close it days after the planned end. We ask how well historical Jira data can predict these two outcomes before a sprint starts, and how much watching the work in progress adds. Using eight open-source projects from the TAWOS dataset, we replay the Jira change log to rebuild what each sprint looked like on its first day, which gives 2,602 modeled sprints, 1,691 of them with committed scope. At planning time the two outcomes behave differently. Spillover follows committed scope: a three-variable logistic model reaches ROC-AUC 0.885 on nonempty sprints and no richer model beats it, while the full-population figure of 0.99 is an artifact of the 35% of sprints that commit nothing. Late closure is harder and depends on team habits: gradient-boosted trees reach 0.751, and exact Shapley values attribute 60% of the model output to past lateness and project identity. Two timing problems matter more than model choice. History features that ignore parallel sprints read outcomes that were not yet known, and fixing this raises cross-project late-closure AUC from 0.54 to 0.63; incomplete change history lowers spillover AUC to 0.77 on the strictly provenanced subset. Once the sprint is running, progress features lift spillover AUC to 0.93 at the 25% checkpoint and 0.96 at 50%, but leave late closure flat. Two reproduced leakage mistakes inflate F1 by up to 0.24. A sprint advisor produces calibrated 80% intervals for issues completed, yet its what-if tables show no causal effect of scope on carryover.

## Abstract (short — use only if the portal caps the length, ~185 words)
We study whether historical Jira data can predict, before a sprint starts, two outcomes it records directly: spillover (a committed issue left unfinished at the planned end) and late closure. Using eight TAWOS open-source projects, we replay the change log to reconstruct each sprint's planning-time state, giving 2,602 sprints (1,691 with committed scope). The two outcomes behave differently: spillover is driven by committed scope (a three-variable logistic model reaches ROC-AUC 0.885 and no richer model beats it), while late closure depends on team habits (gradient boosting 0.751; exact Shapley values put 60% of the signal on past lateness and project identity). Two data-timing problems matter more than model choice — history features that ignore parallel sprints leak future outcomes (fixing this lifts cross-project late-closure AUC from 0.54 to 0.63), and incomplete change history lowers spillover AUC to 0.77 on the strictly provenanced subset. In-sprint progress raises spillover AUC to 0.96 by mid-sprint but not late closure, and two reproduced leakage mistakes inflate F1 by up to 0.24. The practical lessons are about data, not algorithms.

---

## Before you click submit
- Compile the `.tex` on Overleaf (or your TeX) to confirm it lands at **6 pages**.
- Run the final PDF through **IEEE PDF eXpress** (the conference will give the ID) for the camera-ready.
- Verify the track list and the blind-review model on the live 2027 portal — both may differ from the 2026 leftovers.
