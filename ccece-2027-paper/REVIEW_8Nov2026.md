# CCECE sprint-risk paper: hostile review and fixes (8 Nov 2026)

Checked against the data, not the text. Scripts and result files are in `reviewer_checks/`.

## What a strong reviewer would have rejected on

1. **Novelty claim was false.** Choetkiertikul et al., IEEE TSE 44(6), 2018, "Predicting delivery capability in iterative software development" already rebuilt iteration state at prediction time from JIRA change logs, at 0/30/50/80% of the iteration, on 3,834 iterations from five projects including MongoDB. The paper said nobody had done this. A reviewer from that group would end the review there. Fixed: cited, Table I row added, contribution moved to provenance accounting, overlap timing, labels, rule comparisons and leakage reproductions.

2. **Spillover result is arithmetic.** The raw committed issue count (no model) scores AUC 0.881 against the scope-only model's 0.885. The label asks "did any of n issues slip", so big n wins. The test set has 29 negatives. Fixed: said plainly, row 19 and a Table III line added.

3. **The in-sprint "model beats counting" claim was wrong.** The text said the model was worth 0.15 to 0.26 AUC "over a bare count of open issues", but the rule actually used was one minus share done. The real count of open committed issues scores 0.925 / 0.945 / 0.954 on the time split against the model's 0.926 / 0.957 / 0.953, and beats the model under LOPO (0.876 / 0.902 / 0.938 vs 0.860 / 0.897 / 0.927). Contribution 3 does not survive. Fixed: rule added to Table III, text and conclusion rewritten, the "earlier and better than counting" sentence removed.

4. **The timing-fix gain was confounded.** Strict and legacy differ in timing and in content (strict adds concurrency, capacity and past lateness). Same content, strict timing: LOPO late-closure AUC drops 0.542 to 0.519 (LR) and 0.601 to 0.563 (RF). Legacy plus past lateness: 0.645 and 0.682. So the 0.542 to 0.629 lift in the abstract came from past lateness, not the fix. That is actually a cleaner leakage story (fixing a leak costs AUC), so it now reads that way.

5. **Confluence Cloud defect hits the progress features too, and maybe Confluence Server.** 74% of Cloud's nonempty sprints show zero committed issues done at the planned end; Confluence Server is 48%; the rest 11 to 31%. Both the spillover label and the in-sprint features read the same broken field. Dropping Cloud gives scope-only spillover 0.908 (the handover note said 0.903; that figure has no result file I could find, so the paper uses 0.908 from `planning_checks.csv`).

6. **"Zero-scope" sprints are partly an artefact.** Of the 911, 33% get their first issue within an hour after Start and 42% within a day. Teams press Start then load the board. Added to Methods and Threats.

7. **Outside-project claim was overstated.** "Above chance in seven of eight" is point estimates. Only 2 of 8 intervals clear 0.5 (ColumnStore, PHP driver). The scope-only comparator averages 0.362, below chance, so "beats scope-only in all eight" was empty. Also the text said 20 to 40 sprints per project; real range is 21 to 269. Fixed.

8. **Headline 0.751 is the best of five cuts.** The repeats average 0.696. Abstract and results now give both and call 0.70 the fair figure.

## Also changed
- Abstract rewritten around the findings above.
- Advisor and what-if trimmed to one sentence on scope cutting (it was untested and ate space).
- Cluster-bootstrap caveat (8 clusters undercover).
- Shapley section notes it explains the forests, not the best model.

## Not done, needs you
1. **Page count.** IEEEtran is not installed here and CTAN is blocked, so I could not check the 6-page limit. The tex compiles cleanly with a stand-in class. Paste into Overleaf and check. If over, cut the numeric-targets subsection first.
2. **verify_numbers.py** needs checks for the new numbers (row 19, 20, Table III rule rows, Section IV-B rerun, 74%/48%, 33%/42%, 2 of 8). They are all in `reviewer_checks/*.csv`.
3. **Port the timing rerun into phase2/run_models.py** so it comes from the main pipeline. My rerun is within 0.007 of it but is a separate implementation.
4. **Read Choetkiertikul 2018 in full** before submission so the description of it is fair.
5. **Bigger call:** spillover as "any issue left" is not worth a headline. Replace it with carry share or a size-normalised threshold, at least in the journal version.
6. Word copy (.docx) is now stale.
7. Maryam's address is a student-ID email. Fine for review, odd in print.
