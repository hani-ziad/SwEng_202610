# CCECE sprint-risk paper: review record, 8 November 2026

Four passes on the same day: my data-checked review (morning), a second pass of my own, an independent blind review, and a second blind review of the rewritten paper. Every number below comes from a file in `reviewer_checks/` or from the existing result files, and `verify_numbers.py` checks each of them.

## Verdict

Not submittable as it stood this morning. The revised version is a defensible negative-result and audit paper. Expect a borderline outcome at CCECE. It would be a weak reject at a top SE or ML venue, because the novelty is thin once Choetkiertikul et al. (2018) is cited.

## Findings, ranked by rejection risk, and what was done

| # | Finding | Evidence | Fix applied |
|---|---|---|---|
| 1 | Late-closure 0.751 is mostly between-project separation | Within-project GB mean 0.634 (0.409 to 0.828); team's own past late rate 0.510 within (pooled 0.635); LR 0.492 within | Within column in Table III, row 2 in Table II, abstract and conclusion rewritten; "a team's own record ranks its next sprints" removed |
| 2 | Reconstruction is not new | Choetkiertikul et al., IEEE TSE 44(6) 2018, rebuilt iteration state from change logs at 0/30/50/80% | Cited; contribution reframed as an audit; title changed |
| 3 | Spillover label is mechanical | Committed count 0.881 vs scope-only LR 0.885; 29 test negatives (10 in Mule) | Row 8, Table III row, text |
| 4 | Spillover records rollover | 37% of committed issues resolved by planned end, 39% more than 30 days later; 44% of commitments re-commit an earlier issue; scoring at actual closure moves spillover rate by 1 point | Outcomes section and Threats |
| 5 | In-sprint model ties a plain count | Open committed count .925/.945/.954 vs GB .926/.957/.953; paired intervals all include zero (count minus GB at 50%: -0.013 [-0.034, 0.010]); count beats GB under LOPO | Table IV rows, row 14, text; old "model beats counting" claim removed |
| 6 | Size-normalised label fails too | Carry share >= 0.5 (86% positive): RF 0.615, GB 0.583, count 0.562, scope-only 0.526 | Row 9, Section IV-F |
| 7 | Timing fix does not raise AUC | Legacy content with strict timing, LOPO: LR -0.023 [-0.074, 0.023], RF -0.039 [-0.053, -0.007]; adding past lateness: LR +0.103, RF +0.081 (both CIs above 0); GB LOPO 0.632 -> 0.627 | Section IV-B rewritten |
| 8 | "Mostly on past lateness" unsupported for GB | Shapley is on RF; removing past lateness costs GB only 0.007 on the time split | Removed from abstract; attribution text limited to RF |
| 9 | Confluence Cloud resolution-date defect | 51% of its commitments lack a resolution date (3-16% elsewhere); 74% of its sprints show nothing done at planned end (Confluence Server 48%); without it spillover 0.885 -> 0.908 | Section IV-B |
| 10 | Empty sprints partly an artefact | 33% get their first issue within 1 h after start, 42% within a day | Section III-A |
| 11 | External transfer overstated | Only 2 of 8 GB intervals exclude 0.5; scope-only below 0.5 in 7 of 8; on MariaDB's own split GB 0.690 < project rate 0.745 | Section IV-D |
| 12 | Population trap understated | On the 786 test sprints the nonempty flag alone scores 0.956 and the committed count 0.989 | Section IV-A |
| 13 | Best-of-five headline | Repeats mean 0.696 +/- 0.054 | Abstract and IV-A |
| 14 | Weak comparators chosen | Spillover scope-only vs strongest model (RF): +0.045 [-0.010, 0.088]; carry share vs prior-sprint value: -0.017 [-0.029, -0.003]; days late vs project median 2.63 | Row 7 and Section IV-F use the strongest comparators |
| 15 | Fig. 1 described the old paper | Showed advisor and "calibrated intervals" | Redrawn (make_fig1.py), half the height |
| 16 | Late closure includes idle sprints | Within project, smaller scope goes with later closure (count AUC 0.422); nonempty only: GB 0.713 pooled, 0.586 within | Section IV-A and Threats |
| 17 | Statistics | 8 clusters, no multiple-comparison correction; RF seed SD 0.003; GB deterministic | Stated in III-D |
| 18 | Shapley details | 9 groups at checkpoints; in-sprint groups 61% (spillover) and 18% (late), not 6%; capacity partly restates scope | III-E and IV-G |
| 19 | Class weights | LR/RF balanced, GB not, so F1 and Brier not comparable | Stated in III-D |
| 20 | Missing citations | Hanley-McNeil, Kapoor-Narayanan (leakage), Tantithamthavorn (validation), Janzing (interventional Shapley) | Added; XGBoost and Molnar removed as unused |
| 21 | Writing | Aphoristic sentences flagged as machine-like | Whole paper rewritten in plain declarative sentences |

## Second blind review of the rewrite (evening) and what was done

| # | Finding | Evidence (rechecked by me) | Fix applied |
|---|---|---|---|
| 22 | Late-closure signal is short administrative lag | 40% of late sprints are 1-3 days late (median 4.0); planned end Fri/weekend (38% of sprints) late 57% vs 28%; >3 days: GB .727 vs project rate .705, within .623; >7 days: GB .638 vs .623, within .502 | Abstract, IV-A paragraph, Table II row 3, Threats, conclusion recommendation (3) |
| 23 | "37% of committed issues" had the wrong denominator | Among all 20,579 commitments: 32% resolved by planned end, 35% more than 30 days later, 12% never | Abstract and III-B corrected |
| 24 | MariaDB pooled 0.710 repeats the pooled-AUC flaw | Per project .699 (ColumnStore, 54% late) and .592 (MaxScale, 15% late) | Row 6 and IV-D |
| 25 | "No model exceeds 0.62" only at one threshold | Carry share >= .25: count .775 beats best model (RF .730); >= .5: RF .615, count .562 | Rows 10-11, IV-F, abstract reworded |
| 26 | "Statistically tied" is an equivalence claim | Interval -0.034 to 0.010 | "cannot be distinguished" |
| 27 | Timing section mixed two GB pipelines | Time-split GB figures came from a simplified rerun (0.747, not 0.751) | GB time-split numbers dropped; LOPO rerun stated (LR exact, RF within 0.007); legacy + past lateness RF 0.682 reported; strict minus past lateness LR .496, RF .563 |
| 28 | "Reproduces GB exactly and RF within 0.005" was false | Several models differ by up to 0.007 | Threats corrected |
| 29 | "Team" history is project history | No board/team ID in TAWOS | Wording and Threats |
| 30 | Within-project AUC had no interval | Sprint bootstrap within projects: 0.634 [0.570, 0.695]; Mesos 6 and Titanium Mobile 7 test positives | Interval added, "7 of 8" dropped |
| 31 | Overclaims on numeric targets | Days late vs train median CI includes zero; issues completed CI includes zero; carry R2 claim only on time split | Reworded |
| 32 | Explanations wording | GB is stronger for late closure but RF for spillover; nine checkpoint groups not listed; capacity is in recent history at checkpoints | Fixed; groups listed. Ablation 0.652 verified in week-12 assignment_ablation.csv (legacy RF, nonempty) |
| 33 | Fig. 1 caption too broad | Repeats and external checks use other splits | Caption corrected; "x5" now "×5" |
| 34 | Oversampling case study under-specified | Comparison is duplication on training part vs whole pool; AUC .638 vs .688 | Stated with AUC |
| 35 | Drift sentence used the wrong windows | Repeat test windows 27% to 47% late | Corrected, softened to "consistent with drift" |
| 36 | Label timing inside the training split | 29 training sprints (1.6%) labelled only after the first test sprint started | Stated in III-D |
| 37 | Data cleaning | 3 sprints under 1 day, 6 over 60 days; field names matched case-insensitively (code checked) | Stated in III-A |
| 38 | Missing related work, r3 misdescribed | Falessi et al. EMSE 2020 (verified), Choetkiertikul et al. ASE 2015 (verified); Tawosi et al. found Deep-SE did not consistently beat mean/median | Added; r3 corrected; EMSE 2017 delay paper not added because volume/pages could not be verified |
| 39 | Table hygiene | Comparators, captions, uncited rows | Table II rebuilt (14 rows, all cited); captions rewritten |
| 40 | Bibliography style | Months on some entries only; Janzing missing PMLR volume | Months removed; PMLR vol. 108 added |

Not changed, with reason: Fig. 1 kept (helps non-specialist readers, now half height); Obike et al. kept but presented as an example; Zenodo DOI and data licence wording left to you.

## Still open (need you)

1. Overleaf: paste the new tex and upload the new fig1_pipeline.png, compile, check it is 6 pages (estimated about 11 lines of slack), download the PDF.
2. Run `python ccece-2027-paper\verify_numbers.py` (full run) and check it ends with `FAILS: []`.
3. Co-authors must agree to the new title and the reframing.
4. Read Choetkiertikul et al. (2018) in full.
5. Not fixable from the data we have: planned sprint dates and issue status history are not in the TAWOS export used here, so they are stated as threats.
