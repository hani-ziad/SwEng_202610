# Leakage case studies: what "originality" this project claims

This document is the evidence behind a specific claim used to answer
"what's your contribution" for this project: **published sprint-risk /
sprint-performance results in this space report unusually high accuracy,
and at least two of them have a documented, plausible methodological
explanation for why — not because sprint outcomes are actually that
predictable.** Rather than asserting that abstractly, this project
verified it against the papers' own text and then reproduced the
*mechanism* on its own data and pipeline, with real before/after numbers.

## What was verified (not just suspected)

### Perez Castillo et al. (2024, MDPI *Information* 15(11):726)

Their own Table 1 defines all 11 features (R1–R11) used to classify
sprint performance (Poor/Fair/Good/Excellent). Feature R1 is stated as:

> "Percentage of work completed by the end of the Sprint"

...and every one of R1–R11 is, per the paper's own description, extracted
from the sprint's completed burndown chart *after* the sprint ran. That
means their ~0.87 accuracy is a retrospective classification of already-
finished sprints using features derived from their own outcome — a much
easier task than predicting an upcoming sprint's risk before it starts,
which is the task this project (and any Scrum Master actually planning a
sprint) cares about. The paper also does not state whether its train/test
split was random or time-ordered.

### Obike, Ekong & Obot (2025, IJCA 187(49))

Their methodology section states SMOTE was applied to grow 969 PROMISE
instances into "5328 samples with balanced representation," with no
train/test split described before that step. Applying an oversampling
technique to the *whole* dataset before splitting is a well-documented
leakage bug: (near-)duplicate synthetic points can end up on both sides
of the split, so the model is partly being tested on data it has
effectively already seen. Their COQUINA dataset also has no ground-truth
"deliverability" label at all — it's generated via "ensemble
pseudo-labeling" from the same pipeline being evaluated, which is
circular. This is a documented, plausible (not certain, since the paper
doesn't fully specify split methodology) explanation for the reported
AUC of 0.9995.

### Almalki (2025, MDPI *Systems* 13(3):208)

Checked and found *not verifiable either way*: the paper states only an
"80–10–10% data split" with no mention of whether it's random or
time-ordered, gives no exhaustive feature list, and never defines what
counts as a positive risk label. The reported 94% accuracy cannot be
reproduced or audited from what's published. This is reported honestly
as "insufficiently documented to trust or dismiss," not as a confirmed
leak — an important distinction from the two cases above.

## Reproducing the mechanisms on this project's own data

Rather than stopping at "their paper probably has a bug," both
mechanisms were implemented and run on this project's real pipeline
(`src/leakage_case_studies.py`, `scripts/run_leakage_case_studies.py`) —
so the effect is demonstrated with numbers, not just argued.

### Case study A: post-outcome feature (the Perez Castillo mechanism)

Adds this project's own `completion_ratio` (the sprint's own, already-
known-at-the-end completion percentage — the direct analogue of their
R1) into the feature set, and compares against the leakage-safe set:

| Label | Feature set | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| spillover | leakage-safe | 0.917 | 0.846 | 0.880 | 0.758 |
| spillover | + completion_ratio (leaky) | 1.000 | 0.923 | 0.960 | **0.991** |
| delay | leakage-safe | 0.154 | 0.400 | 0.222 | 0.408 |
| delay | + completion_ratio (leaky) | 0.154 | 0.400 | 0.222 | 0.406 |

For **spillover**, adding one post-outcome feature pushes ROC-AUC from
0.758 to 0.991 — right in the range of the suspiciously perfect scores
reported in the literature — because spillover is *defined* partly from
completion ratio, so this is close to leaking the label into the
features directly. For **delay**, the same feature changes almost
nothing, because delay is defined from dates, not completion ratio — the
leak only inflates results when the leaked feature is causally close to
how the label itself was defined. That's a real, useful, and non-obvious
finding in its own right: leakage risk isn't uniform across label
definitions, it depends on how correlated the "leaked" signal is with
the specific label.

### Case study B: oversample-before-split (the Obike et al. mechanism)

Compares oversampling the minority class correctly (after the train/test
split, on the training fold only) against oversampling the *whole* pool
first and splitting afterward:

| Label | Order | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| delay | correct (oversample train only) | 0.160 | 0.400 | 0.229 | 0.421 |
| delay | incorrect (oversample pool, then split) | 0.872 | 0.683 | 0.766 | **0.733** |
| spillover | correct (oversample train only) | 0.903 | 1.000 | 0.949 | 0.664 |
| spillover | incorrect (oversample pool, then split) | 0.901 | 1.000 | 0.948 | 0.656 |

For **delay**, getting the order wrong nearly doubles F1 (0.229 → 0.766)
and moves ROC-AUC from barely-better-than-chance to a respectable-looking
0.733 — using the exact same model and features, changing only *when*
oversampling happens relative to the split. For **spillover**, it barely
matters, because spillover's minority class is already the *majority*
(89% positive), so there's little class imbalance to (mis)correct in the
first place.

## What this project claims, precisely

Not "those papers are wrong" (their reported numbers may be exactly what
their described pipeline produces) and not "sprint risk is unpredictable"
(spillover clearly has real signal: leakage-safe ROC-AUC 0.758 is a solid
result on its own). The claim is narrower and defensible: **a meaningful
share of the very high accuracy reported in this specific literature is
consistent with, and in two cases explicitly traceable to, common and
avoidable leakage mechanisms — and this project demonstrates, on its own
data, exactly how much those mechanisms can inflate results, and under
what conditions.**

Reproduce these results yourself with:
```bash
python -m scripts.run_leakage_case_studies
```
