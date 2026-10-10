# Phase 2: explainable sprint advisor

Phase 1 asked two yes/no questions about a sprint: will it spill over (at least one committed issue not done by the planned end), and will it close late (more than a day after the planned end)? Phase 2 goes further. It asks how much, explains the predictions, and turns them into a small what-if tool.

| | Spillover side (1,691 nonempty sprints) | Late-closure side (all 2,602 sprints) |
|---|---|---|
| Yes/no (same as Phase 1) | `spillover` | `late_closure` |
| How much? | `carry_share` (share of committed issues not done), `issues_done` (how many got done) | `late_days` (days closed after the planned end, 0 if on time, capped at 30) |

It reuses the frozen Phase 1 pipeline as it is: scope rebuilt from the change log, closed sprints only, the same cold-start handling, the same per-project 70/30 time split and the same leave-one-project-out check. Phase 1 is not touched.

## What is new

1. **Stricter history features.** Projects often run several sprints at once (on average 1.9 other sprints are already running when a sprint starts, up to 11). In 71% of cases a sprint starts before the previous one is supposed to end. Phase 1's history features ignore this, so they can accidentally use outcomes nobody knew yet at planning time. The strict features only use sprints whose outcome was already in the past on the start date. They also add the number of parallel sprints, the team's recent capacity, scope relative to capacity and past lateness. Phase 1's features are kept as the `legacy` set so we can measure the difference.
2. **Numeric targets** next to the yes/no ones, with learned models compared against simple no-learning baselines, under the time split and leave-one-project-out, with bootstrap intervals over projects (2,000 resamples).
3. **Explanations.** Exact Shapley values per feature group (our own code in `phase2/shapley.py`, tested against closed-form cases) for five models.
4. **A sprint advisor.** For a planned sprint it gives the expected issues done with an 80% interval, expected carryover, the chance of any spillover, the chance of late closure, the median and 90th percentile of days late, and a what-if table for different scopes.

## What we found (strict features unless noted)

### Spillover side

- The yes/no spillover label is mostly the base rate (94% positive in the test set). Always guessing "spillover" gets F1 0.971 and the best learned model gets 0.974. Logistic regression on scope alone has the best AUC (0.885) and beats every richer model by 0.06 to 0.07. History adds nothing for ranking here.
- `issues_done` is mostly arithmetic. Scope times the team's usual completion rate gets R² 0.42 and an average error of 2.10 issues. The best learned model (a gradient booster trained on median loss) gets 1.96, only slightly better (difference −0.13, interval −0.31 to +0.00). Across projects, ridge regression beats the baseline by 0.10 issues (interval −0.19 to −0.00). Scope accounts for 71% of the explanation.
- `carry_share` is close to unpredictable (R² around 0 for everything). The median-loss booster still trims the error a bit against just predicting the training median: 0.201 vs 0.232 on the time split and 0.210 vs 0.244 across projects.

### Late-closure side

- Late closure is heavy-tailed. The median sprint closes 0.21 days late, the 90th percentile is 10 days, and the worst is 2,603 days. 34.5% of sprints close before the planned end. 38.7% are late on the full cohort, and 31.3% in the time-split test set.
- It depends on team habits, not on scope. On the yes/no label, AUC is 0.68 to 0.75 on the time split and 0.63 to 0.64 across projects, against 0.53 for a scope-only model. Against a project-base-rate predictor the gain is smaller (+0.04 to +0.11, and only gradient boosting excludes zero). Scope is only 9% of the explanation, while past lateness (38%) and project (22%) dominate.
- F1 is misleading again. On the time split the best model gets 0.553 against 0.477 for always guessing late. Across projects, always guessing late (0.558) beats every model.
- For `late_days`, models trained on squared error are worse than just predicting the median (average error 3.7 to 4.3 vs 2.66), because a few extreme sprints throw them off. The median-loss booster gets 2.48 on the time split (not significant) and 3.13 vs 3.30 across projects. R² is about 0 for all of them.
- The late-closure probability is well calibrated on the time split (top quartile: predicted 0.66, observed 0.66) but not across projects (bottom quartile: predicted 0.16, observed 0.30).

### The history-timing fix

Switching from legacy to strict history features lowers the error for every regression model and protocol (for example carry share across projects: 0.227 to 0.210). Late-closure AUC goes up for most models (across projects, logistic regression 0.542 to 0.629 and random forest 0.600 to 0.642; time split, gradient boosting 0.726 to 0.751). Spillover AUC stays the same or drops by up to about 0.02, which is within noise. No headline result flips, but the legacy history features are less useful than they look and should not be described as "known at planning time".

### The advisor

- The 80% intervals for issues done cover 0.84 of outcomes on the time split and 0.85 across projects, and are about a third narrower than the scope-times-rate baseline (width 5.4 vs 8.4, and 6.6 vs 9.5). Coverage drops for sprints with 13 or more issues (0.70 and 0.80).
- The 90th percentile forecast of days late is exceeded about 10% of the time, as it should be (0.90 and 0.89), with lower pinball loss than global quantiles.
- **Do not treat the what-if table as advice.** Sprints planned heavy relative to team capacity carry over only 3.2 points more than the lightest quartile (interval −2.3 to +8.4) and close late 5.6 points more often (interval −5.2 to +16.4). Inside single projects the pattern is mixed (carryover goes up in 2 of 8 projects, late closure in 5 of 8). The spillover rate does rise with load (0.83 to 0.99), but that is largely arithmetic. The late-closure chance is not monotone in scope either (in the worked Lsstcorp example it falls from 0.56 to 0.31 as scope grows). The table shows what the fitted model expects, not what would happen if you cut scope.

## How to run it

```
cd phase-2-explainable-sprint-advisor
pip install -r requirements.txt
python -m experiments.run_tests      # 15 tests (simple runner, no pytest needed)
python -m experiments.run_all        # about 6 minutes, writes results/
```

You can also run the steps one by one: `run_models`, `run_explanations`, `run_advisor`, `make_figure`. Phase 2 imports the Phase 1 loader, labels and features, so keep `../week-12-method-preliminary-results/sprint-risk-prediction` where it is. Results are deterministic (seed 42).

## Files

- `phase2/data.py`: cohort, both outcomes, strict and legacy features, splits
- `phase2/models.py`: the five tasks, models, baselines, metrics
- `phase2/shapley.py`: exact Shapley values per feature group
- `phase2/advisor.py`: `SprintAdvisor` (forecasts and what-if)
- `experiments/`: the run scripts
- `tests/`: timing, Shapley and advisor tests
- `results/`: CSV and JSON outputs and `explanation_group_shares.png`

## Limits

Eight open-source Jira projects, observational data only. Late closure depends on teams actually clicking "complete sprint", which is why it is so heavy-tailed. Shapley values describe the fitted models, not causes. The 80% intervals are too narrow for very large sprints.
