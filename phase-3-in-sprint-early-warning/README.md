# Phase 3: in-sprint early warning

Phases 1 and 2 predicted at planning time, before any work had started. Phase 3 asks what changes if we predict while the sprint is running. For every sprint we rebuild what a team could have seen at 25%, 50%, 75% and 100% of the planned duration, then predict the same outcomes as before:

| Outcome | Tasks |
|---|---|
| Spillover (any committed issue not done by the planned end) | `spillover` (yes/no), `carry_share` (share of committed issues not done) |
| Late closure (closed more than 1 day after the planned end) | `late_closure` (yes/no), `late_days` (days late, capped at 30) |

Phases 1 and 2 are not modified. Phase 3 imports them, so the three folders have to stay side by side.

## What is new

1. **Sprint state at a checkpoint**, rebuilt from the change log and the issue resolution dates, using only events up to that checkpoint. It includes how many of the starting issues are already done, which issues were added or removed since, how story points changed, how much time is left, and how fast the remaining work would need to go. At 100% the reconstruction matches Phase 1's committed and completed counts exactly for all 2,602 sprints (there is a test for it).
2. **A fair set of sprints at each checkpoint.** A sprint that has already closed at the checkpoint is dropped, because its outcome is known. At 100%, spillover and carry share are fully decided by the state, so only late closure is run there. The train/test assignment is Phase 2's time split, fixed per sprint, so checkpoints can be compared.
3. **Direct comparisons.** Each progress-aware model is compared with the same algorithm on planning-time features only (Phase 2's strict set), and with a no-learning rule built from progress (`1 − share of committed issues done`, or the number of committed issues still open). Intervals come from bootstrapping the 8 projects 2,000 times, under the time split and leave-one-project-out.
4. **Explanations** at the 50% checkpoint, with the same exact Shapley code as Phase 2.

## What we found

### Spillover: watching the work helps a lot, and early

Time split, gradient boosting, with leave-one-project-out in brackets.

- AUC goes from 0.83 (0.80) with planning information only to 0.93 (0.86) at 25%, 0.96 (0.90) at 50% and 0.95 (0.93) at 75%. The gain over the same model without progress is +0.10, +0.12 and +0.13, and the intervals exclude zero (except random forest at 25% across projects).
- The plain rule "share of committed issues still open" only reaches 0.67, 0.77 and 0.80, so the learned model beats it by 0.10 to 0.26. Time pressure, scope and team history add to raw progress.
- F1 is still not informative here, since 94% of sprints spill over. Always saying "spillover" gets 0.970, and progress-aware boosting gets 0.982, 0.987 and 0.990. Use AUC.
- At 50%, the explanation splits into progress so far 34%, time pressure 23%, scope 16% and recent history 15%.

### Carryover share: mostly arithmetic

- With planning information only, the average error is about 0.20 at every checkpoint. With progress it falls to 0.146, 0.112 and 0.070 at 25%, 50% and 75% (across projects: 0.162, 0.125, 0.077).
- But the simple rule "carryover is about 1 minus the share done so far" gets 0.165, 0.115 and 0.064. On the time split the learned model is no better than the rule at 50% and 75% (at 75% the rule is slightly better, +0.006, interval +0.001 to +0.012). Across projects the model does help at 25% (−0.051, interval −0.080 to −0.017) and at 50% (−0.021, interval −0.037 to −0.004).
- At 50%, progress so far accounts for 60% of the explanation.

### Late closure: progress does not help

- AUC stays flat. Gradient boosting gets 0.751 with planning information only, and 0.754, 0.760, 0.749 and 0.704 with progress at 25%, 50%, 75% and 100%. The differences are small and the intervals include zero. The one exception is random forest at 25% (+0.02, interval +0.002 to +0.049).
- The rule "number of committed issues still open" has AUC around 0.5. Open work is not what makes a sprint close late.
- F1 is beaten again by always guessing "late" across projects. On the time split the best model gets 0.56 to 0.58 against 0.48 for always-late (and 0.57 to 0.59 without progress).
- At 50%, the explanation is past lateness 37%, project 17%, recent history 12%, sprint length 9%, time pressure 9% and progress only 6%. This matches Phase 2: late closure reflects how a team runs its sprints (when it clicks "complete sprint"), not how much work is left.
- At 100% of the planned duration 65% of sprints are still open. Of the test sprints still open, 51% close more than a day late, and no model beats always-late.

### Days late

Progress does not improve the median-loss model over its planning-only version (all intervals include zero except one checkpoint across projects). Against predicting the training median, the model is better by 0.2 to 0.25 days at 50%, 75% and 100% (time split, intervals exclude zero). That is the same small edge as in Phase 2.

### The short version

The Phase 2 split holds. Spillover is a work outcome, so watching the work improves the prediction a lot. Late closure is an administrative outcome, so neither scope nor progress explains it. Team habits do.

## How to run it

```
cd phase-3-in-sprint-early-warning
pip install -r requirements.txt
python -m experiments.run_tests      # 11 tests (simple runner, no pytest needed)
python -m experiments.run_all        # about 11 minutes, writes results/
```

You can also run `run_models`, `run_explanations` and `make_figure` one by one. Results are deterministic (seed 42).

## Files

- `phase3/data.py`: checkpoint reconstruction, progress features, splits
- `experiments/`: the run scripts
- `tests/`: reconstruction (synthetic and against Phase 1), leakage and split tests
- `results/`: CSV and JSON outputs, `early_warning_curves.png`, `explanation_group_shares.png`

## Limits

Eight open-source Jira projects. The change log in our export only has Sprint and Story Points fields, so completion is read from the final issue resolution date. An issue that was resolved, reopened and resolved again counts at its last resolution. That can only make early progress look slower, never leak the future. Sprint closure is a human decision, which limits what any model can say about late closure. Shapley values describe the models, not causes. Checkpoints are fractions of the planned duration, and sprints that already closed are left out, so later checkpoints cover a slightly different set of sprints (98.6% are still open at 25%, 96% at 75% and 65% at 100%).
