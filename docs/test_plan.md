# Test plan

Checkpoint 2 (Week 10) asks for a test plan alongside the working
baseline. This covers what is tested now, how to run it, and what's
deferred to later checkpoints as the modeling work grows.

## What is tested now (15 tests, all passing)

| File | Covers |
|---|---|
| `tests/test_load_koralage.py` | Data loading: all 4 projects present, dates parse, sprint ids unique, issue-to-sprint references resolve. Catches silently-broken ingestion (e.g. a column rename that doesn't match one project's CSV headers -- this actually happened once during development and was caught by this suite, see "Meso/Spring XD use `totalNumberOfIssues` instead of `total`" in the loader code). |
| `tests/test_labels.py` | Label construction logic: on-time/complete sprints are not flagged, late completion triggers the delay label at the right threshold (and *not* at exactly the threshold), explicit carryover and low completion ratio both trigger spillover, division-by-zero on an empty sprint does not crash. |
| `tests/test_features.py` | Feature engineering: the first sprint of a project (no history) is dropped rather than silently given zero/NaN history; rolling averages use *strictly prior* sprints, not the current one; one project's history never leaks into another's. |
| `tests/test_no_leakage.py` | The single most important test for this project's validity: asserts the feature list contains none of the outcome columns (`completed_issue_count`, `delivered_story_points`, the label columns, etc.). This is checked structurally against the feature list itself, so it keeps failing (loudly) if a future edit accidentally adds an outcome column as a "feature" -- exactly the mistake Section 6 (Threats to Validity) of the proposal flags as the main internal-validity risk. |

**Run them:**
```bash
pip install -r requirements.txt
pytest tests/ -v
```
(The sandbox this checkpoint was built in could not install pytest itself
-- no reachable package index, see `docs/data_access_and_versioning.md` --
so `scripts/run_tests.py` is a dependency-free runner that executes the
exact same test functions with only the standard library, used to produce
the 15/15 passing result reported for this checkpoint. Once you have
`pytest` installed normally, prefer it — it gives better output, and CI
will typically use it directly.)

## Planned additions for later checkpoints

- **Checkpoint 3 (method + preliminary results):** a regression test that
  pins the current baseline numbers (`results/time_ordered_summary.csv`)
  so an unintentional pipeline change is caught by a metric moving
  outside a tolerance band, not just by eyeballing a printout.
- **Checkpoint 3-4:** tests for the TAWOS loader (`test_load_tawos.py`)
  once real data is available, mirroring `test_load_koralage.py`'s
  structure (project coverage, date parsing, id uniqueness) plus a test
  that `derive_sprint_aggregates` agrees with a hand-computed example.
- **Checkpoint 4 (full results/ablations):** property-based tests on the
  ablation grid (e.g. every feature-subset configuration produces a valid
  probability output in `[0, 1]`) and a fairness/robustness check across
  projects (no single project should dominate reported aggregate metrics
  -- relevant given the class-imbalance and small-N-projects threats
  already flagged in the proposal).

## What is intentionally *not* unit-tested

Model quality itself (whether F1/ROC-AUC are "good") is a research
question, not a test assertion -- that's what Sections 5 and the later
checkpoints' "full results and analysis" are for. Tests here check that
the pipeline computes what it claims to compute, not that the resulting
numbers are impressive.
