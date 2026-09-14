"""
One-time backfill: seeds results/experiment_log.csv with the two runs
already on record before this logging mechanism existed --

  1. The Week 10 baseline (Koralage substitute dataset, 4 projects,
     238 sprints, 3 models -- no XGBoost, not available in the Week 10 environment), read
     from the Week 10 stage's own results/baseline_results.json.
  2. The Week 12 baseline (real TAWOS dataset, 8 projects, 2,657 sprints,
     4 models including XGBoost), read from this stage's
     results/baseline_results.json.

Going forward, scripts/run_baseline.py appends new rows itself via
scripts/log_experiment.py -- this script only needs to run once, and is
kept for anyone who wants to regenerate the log from scratch.

Usage:
    python -m scripts.backfill_experiment_log
"""
from __future__ import annotations

import json
from pathlib import Path

from scripts.log_experiment import LOG_PATH, log_run

REPO_ROOT = Path(__file__).resolve().parents[1]
WEEK_10_RESULTS = REPO_ROOT.parents[1] / "week-10-baseline-artifact-plan" / "sprint-risk-prediction" / "results" / "baseline_results.json"
WEEK_12_RESULTS = REPO_ROOT / "results" / "baseline_results.json"


def main() -> None:
    if LOG_PATH.exists():
        LOG_PATH.unlink()

    if WEEK_10_RESULTS.exists():
        week_10 = json.loads(WEEK_10_RESULTS.read_text())
        log_run(
            week_10,
            dataset="koralage",
            n_projects=4,
            n_sprints=238,
            notes="Week 10 stage (recorded) -- no XGBoost, not available in the Week 10 environment",
            timestamp="2026 W10 stage",
        )
        print(f"Recorded Week 10 (koralage) run from {WEEK_10_RESULTS}")
    else:
        print(f"Skipped Week 10 backfill -- {WEEK_10_RESULTS} not found")

    week_12 = json.loads(WEEK_12_RESULTS.read_text())
    log_run(
        week_12,
        dataset="tawos",
        n_projects=8,
        n_sprints=2657,
        notes="Week 12 stage (recorded from the author's real TAWOS run)",
        timestamp="2026 W12 stage",
    )
    print(f"Recorded Week 12 (tawos) run from {WEEK_12_RESULTS}")
    print(f"\nWrote {LOG_PATH}")


if __name__ == "__main__":
    main()
