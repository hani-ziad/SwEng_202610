"""
Minimal experiment tracking, satisfying the Week 12 syllabus requirement
("...with experiment tracking") without pulling in a tool like MLflow or
Weights & Biases, which would be overkill at this scale.

What it does: every time scripts/run_baseline.py finishes a run, it calls
log_run() here, which appends one row per (label, validation_scheme, model)
to results/experiment_log.csv -- date, git commit, dataset identity and
size, and the metrics -- instead of only keeping the latest run's outcome
(results/baseline_results.json, which is overwritten every run).

This means results/experiment_log.csv is a running history across runs and
datasets, while results/baseline_results.json stays what it always was:
the full detail of the most recent run.

Usage: imported and called from scripts/run_baseline.py's main(). Can also
be invoked directly to backfill a run from an existing results JSON (see
scripts/backfill_experiment_log.py, used once to seed this log with the
Week 10 and Week 12 runs already on record).
"""
from __future__ import annotations

import csv
import subprocess
from datetime import datetime, timezone
from pathlib import Path

LOG_PATH = Path(__file__).resolve().parents[1] / "results" / "experiment_log.csv"
FIELDNAMES = [
    "timestamp",
    "git_commit",
    "dataset",
    "n_projects",
    "n_sprints",
    "label",
    "validation_scheme",
    "model",
    "n_test_or_held_out",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
    "notes",
]


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip() if out.returncode == 0 else "no-git"
    except Exception:
        return "no-git"


def _is_nan(x) -> bool:
    return x != x  # NaN != NaN is the cheapest NaN check without importing math/numpy here


def _round_or_blank(x):
    return round(x, 4) if x is not None and not _is_nan(x) else ""


def _mean_metrics(metrics_list: list[dict]) -> dict:
    out = {}
    for key in ["precision", "recall", "f1", "roc_auc", "pr_auc"]:
        vals = [m[key] for m in metrics_list if m.get(key) is not None and not _is_nan(m[key])]
        out[key] = sum(vals) / len(vals) if vals else float("nan")
    return out


def _row(timestamp, commit, dataset, n_projects, n_sprints, label, scheme, model, m, notes, n_override=None):
    return {
        "timestamp": timestamp,
        "git_commit": commit,
        "dataset": dataset,
        "n_projects": n_projects,
        "n_sprints": n_sprints,
        "label": label,
        "validation_scheme": scheme,
        "model": model,
        "n_test_or_held_out": n_override if n_override is not None else m.get("n", ""),
        "precision": _round_or_blank(m.get("precision")),
        "recall": _round_or_blank(m.get("recall")),
        "f1": _round_or_blank(m.get("f1")),
        "roc_auc": _round_or_blank(m.get("roc_auc")),
        "pr_auc": _round_or_blank(m.get("pr_auc")),
        "notes": notes,
    }


def log_run(
    all_results: dict,
    dataset: str,
    n_projects: int,
    n_sprints: int,
    notes: str = "",
    timestamp: str | None = None,
) -> Path:
    """
    Append one row per (label, validation_scheme, model) from a
    run_baseline.py-shaped results dict to results/experiment_log.csv.

    Leave-one-project-out results are logged as their mean across held-out
    projects (one row per model per label), matching
    results/leave_one_project_out_summary.csv, rather than one row per
    project -- the per-project detail already lives in
    results/baseline_results.json.
    """
    LOG_PATH.parent.mkdir(exist_ok=True)
    is_new = not LOG_PATH.exists()
    timestamp = timestamp or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    commit = _git_commit()

    rows = []
    for label, schemes in all_results.items():
        for model, m in schemes.get("time_ordered", {}).items():
            rows.append(
                _row(timestamp, commit, dataset, n_projects, n_sprints, label, "time_ordered", model, m, notes)
            )

        lopo = schemes.get("leave_one_project_out", {})
        if lopo:
            model_names: set[str] = set()
            for per_model in lopo.values():
                model_names.update(per_model.keys())
            for model in sorted(model_names):
                metrics_list = [per_model[model] for per_model in lopo.values() if model in per_model]
                mean_m = _mean_metrics(metrics_list)
                rows.append(
                    _row(
                        timestamp,
                        commit,
                        dataset,
                        n_projects,
                        n_sprints,
                        label,
                        "leave_one_project_out_mean",
                        model,
                        mean_m,
                        notes,
                        n_override=len(metrics_list),
                    )
                )

    with LOG_PATH.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if is_new:
            writer.writeheader()
        writer.writerows(rows)

    return LOG_PATH


if __name__ == "__main__":
    print(f"Log file: {LOG_PATH}")
    print("This module is normally imported and called from scripts/run_baseline.py after a run.")
    print("To backfill a historical run from an existing results JSON, see scripts/backfill_experiment_log.py.")
