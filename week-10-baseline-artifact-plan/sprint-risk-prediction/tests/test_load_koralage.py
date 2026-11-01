import unittest
from pathlib import Path

from src.data.load_koralage import load_dataset

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "agile_scrum_sprint_velocity"


def _require_data():
    # The Koralage CSVs carry no licence, so they are not distributed with this
    # repository. Drop them into RAW_DIR to run these tests.
    if not RAW_DIR.exists():
        raise unittest.SkipTest(f"Koralage data not present at {RAW_DIR}")


def test_loads_all_four_projects():
    _require_data()
    sprints, issues = load_dataset(RAW_DIR)
    assert set(sprints["project"].unique()) == {"usergrid", "aurora", "meso", "spring_xd"}
    assert len(sprints) > 200
    assert len(issues) > 3000


def test_sprint_dates_parsed_and_ordered():
    _require_data()
    sprints, _ = load_dataset(RAW_DIR)
    assert sprints["start_date"].notna().all()
    assert sprints["end_date"].notna().all()
    # end_date should not be before start_date for any sprint.
    assert (sprints["end_date"] >= sprints["start_date"]).all()


def test_sprint_ids_unique_across_projects():
    _require_data()
    sprints, _ = load_dataset(RAW_DIR)
    assert sprints["sprint_id"].is_unique


def test_issue_sprint_ids_reference_real_sprints():
    _require_data()
    sprints, issues = load_dataset(RAW_DIR)
    known_sprint_ids = set(sprints["sprint_id"])
    # Most issues should reference a sprint that exists in the Sprints table
    # (a small fraction of orphaned references is expected/documented).
    matched = issues["sprint_id"].isin(known_sprint_ids).mean()
    assert matched > 0.5
