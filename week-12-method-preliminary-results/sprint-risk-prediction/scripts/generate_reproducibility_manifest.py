"""
Round-4 review addition: generate a row-level split manifest and exact
input-file identifiers so a reader can verify -- without re-running the
pipeline -- exactly which sprints landed in which fold for every
validation scheme reported in the analysis.

This does not require re-deriving the split logic: it imports the same
`time_ordered_split` used by `run_baseline.py` (and every other results
script) and the same `load_dataset`/`build_model_frame` pipeline, so the
manifest is guaranteed to describe the actual split used to produce every
reported number, not a re-implementation that could drift from it.

Writes three files to `results/reproducibility/`:
  - `input_files.json`        -- exact filenames, byte sizes, row counts,
                                  and SHA-256 checksums of the two raw
                                  TAWOS CSVs this pipeline was run against.
  - `time_ordered_split_manifest.json`
                                -- every sprint_id in the time-ordered
                                  train/test split (identical across both
                                  labels; verified and asserted below).
  - `lopo_fold_manifest.json`  -- every sprint_id's project (equivalently,
                                  its LOPO fold: held out when
                                  fold == project, else train).

Usage:
    python -m scripts.generate_reproducibility_manifest
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from scripts.run_baseline import time_ordered_split
from src.data.load_tawos import load_reconstructed_sprints
from src.features import add_features, build_model_frame
from src.labels import add_labels

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw" / "tawos"
OUT_DIR = REPO_ROOT / "results" / "reproducibility"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_input_file_identifiers() -> None:
    entries = {}
    for name in ["tawos_sprint.csv", "tawos_issue.csv"]:
        path = RAW_DIR / name
        with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
            n_lines = sum(1 for _ in f)
        entries[name] = {
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
            "lines_including_header": n_lines,
        }
    payload = {
        "dataset": "TAWOS (Tawosi, Moussa, and Sarro, MSR 2022)",
        "note": (
            "These are the exact two CSVs this pipeline was run against, "
            "exported from the raw TAWOS MySQL dump by "
            "scripts/export_tawos_subset.sql (Section 3.1 of the analysis). "
            "A checksum mismatch means a different export -- re-running "
            "the SQL script against the same MySQL dump should reproduce "
            "an identical file; TAWOS's own MSR 2022 release is the "
            "canonical source of the underlying dump."
        ),
        "files": entries,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "input_files.json").write_text(json.dumps(payload, indent=2))


def write_split_manifests() -> None:
    sprints = load_reconstructed_sprints()
    labeled = add_labels(sprints)
    featured = add_features(labeled)

    # The cold-start filter in build_model_frame does not depend on which
    # label column is requested, so the time-ordered split is identical
    # for label_delay and label_spillover. We verify that here rather than
    # assuming it, and fail loudly if a future code change breaks it.
    frames = {lc: build_model_frame(featured, lc) for lc in ["label_delay", "label_spillover"]}
    splits = {lc: time_ordered_split(f) for lc, f in frames.items()}

    train_ids = {lc: sorted(tr["sprint_id"].tolist()) for lc, (tr, _te) in splits.items()}
    test_ids = {lc: sorted(te["sprint_id"].tolist()) for lc, (_tr, te) in splits.items()}
    assert train_ids["label_delay"] == train_ids["label_spillover"], (
        "Time-ordered train split now differs by label -- update the "
        "paper's Section 3.7/Algorithm 1 description before trusting "
        "this manifest."
    )
    assert test_ids["label_delay"] == test_ids["label_spillover"], (
        "Time-ordered test split now differs by label -- update the "
        "paper's Section 3.7/Algorithm 1 description before trusting "
        "this manifest."
    )

    time_ordered_payload = {
        "note": (
            "Identical for label_delay and label_spillover (asserted at "
            "generation time). One row per sprint that survived the "
            "cold-start filter (n=2,002); each sprint_id is "
            "'<project>__<raw TAWOS Sprint.ID>'."
        ),
        "n_train": len(train_ids["label_delay"]),
        "n_test": len(test_ids["label_delay"]),
        "train_sprint_ids": train_ids["label_delay"],
        "test_sprint_ids": test_ids["label_delay"],
    }
    (OUT_DIR / "time_ordered_split_manifest.json").write_text(
        json.dumps(time_ordered_payload, indent=2)
    )

    # LOPO fold manifest: every modeled sprint's project IS its fold
    # assignment (held out when fold == project, trained on otherwise) --
    # so a full listing is just sprint_id -> project, using the same
    # n=2,002 modeled population as the time-ordered split above.
    frame = frames["label_delay"]
    lopo_payload = {
        "note": (
            "For LOPO fold F, the row is in the TEST fold iff "
            "project == F, else TRAIN. Same n=2,002 modeled population "
            "as the time-ordered manifest (LOPO re-uses build_model_frame "
            "'s cold-start-filtered frame, not the raw n=2,657 sprints)."
        ),
        "n_total": int(len(frame)),
        "project_counts": frame["project"].value_counts().to_dict(),
        "sprint_id_to_project": dict(
            sorted(zip(frame["sprint_id"], frame["project"]))
        ),
    }
    (OUT_DIR / "lopo_fold_manifest.json").write_text(json.dumps(lopo_payload, indent=2))


def main() -> None:
    write_input_file_identifiers()
    write_split_manifests()
    print(f"Wrote reproducibility manifests to {OUT_DIR}")


if __name__ == "__main__":
    main()
