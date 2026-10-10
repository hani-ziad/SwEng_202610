"""Phase 3 extension: in-sprint early warning.

Reuses (does not copy) the Phase 1 code/data and the Phase 2 package, which
must sit next to this folder in the submission package:

  ../week-12-method-preliminary-results/sprint-risk-prediction   (Phase 1)
  ../phase-2-explainable-sprint-advisor                          (Phase 2)

Phase 3's own code lives in the ``phase3`` package.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PHASE2_ROOT = ROOT.parent / "phase-2-explainable-sprint-advisor"
RESULTS_DIR = ROOT / "results"

if not PHASE2_ROOT.exists():
    raise FileNotFoundError(
        f"Phase 2 project not found at {PHASE2_ROOT}. Phase 3 expects to sit next "
        "to the phase-2-explainable-sprint-advisor and week-12 folders."
    )
if str(PHASE2_ROOT) not in sys.path:
    sys.path.append(str(PHASE2_ROOT))

import phase2  # noqa: E402,F401  (also makes the Phase 1 ``src`` package importable)

PHASE1_ROOT = phase2.PHASE1_ROOT
RAW_DIR = PHASE1_ROOT / "data" / "raw" / "tawos"
