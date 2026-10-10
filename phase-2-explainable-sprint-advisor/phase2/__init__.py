"""Phase 2 extension of the Phase 1 sprint-risk project.

Phase 2 reuses (does not copy) the Phase 1 code and TAWOS data that live in
../week-12-method-preliminary-results/sprint-risk-prediction. This module only
makes that Phase 1 package importable (as ``src``); Phase 2's own code lives in
the ``phase2`` package.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PHASE1_ROOT = (
    ROOT.parent / "week-12-method-preliminary-results" / "sprint-risk-prediction"
)
RESULTS_DIR = ROOT / "results"

if not PHASE1_ROOT.exists():
    raise FileNotFoundError(
        f"Phase 1 project not found at {PHASE1_ROOT}. Phase 2 expects to sit "
        "next to the week-12 folder of the submission package."
    )
# Appended (not prepended) so Phase 2 module names always win.
if str(PHASE1_ROOT) not in sys.path:
    sys.path.append(str(PHASE1_ROOT))
