"""Phase 4: fixing the weak spots of Phases 1-3 after expert feedback.

Reuses (does not copy or change) the Phase 1-3 code, which must sit next to
this folder. Phase 4's own code lives in the ``phase4`` package.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
PHASE3_ROOT = ROOT.parent / "phase-3-in-sprint-early-warning"
if str(PHASE3_ROOT) not in sys.path:
    sys.path.insert(0, str(PHASE3_ROOT))

import phase3  # noqa: E402,F401  (also loads Phase 2 and the Phase 1 ``src`` package)
