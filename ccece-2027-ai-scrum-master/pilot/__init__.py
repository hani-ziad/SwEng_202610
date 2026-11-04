"""Sprint-replay pilot for the CCECE 2027 position paper.

Reuses the Phase 1-3 code and data, which must sit next to this folder:
  ../../phase-3-in-sprint-early-warning (which pulls in Phase 2 and Phase 1).
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PHASE3 = REPO / "phase-3-in-sprint-early-warning"
if str(PHASE3) not in sys.path:
    sys.path.insert(0, str(PHASE3))
import phase3  # noqa: E402,F401  (makes phase2 and the Phase 1 src package importable)
