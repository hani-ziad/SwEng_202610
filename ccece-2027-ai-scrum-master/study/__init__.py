"""Sprint-replay study (120 sprints, three agent runs), added 15 Nov 2026.

Extends the 4 Nov pilot in ../pilot. Reuses the pilot's board builder and the Phase 1-3 code
and data, which must sit next to this folder (see ../pilot/__init__.py).
"""
from pathlib import Path

import pilot  # noqa: F401  (path setup for phase2, phase3 and the Phase 1 src package)

HERE = Path(__file__).resolve().parent
