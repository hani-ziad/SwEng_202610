"""Dependency-free test runner (same idea as Phase 1's scripts/run_tests.py).

Usage (from the phase-3 folder):  python -m experiments.run_tests
With pytest installed you can simply run:  pytest tests/
"""
from __future__ import annotations

import importlib.util
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    import phase3  # noqa: F401

    passed = failed = 0
    for path in sorted((ROOT / "tests").glob("test_*.py")):
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for name in dir(module):
            if not name.startswith("test_"):
                continue
            try:
                getattr(module, name)()
                print(f"PASS  {path.stem}.{name}")
                passed += 1
            except Exception:
                print(f"FAIL  {path.stem}.{name}")
                traceback.print_exc()
                failed += 1
    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
