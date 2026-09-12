"""
Minimal dependency-free test runner.

This project's tests are ordinary pytest-style test functions (run them
with `pytest` normally -- see requirements.txt). This script exists only
because the sandbox this checkpoint was built in could not install pytest
itself (no external package index reachable -- see
docs/data_access_and_versioning.md); it discovers and runs the same
test_*.py files using nothing but the standard library, so the suite has
real, executed evidence attached to this checkpoint either way.
"""
from __future__ import annotations

import importlib.util
import sys
import traceback
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parents[1] / "tests"
REPO_ROOT = Path(__file__).resolve().parents[1]


def discover_test_functions():
    sys.path.insert(0, str(REPO_ROOT))
    for path in sorted(TESTS_DIR.glob("test_*.py")):
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for name in dir(module):
            if name.startswith("test_"):
                yield f"{path.stem}.{name}", getattr(module, name)


def main() -> int:
    passed, failed = 0, []
    for full_name, fn in discover_test_functions():
        try:
            fn()
        except Exception:
            failed.append((full_name, traceback.format_exc()))
            print(f"FAIL  {full_name}")
        else:
            passed += 1
            print(f"PASS  {full_name}")

    print(f"\n{passed} passed, {len(failed)} failed")
    for name, tb in failed:
        print(f"\n--- {name} ---\n{tb}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
