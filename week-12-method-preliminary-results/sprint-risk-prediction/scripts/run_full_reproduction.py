"""Run the analyses used by the current manuscripts in a fixed order.

This entry point does not edit the raw CSV files. Each analysis writes its own
CSV, JSON, or PDF output under results/.
"""
from __future__ import annotations

import subprocess
import sys

STEPS = [
    "scripts.run_baseline",
    "scripts.audit_commitment_provenance",
    "scripts.run_validation_sensitivity_analyses",
    "scripts.run_label_sensitivity",
    "scripts.run_leakage_case_studies",
    "scripts.run_tests",
]


def main() -> int:
    for module in STEPS:
        print(f"\n=== Running {module} ===")
        result = subprocess.run([sys.executable, "-m", module], check=False)
        if result.returncode != 0:
            print(f"Stopped because {module} returned {result.returncode}.")
            return result.returncode
    print("\nAll requested analysis steps completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
