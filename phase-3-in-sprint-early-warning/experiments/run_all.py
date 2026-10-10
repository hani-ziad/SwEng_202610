"""Run every Phase 3 step in order (about eleven minutes in total).

Usage (from the phase-3 folder):  python -m experiments.run_all
"""
from __future__ import annotations

from experiments import make_figure, run_explanations, run_models


def main():
    for name, mod in [
        ("1/3 checkpoint model study", run_models),
        ("2/3 Shapley explanations", run_explanations),
        ("3/3 figures", make_figure),
    ]:
        print(f"\n===== {name} =====")
        mod.main()


if __name__ == "__main__":
    main()
