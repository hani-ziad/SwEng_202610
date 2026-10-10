"""Run every Phase 2 step in order (about six minutes in total).

Usage (from the phase-2 folder):  python -m experiments.run_all
"""
from __future__ import annotations

from experiments import make_figure, run_advisor, run_explanations, run_models


def main():
    for name, mod in [
        ("1/4 model study (both outcomes)", run_models),
        ("2/4 Shapley explanations", run_explanations),
        ("3/4 sprint advisor", run_advisor),
        ("4/4 figure", make_figure),
    ]:
        print(f"\n===== {name} =====")
        mod.main()


if __name__ == "__main__":
    main()
