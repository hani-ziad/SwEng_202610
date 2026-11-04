"""Run all Phase 4 experiments in order (about 15 minutes).

    python -m experiments.run_all      (from the phase-4-fixes folder)
"""
from experiments import e1_repeated_splits, e2_roc, e3_external

if __name__ == "__main__":
    e1_repeated_splits.main()
    e2_roc.main()
    e3_external.main()
    e3_external.figure()
