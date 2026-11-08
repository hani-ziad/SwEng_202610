"""Run all Phase 4 experiments in order (about 30 minutes).

    python -m experiments.run_all      (from the phase-4-fixes folder)
"""
from experiments import e1_repeated_splits, e2_roc, e3_external, e4_sanity_checks, e5_relabelled, e6_within_project

if __name__ == "__main__":
    e1_repeated_splits.main()
    e2_roc.main()
    e3_external.main()
    e3_external.figure()
    e4_sanity_checks.main()
    e5_relabelled.main()
    e6_within_project.main()
