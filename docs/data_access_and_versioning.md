# Data access and versioning

## What the Week 8 proposal specified

The proposal's primary dataset is **TAWOS** (Tawosi et al., MSR 2022): a
relational MySQL dump of 44 open-source Scrum projects, distributed from
UCL's research data repository at `doi.org/10.5522/04/21308124`
(`rdr.ucl.ac.uk`), Apache 2.0 licensed. The code that expects this dataset
(`src/data/load_tawos.py`, `scripts/export_tawos_subset.sql`) is already
written and tested against the schema documented in the TAWOS paper and its
GitHub repo (`github.com/SOLAR-group/TAWOS`, schema creation script
included there) -- it is not exercised in this checkpoint's results because
of the access issue below.

## What actually happened when building this checkpoint

The sandbox this repository was built in has an organization-managed
network egress policy that only allows a fixed list of hosts (package
registries, GitHub's git protocol, a few others). `rdr.ucl.ac.uk`,
`api.figshare.com`, `zenodo.org`, and generic S3 endpoints are all outside
that allowlist and returned `403` on every attempt (both directly and
through the sandbox's proxy). GitHub's git protocol (`git clone`) *is*
allowed, but the TAWOS GitHub repo only contains the schema script and
documentation -- the actual data lives at the blocked UCL host, not in
git.

This is a constraint of the sandbox this checkpoint was executed in, not
of the dataset or the plan -- **your own laptop or university network almost
certainly is not behind this restriction**, so this is expected to be a
non-issue once you (or a teammate) fetch the data outside this sandbox.

### How to get the real TAWOS data in

1. Download the dump from `doi.org/10.5522/04/21308124` on a network that
   can reach it (your own machine).
2. Load it into a local MySQL instance: `mysql -u USER -p DB_NAME < tawos_dump.sql`.
3. Run `scripts/export_tawos_subset.sql` (edit the project name list to
   your final 5-8 selection -- the script's query 0 shows sprint counts
   per project so you can pick ones with >= 30 sprints, per the proposal).
4. Copy the two resulting CSVs into `data/raw/tawos/`.
5. Swap `from src.data.load_koralage import load_dataset` for
   `from src.data.load_tawos import load_dataset, derive_sprint_aggregates`
   in `scripts/run_baseline.py` (one line) -- everything downstream
   (`labels.py`, `features.py`, `baselines.py`) is written against the same
   canonical column names both loaders produce, so no other code changes
   are required.
6. If your laptop is linked to this workspace, the two CSVs can be staged
   in directly rather than re-uploaded each time; otherwise attach them to
   the conversation.

## What this checkpoint's working baseline actually used instead

To have a genuinely *working*, real-data baseline for this checkpoint
rather than a code skeleton with no results, the pipeline was run against
a smaller, directly obtainable substitute: the **Agile Scrum Sprint
Velocity dataset** (Koralage, 2020), covering 4 real open-source Jira
Scrum projects (Spring XD, Meso, Aurora, UserGrid) -- 238 sprints, 3,546
issues in total. It is small enough (3.2 MB) that it was cloned directly
into this repo's `data/raw/agile_scrum_sprint_velocity/` folder rather
than requiring any external hosting.

- **Source:** `github.com/RandulaKoralage/AgileScrumSprintVelocityDataSet`
- **Pinned commit:** `544b07e3f36e9c42040609effae5b68a13fdefc9` (2020-08-19)
  -- this is the exact version vendored into this repo; re-cloning the
  source repo later could pick up edits, so the vendored copy here is the
  version of record for this checkpoint's results.
- **License:** the source repository does not carry an explicit license
  file. It is used here for non-commercial course research and results
  are attributed to the original author (Randula Koralage, 2020); if this
  artifact is made public, get an explicit license clarification from the
  author before broader redistribution of the data itself (the code in
  this repo is unaffected).
- **Known limitations relative to TAWOS**, documented so they aren't
  mistaken for TAWOS-scale findings: only 4 projects (vs. 5-8 planned from
  a pool of 44), no issue-level resolution timestamps (sprint outcomes are
  pre-aggregated into summary columns), so "committed scope" is
  approximated from those aggregates rather than computed from raw
  issue timestamps (see the comment in `src/data/load_koralage.py`).

## Versioning approach

- The vendored dataset is committed into `data/raw/` as-is, at the pinned
  commit above -- for a dataset this small, a full copy is simpler and more
  reproducible than a pointer/checksum scheme.
- `results/baseline_results.json` and the two summary CSVs record exactly
  which run produced which numbers; re-running `scripts/run_baseline.py`
  regenerates them deterministically (fixed `random_state=42` on the
  random forest; logistic regression and the heuristic are deterministic).
- If/when real TAWOS data is added under `data/raw/tawos/`, record its
  export date and the `mysql` dump's own version/date (TAWOS is a static,
  versioned release, so this is mainly for your own bookkeeping) in this
  file rather than committing the raw MySQL dump itself (too large for
  git) -- the CSVs exported by `scripts/export_tawos_subset.sql` are the
  right size to commit instead.

## Other environment note: XGBoost

The proposal names XGBoost as the strongest model across the cited
related work. The sandbox used to build this checkpoint could not reach
any Python package index beyond a small pre-installed set (the same
network policy as above), so XGBoost could not be installed here.
`src/baselines.py` uses `RandomForestClassifier` as the tree-ensemble
entry for this checkpoint's results instead, with XGBoost left in
`requirements.txt` (commented) to install wherever an internet connection
is available (e.g. `pip install xgboost` on your own machine) for Week 12.
