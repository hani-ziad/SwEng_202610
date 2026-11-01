# Koralage sprint-velocity data (not distributed)

The Week 10 baseline used the "Agile Scrum Sprint Velocity" CSVs (Koralage; four
projects: Aurora, Meso, Spring XD, Usergrid). They were removed from this
repository on 1 November 2026 because they carry no licence. Nothing in the
final results depends on them: Phases 1 to 3 and the papers use TAWOS only.

To rerun the Week 10 Koralage baseline, place the original folder at
`data/raw/agile_scrum_sprint_velocity/` (the four "Finalized Datasets for ..."
subfolders). The tests in `tests/test_load_koralage.py` skip while it is absent.
