-- Run this against a locally-loaded TAWOS MySQL database (see
-- docs/data_access_and_versioning.md for how to obtain and load the dump)
-- to export exactly the two CSVs src/data/load_tawos.py expects, scoped to
-- the 5-8 selected projects named in the Week 8 proposal.
--
-- Replace the project names below with your final selection (criterion:
-- >= 30 completed sprints -- check candidate counts first with the
-- `-- 0. Candidate check` query).

-- 0. Candidate check: sprint counts per project, to pick the 5-8 to use.
SELECT p.Name AS project, COUNT(*) AS n_sprints
FROM Sprint s
JOIN Project p ON p.ID = s.Project_ID
WHERE s.State = 'CLOSED'
GROUP BY p.Name
ORDER BY n_sprints DESC;

-- 1. Sprint export (adjust the IN (...) list to your selected projects).
SELECT
    s.ID, s.Name, s.State, s.Start_Date, s.End_Date, s.Complete_Date,
    p.Name AS Project_Name
FROM Sprint s
JOIN Project p ON p.ID = s.Project_ID
WHERE p.Name IN ('project_a', 'project_b', 'project_c', 'project_d', 'project_e')
INTO OUTFILE '/var/lib/mysql-files/tawos_sprint.csv'
FIELDS ENCLOSED BY '"' TERMINATED BY ',' LINES TERMINATED BY '\n'
HEADER;

-- 2. Issue export for the same projects.
SELECT
    i.Issue_Key, i.Type, i.Status, i.Story_Point, i.Priority,
    i.Resolution_Date, i.Sprint_ID, i.Assignee_ID,
    p.Name AS Project_Name
FROM Issue i
JOIN Project p ON p.ID = i.Project_ID
WHERE p.Name IN ('project_a', 'project_b', 'project_c', 'project_d', 'project_e')
  AND i.Sprint_ID IS NOT NULL
INTO OUTFILE '/var/lib/mysql-files/tawos_issue.csv'
FIELDS ENCLOSED BY '"' TERMINATED BY ',' LINES TERMINATED BY '\n'
HEADER;

-- If --secure-file-priv blocks INTO OUTFILE, export via the mysql CLI instead, e.g.:
--   mysql -u USER -p TAWOS_DB -e "SELECT ... " | sed 's/\t/,/g' > tawos_sprint.csv
-- then copy tawos_sprint.csv and tawos_issue.csv into data/raw/tawos/.
