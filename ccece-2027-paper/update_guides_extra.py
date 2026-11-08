"""Two small walkthrough fixes that update_guides.py does not make.

Run it right after update_guides.py, from the SwEng_202610 folder:
    python ccece-2027-paper/update_guides.py guides
    python ccece-2027-paper/update_guides_extra.py guides

1. Section 15 now lists eight habits, so its opening sentence must say eight, not six.
2. Section 13 said the MariaDB result is "not a quirk of one dataset" without the per-project
   scores. Pooled over the two MariaDB projects the model scores 0.710, but 0.699 on ColumnStore
   and 0.592 on MaxScale (phase-4-fixes/results/e3_external_per_project.csv).

Every edit must match exactly once, otherwise nothing is written.
"""
import sys
from pathlib import Path

HTML = Path(sys.argv[1] if len(sys.argv) > 1 else "guides") / "Sprint_Risk_Prediction_Walkthrough.html"

EDITS = [
    ("It is a careful method for using Jira history, and six lessons that apply",
     "It is a careful method for using Jira history, and eight lessons that apply"),
    ("So the late-closure finding is not a quirk of one dataset.</p>",
     "So the late-closure finding is not a quirk of one dataset. Part of the 0.710 again comes from telling the two "
     "projects apart, though: scored on each project alone, the model gets 0.699 on ColumnStore and 0.592 on MaxScale.</p>"),
]

s = HTML.read_text(encoding="utf-8")
if "Eight habits carry over" not in s:
    sys.exit("Run update_guides.py first; this script expects its changes.")
bad = [a[:60] for a, _ in EDITS if s.count(a) != 1]
if bad:
    print("Walkthrough NOT changed. These edits did not match exactly once:")
    for b in bad:
        print("  -", b)
    sys.exit(1)
for a, b in EDITS:
    s = s.replace(a, b)
HTML.write_text(s, encoding="utf-8")
print(f"Walkthrough updated: {len(EDITS)} extra edits")
