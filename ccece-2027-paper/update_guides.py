"""Apply the 8 November 2026 review findings to the two guide files.

Usage (from the SwEng_202610 folder):
    python ccece-2027-paper/update_guides.py guides

Edits guides/Sprint_Risk_Prediction_Walkthrough.html (plain-language explanation) and
guides/Rebuild_From_Scratch_Manual.docx (manual re-run instructions). Every replacement must
match exactly once, otherwise nothing is written and the failing edit is printed.
Needs python-docx for the manual (pip install python-docx).
"""
import copy
import sys
from pathlib import Path

G = Path(sys.argv[1] if len(sys.argv) > 1 else "guides")
HTML = G / "Sprint_Risk_Prediction_Walkthrough.html"

# ---------------------------------------------------------------- walkthrough
EDITS = [
    # 2. the question: what the spillover rate really means
    ("""Because almost every sprint spills a little, the useful question for spillover is not "will anything be left over?" but "which sprints are most at risk?"</div>""",
     """Because almost every sprint spills a little, the useful question for spillover is not "will anything be left over?" but "which sprints are most at risk?"</div>
  <div class="warnbox"><strong>Found in the 8 November review: spillover is mostly rollover.</strong> Only 32% of commitments (one issue on one sprint's day-one board) are marked resolved by the sprint's planned end, 35% are resolved more than a month later, and 12% never. Many issues are simply carried into the next sprint: 43.5% of commitments are issues already committed to an earlier sprint. Checking at the moment the sprint was actually closed changes almost nothing (93% still spill). So in these open-source projects "spillover" mostly records issues moving on to later sprints, which is normal for them. It is not a reliable sign that a sprint was in trouble.</div>"""),
    # 3. the data: empty sprints are often loaded just after the start
    ("""<tr class="tot"><td>Left for <strong>spillover</strong></td><td class="n">1,691</td><td>Sprints that had at least one committed issue.</td></tr>
    </tbody>
  </table></div>""",
     """<tr class="tot"><td>Left for <strong>spillover</strong></td><td class="n">1,691</td><td>Sprints that had at least one committed issue.</td></tr>
    </tbody>
  </table></div>
  <div class="note"><strong>Many "empty" sprints are not really empty.</strong> In a third of the 911 (33%), the first issue was added within an hour after the sprint started, and in 42% within a day. The team pressed Start before loading the board. We still count them as empty, because the rule is strictly "what was on the board at the start", but it means the empty group is partly an accident of how teams use Jira.</div>"""),
    # 7. overlapping sprints: the gain was the new features, not the fix
    ("""So the higher scores mostly come from the new features, and the fixed rule makes sure none of them uses the future. The paper's main numbers, such as 0.751, all use the strict features.</p>""",
     """A check on 8 November separated the two changes. Keeping Phase 1's features but applying the fixed rule <em>lowers</em> the unseen-project scores (logistic regression 0.542 to 0.519, random forest 0.601 to 0.563). Adding only the two past-lateness features to Phase 1's list raises them (0.645 and 0.682). So the fix on its own costs a little accuracy, which is exactly what you expect when you take away information nobody had on day one, and the whole gain comes from past lateness. The fix matters because it makes the numbers honest, not because it makes them bigger. The paper's main numbers, such as 0.751, all use the strict features.</p>"""),
    # 8. phase 1: the count rule
    ("""<p>Spillover depends on how much the team took on: three simple numbers about the sprint's size give the best ranking (0.885). Late closure depends on team habits""",
     """<p>Spillover depends on how much the team took on: three simple numbers about the sprint's size give the best ranking (0.885), and the plain count of committed issues, with no model at all, gets 0.881. Late closure depends on team habits"""),
    ("""<tr><td>Spillover, sprints with work, time split</td><td>Logistic regression, scope only</td><td class="n">.885 (likely range .818 to .922)</td><td>Gradient boosting .812</td></tr>""",
     """<tr><td>Spillover, sprints with work, time split</td><td>Logistic regression, scope only</td><td class="n">.885 (likely range .818 to .922)</td><td>Gradient boosting .812</td></tr>
      <tr><td>Spillover, sprints with work, time split</td><td>No model: just count the committed issues</td><td class="n">.881</td><td>Almost the same as the size-only model. A sprint with more issues has more chances to leave one unfinished, so this is mostly arithmetic.</td></tr>"""),
    # 9. numeric targets: size-normalised label
    ("""<li><strong>Share of work left over</strong> (called <em>carry share</em> in the paper): almost impossible to predict on day one. The best model is only slightly better than always guessing the typical value.</li>""",
     """<li><strong>Share of work left over</strong> (called <em>carry share</em> in the paper): almost impossible to predict on day one. The best model is only slightly better than always guessing the typical value. Turning it into a yes/no label ("half or more of the committed issues left over", true for 86% of test sprints) does not help: the best model scores 0.615 and the plain count 0.562. So the day-one features say whether <em>something</em> will slip, mostly through sprint size, but almost nothing about <em>how much</em>.</li>"""),
    # 10. phase 3: in-sprint warning vs the count
    ("""<p>Once the sprint is running, progress information makes spillover much easier to predict (0.83 to 0.96 by mid-sprint). It does nothing for late closure, which is about when someone clicks the button, not about the work.</p>""",
     """<p>Once the sprint is running, spillover becomes much easier to predict (0.83 to 0.96 by mid-sprint), but simply counting the committed issues still open does just as well. Progress does nothing for late closure, which is about when someone clicks the button, not about the work.</p>"""),
    ("""<i style="background:var(--s4)"></i>spillover, simple rule (share of work still open)</div>""",
     """<i style="background:var(--s4)"></i>spillover, simple rule (share of work still open) <i style="background:var(--s3)"></i>spillover, plain count of committed issues still open</div>"""),
    ("""{n:'spillover, learning-free rule',c:'var(--s4)',v:[null,.667,.767,.802,null]}];""",
     """{n:'spillover, learning-free rule',c:'var(--s4)',v:[null,.667,.767,.802,null]},{n:'spillover, count of open issues',c:'var(--s3)',v:[null,.925,.945,.954,null]}];"""),
    ("""<p>Is the model really needed, or would a simple rule do? The yellow line is the rule "the more committed work is still open, the higher the risk". For spillover it scores 0.667, 0.767 and 0.802 at the 25%, 50% and 75% checkpoints, well below the model, which also knows how much time is left, how big the sprint is and how the team usually performs. For predicting <em>how much</em> work will be left over, though, the simple rule is as good as the model from halfway on.</p>""",
     """<p>Is the model really needed, or would a simple rule do? <strong>A simple rule does.</strong> The yellow line is the rule "the bigger the <em>share</em> of committed work still open, the higher the risk": 0.667, 0.767 and 0.802 at the 25%, 50% and 75% checkpoints. That was the wrong yardstick, because spillover only asks whether <em>any</em> issue is left, and a big sprint can be half done and still have many issues open. The green line is the plain <em>count</em> of committed issues still open: 0.925, 0.945 and 0.954, practically the same as the model. A careful test, re-sampling the eight projects 2,000 times, cannot tell the model and the count apart at any checkpoint, and on unseen projects the count is slightly ahead. Earlier versions of this page and the paper said the model beat counting. That was wrong, and the 8 November review corrected it. For predicting <em>how much</em> work will be left over, the share rule is as good as the model from halfway on.</p>"""),
    # 14. outside projects: intervals
    ("""The average AUC is ${mean.toFixed(3)}, and the model beats chance in 7 of them (Compass is the exception).""",
     """The average AUC is ${mean.toFixed(3)}, and the model scores above chance in 7 of them (Compass is the exception), but with 21 to 269 sprints per project only 2 of the 8 likely ranges stay above 0.5 (ColumnStore and the PHP driver)."""),
    # 15. what it means
    ("""<p>The contribution is a careful way of working with Jira history, not a tool to deploy. Six habits carry over to any similar study.</p></div>""",
     """<p>The contribution is a careful way of working with Jira history, not a tool to deploy. Eight habits carry over to any similar study.</p></div>
  <p>One honest note on what is new. Rebuilding an iteration's state from the change log was first done by Choetkiertikul and colleagues in 2018 (IEEE Transactions on Software Engineering), who predicted delivered story points at the start and at 30%, 50% and 80% of an iteration. This project follows the same idea. What it adds is the record of where the change log runs out, the fix for overlapping sprints, the empty-sprint and label checks, the comparison of every model with a simple count, and the reproduced leakage mistakes.</p>"""),
    ("""<li><span class="k">6</span><div><strong>Test on teams the model never saw, and expect to re-set the threshold.</strong> The ranking carried over to ten outside projects; the yes/no cut-off did not.</div></li>""",
     """<li><span class="k">6</span><div><strong>Test on teams the model never saw, and expect to re-set the threshold.</strong> The ranking carried over to ten outside projects, weakly; the yes/no cut-off did not.</div></li>
    <li><span class="k">7</span><div><strong>Compare every model with a plain count.</strong> For spillover, counting the committed issues (0.881) or the ones still open mid-sprint (0.945) was as good as any model.</div></li>
    <li><span class="k">8</span><div><strong>Score each project on its own as well as all together.</strong> A model that knows which project a sprint belongs to can look strong just by telling projects apart. Late closure scored 0.751 pooled but 0.634 within a project.</div></li>"""),
    ("""During the sprint, the quarter-way and halfway checks spot likely spillover earlier and better than just counting open issues. Late closure""",
     """During the sprint, simply counting the committed issues still open is as good a spillover warning as any model we built. Late closure"""),
    ("""<li>Jira's completion timestamp marks when a sprint was closed, not when its work was done.</li>""",
     """<li>Jira's completion timestamp marks when a sprint was closed, not when its work was done.</li>
    <li>"Finished" is read from each issue's resolution date. In Confluence Cloud most issues are closed without one (74% of its sprints show nothing finished at the planned end, 48% in Confluence Server, 11% to 31% elsewhere), so its spillover labels are unreliable. Leaving Confluence Cloud out moves the size-only spillover score from 0.885 to 0.908. Late closure is not affected.</li>"""),
    # 16. journey
    ("""<li><b>4 November 2026 · Phase 4</b>""",
     """<li><b>8 November 2026 · Hostile review of the CCECE paper</b><small>Every claim checked against the data. Found: an earlier study (Choetkiertikul et al., 2018) had already rebuilt iteration state from the change log, so the paper now cites it; a plain count of committed issues matches the spillover models, before and during the sprint; the overlapping-sprint fix does not raise the scores on its own; Confluence Cloud's missing resolution dates; only 2 of the 8 outside-project ranges clear 0.5; most late closures are only one to three days late; and the late-closure score of 0.751 falls to 0.634 when each project is scored on its own. The paper was retitled \"Predicting Sprint Outcomes from Reconstructed Jira State: Count Baselines and a Leakage Audit\", and the paper, its number checker and this page were corrected. New check scripts are in <code>ccece-2027-paper/reviewer_checks</code>.</small></li>
    <li><b>4 November 2026 · Phase 4</b>"""),
    ("""and a script (<code>verify_numbers.py</code>) that checks all 319 numbers in the paper against those results.""",
     """and a script (<code>verify_numbers.py</code>) that checks every number in the paper against those results."""),
    # glossary
    ("""<dt>Checkpoint</dt>""",
     """<dt>Carry share</dt><dd>The share of a sprint's committed issues still unfinished at its planned end. A size-fair version of spillover.</dd>
    <dt>Count rule</dt><dd>A baseline with no learning: rank sprints by how many committed issues they have (at planning) or how many are still open (during the sprint).</dd>
    <dt>Checkpoint</dt>"""),
    # 8. phase 1: late closure is mostly between-project
    ("""Late closure depends on team habits: the sprint's size tells you almost nothing, while the team's history gets to 0.751.""",
     """Late closure looks predictable (0.751), but much of that comes from telling the eight projects apart. Within one project the model scores 0.634, and the project's own recent late rate predicts its next sprints no better than a coin toss (0.510). Most lateness is also short: 40% of late sprints close only one to three days late, sprints planned to end on a Friday or a weekend are late twice as often (57% against 28%), and if only sprints more than a week late count, the within-project score falls to 0.502, a coin toss. So late closure mostly measures how promptly someone closes the board."""),
    ("""<tr><td>Late closure, time split</td><td>Gradient boosting, strict features</td><td class="n">.751</td><td>Each project's own late rate .640; scope only .530</td></tr>""",
     """<tr><td>Late closure, time split</td><td>Gradient boosting, strict features</td><td class="n">.751</td><td>Each project's own late rate .640; scope only .530</td></tr>
      <tr><td>Late closure, scored within each project</td><td>Gradient boosting, strict features</td><td class="n">.634 (average of the 8 projects)</td><td>The team's own late rate over its last 3 finished sprints: .510, a coin toss. The pooled .751 is high partly because the model knows which project each sprint is from.</td></tr>"""),
    # 15. what a team can do
    ("""At planning, the scores can put sprints in order of risk, so a team or Scrum Master knows which ones to keep an eye on.""",
     """At planning, the scores mostly tell projects apart. Within one team they rank its sprints only a little better than chance (0.634), so they are not yet good enough to pick which sprint to watch."""),
    # glossary: within-project
    ("""<dt>LOPO</dt>""",
     """<dt>Within-project score</dt><dd>The AUC worked out separately for each project and then averaged. It shows whether a model can tell a team's risky sprints from its safe ones, not just risky teams from safe teams.</dd>
    <dt>LOPO</dt>"""),
]


def patch_html():
    s = HTML.read_text(encoding="utf-8")
    bad = [a[:70] for a, _ in EDITS if s.count(a) != 1]
    if bad:
        print("Walkthrough NOT changed. These edits did not match exactly once:")
        for b in bad:
            print("  -", b)
        return False
    for a, b in EDITS:
        s = s.replace(a, b)
    HTML.write_text(s, encoding="utf-8")
    print(f"Walkthrough updated: {len(EDITS)} edits")
    return True


# ---------------------------------------------------------------- manual
MANUAL_SECTION = [
    ("h", "19. The 8 November 2026 review checks"),
    ("p", "These checks were added after a line-by-line review of the CCECE paper. They test claims the paper makes against the simplest possible baselines. Each script reads result files the main pipeline already produced, so run them after Phases 2 and 3 and the outside checks."),
    ("p", "Run every command from the SwEng_202610 folder, with the virtual environment active:"),
    ("code", "set R=phase-2-explainable-sprint-advisor\\results"),
    ("code", "set C=phase-3-in-sprint-early-warning\\results"),
    ("code", "set T=week-12-method-preliminary-results\\sprint-risk-prediction"),
    ("code", "set K=ccece-2027-paper\\reviewer_checks"),
    ("code", "python %K%\\planning_checks.py %R%\\phase2_modeling_frame.csv %T%\\data\\raw\\tawos\\tawos_sprint.csv %T%\\data\\raw\\tawos\\tawos_change_log.csv phase-4-fixes\\results\\e3_external_per_project.csv %K%\\planning_checks.csv"),
    ("code", "python %K%\\progress_rules.py %C%\\phase3_checkpoint_frame.csv %K%\\progress_rules.csv"),
    ("code", "python %K%\\insprint_vs_count.py %C%\\phase3_checkpoint_frame.csv %K%\\insprint_vs_count.csv"),
    ("code", "python %K%\\timing_confound.py %R%\\phase2_modeling_frame.csv %K%\\timing_confound.csv"),
    ("code", "python %K%\\timing_confound_time.py %R%\\phase2_modeling_frame.csv %K%\\timing_confound_time.csv"),
    ("code", "python %K%\\normalised_label.py %R%\\phase2_modeling_frame.csv %K%\\normalised_label.csv"),
    ("code", "python %K%\\resolution_timing.py %T%\\results\\commitment_provenance.csv %T%\\data\\raw\\tawos\\tawos_issue.csv %T%\\data\\raw\\tawos\\tawos_sprint.csv %R%\\phase2_modeling_frame.csv %K%\\resolution_timing.csv"),
    ("code", "python %K%\\overlap_claim.py %T%\\data\\raw\\tawos\\tawos_sprint.csv %R%\\phase2_modeling_frame.csv %K%\\overlap_claim.csv"),
    ("code", "python %K%\\late_closure_checks.py %R%\\phase2_modeling_frame.csv %K%\\late_closure_checks.csv"),
    ("code", "python %K%\\label_and_timing_extra.py %T%\\results\\commitment_provenance.csv %T%\\data\\raw\\tawos\\tawos_issue.csv %R%\\phase2_modeling_frame.csv %K%\\label_and_timing_extra.csv"),
    ("code", "python %K%\\late_threshold_checks.py %R%\\phase2_modeling_frame.csv %T%\\data\\raw\\tawos\\tawos_sprint.csv phase-4-fixes\\results\\e1_repeats.csv %K%\\late_threshold_checks.csv"),
    ("code", "python %K%\\misc_numbers.py %R%\\phase2_modeling_frame.csv phase-4-fixes\\results\\e3_external_per_project.csv %K%\\misc_numbers.csv"),
    ("p", "Expected values (small differences in the third decimal are normal across scikit-learn versions):"),
    ("table", [("Check", "Expected"),
               ("Spillover, raw count of committed issues, time split", "AUC 0.881 (scope-only model 0.885)"),
               ("Spillover, open committed count at 25/50/75%", "0.925 / 0.945 / 0.954"),
               ("Gradient boosting with progress minus count, 95% interval", "includes zero at every checkpoint"),
               ("Late closure LOPO, legacy content with strict timing", "LR 0.519, RF 0.563"),
               ("Late closure LOPO, legacy plus past lateness", "LR 0.645, RF 0.682"),
               ("Carry share >= 0.5 label: best model / count", "0.615 (random forest) / 0.562"),
               ("Commitments resolved by planned end", "32% (35% more than 30 days later, 12% never)"),
               ("Confluence Cloud sprints with nothing done at planned end", "74% (Confluence Server 48%)"),
               ("Previous sprint unfinished at the next start", "71%"),
               ("Late closure, GB, mean within-project AUC", "0.634 (pooled 0.751)"),
               ("Late closure, team's own past late rate, within-project", "0.510"),
               ("Commitments that re-commit an earlier issue", "44%"),
               ("Confluence Cloud commitments with no resolution date", "51%"),
               ("LOPO, strict timing minus legacy (RF)", "-0.039 [-0.053, -0.007]"),
               ("Late closure > 7 days: GB pooled / within-project", "0.638 / 0.502"),
               ("Late rate when the planned end is Fri/weekend vs Mon-Thu", "57% vs 28%")]),
    ("p", "Then check the paper: python ccece-2027-paper\\verify_numbers.py must end with FAILS: []. The Word copy of the paper is rebuilt from the LaTeX source with: python ccece-2027-paper\\tex2blocks.py ccece-2027-paper\\CCECE_2027_Sprint_Risk_Prediction.tex blocks.json, then node ccece-2027-paper\\build_docx.js blocks.json ccece-2027-paper\\CCECE_2027_Sprint_Risk_Prediction.docx (needs Node.js and npm install docx)."),
    ("p", "Figure 1 is drawn by python ccece-2027-paper\\make_fig1.py. What these checks changed in the paper: it was retitled, late closure is now reported within each project as well as pooled, the spillover results are now compared with a plain count, the in-sprint model is reported as tied with the count, the parallel-sprint fix is reported as costing a little accuracy rather than adding it, and the resolution-date defect is stated as a limitation."),
]


def patch_manual():
    try:
        import docx
    except ImportError:
        print("Manual NOT changed: install python-docx first (python -m pip install python-docx)")
        return False
    path = G / "Rebuild_From_Scratch_Manual.docx"
    d = docx.Document(str(path))
    if any(p.text.startswith("19. The 8 November 2026 review checks") for p in d.paragraphs):
        print("Manual already has section 19; skipped")
        return True
    anchor = next((p for p in d.paragraphs if p.text.strip().startswith("Appendix A")
                   and p.style.name.lower().startswith("heading")), None)
    h_style = next((p.style for p in d.paragraphs if p.text.strip() == "18. Final reproducibility checklist"
                    and p.style.name.lower().startswith("heading")), None)
    code_like = next((p for p in d.paragraphs if p.text.startswith("python -m pip install -r requirements.txt")), None)
    if anchor is None or h_style is None:
        print("Manual NOT changed: could not find the Appendix A or section 18 headings")
        return False
    for kind, val in MANUAL_SECTION:
        if kind == "h":
            anchor.insert_paragraph_before(val, style=h_style)
        elif kind == "p":
            anchor.insert_paragraph_before(val)
        elif kind == "code":
            p = anchor.insert_paragraph_before(val, style=code_like.style if code_like is not None else None)
            for r in p.runs:
                r.font.name = "Consolas"
        elif kind == "table":
            t = d.add_table(rows=len(val), cols=2)
            t.style = d.tables[0].style if d.tables else None
            for i, (a, b) in enumerate(val):
                t.cell(i, 0).text, t.cell(i, 1).text = a, b
                if i == 0:
                    for c in t.rows[0].cells:
                        for r in c.paragraphs[0].runs:
                            r.bold = True
            anchor._p.addprevious(t._tbl)
    # contents list entry and checklist items
    toc = next((p for p in d.paragraphs if p.text.strip() == "18. Final reproducibility checklist"
                and not p.style.name.lower().startswith("heading")), None)
    if toc is not None:
        new = copy.deepcopy(toc._p)
        toc._p.addnext(new)
        para = docx.text.paragraph.Paragraph(new, toc._parent)
        for r in para.runs[1:]:
            r.text = ""
        para.runs[0].text = "19. The 8 November 2026 review checks"
    last_check = [p for p in d.paragraphs if p.text.startswith("☐")]
    if last_check:
        for txt in ["☐  The 8 November review checks (section 19) match their expected values.",
                    "☐  verify_numbers.py ends with FAILS: []."]:
            new = copy.deepcopy(last_check[-1]._p)
            last_check[-1]._p.addnext(new)
            para = docx.text.paragraph.Paragraph(new, last_check[-1]._parent)
            for r in para.runs[1:]:
                r.text = ""
            para.runs[0].text = txt
            last_check[-1] = para
    d.save(str(path))
    print("Manual updated: section 19 added")
    return True


if __name__ == "__main__":
    ok1 = patch_html()
    ok2 = patch_manual()
    sys.exit(0 if ok1 and ok2 else 1)
