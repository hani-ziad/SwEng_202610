"""Rewrite the two guide documents (guide/Plain_Explanation.docx and guide/Rebuild_Manual.docx)
from the study's result files, keeping the documents' existing styles.

Usage (from the ccece-2027-ai-scrum-master folder):  python -m study.write_guides
"""
from __future__ import annotations

import docx

import study
from study.paper_numbers import expected

G = study.HERE.parent / "guide"


def clear(d):
    body = d.element.body
    for el in list(body):
        if el.tag.endswith("sectPr"):
            continue
        body.remove(el)


def plain(e):
    d = docx.Document(G / "Plain_Explanation.docx")
    clear(d)
    H = lambda t: d.add_paragraph(t, style="Heading 1")
    P = lambda t: d.add_paragraph(t, style="Normal")
    B = lambda t: d.add_paragraph(t, style="List Bullet")
    d.add_paragraph("Grounding the AI Scrum Master, explained simply", style="Title")
    H("The idea")
    P("Several tools now call an AI agent a \"Scrum Master\", but none has been checked against what really happened "
      "in real sprints. This paper argues that before anyone trusts such an agent, it should pass a cheap test we call "
      "sprint replay.")
    H("How sprint replay works")
    P("Take a real sprint from public Jira data and rebuild the board exactly as it looked at a chosen moment, such as "
      "the first morning or halfway through. Show it to three players: a plain prediction model (M), the AI agent on its "
      "own (L), and the AI agent that also sees the model's prediction (G). Each decides whether to warn the team that "
      "the sprint will close late, and guesses how much of the planned work will be left unfinished. Then compare their "
      "answers with what the sprint actually did.")
    H("Two ways an agent can fail")
    B("Ungrounded: it says confident things the board does not support.")
    B("Over-grounded: it just repeats the model and adds nothing but a friendly voice.")
    H(f"What the replay found ({e['sample sprints']} sprints, three runs)")
    B(f"The agent that saw the model agreed with it {e['A_G percent'].replace(chr(92), '')} of the time. When it "
      f"disagreed and warned anyway, it was wrong every time ({e['G extra warnings']} extra warnings per run, all on "
      "sprints that finished on time). So, under every cost assumption we tried, it was no cheaper than the model "
      "alone, even when the model's alarm thresholds were tuned to each assumption.")
    B(f"The agent without the model behaved almost exactly like a one-line rule: warn if the team was late in at least "
      f"half of its last three sprints. It matched that rule on {e['rule agree runs']} of 120 sprints.")
    B(f"To check whether the agent was simply copying, we changed what the model told it. With the verdict hidden it "
      f"still agreed with the model's real call {e['P agree'].replace(chr(92), '')} of the time. When we turned the "
      f"model's number around so its verdict was the opposite, it followed the false verdict on {e['F follow n']} "
      "decisions (96 in a second run), about as often when the false verdict was wrong as when it happened to be right. "
      "When only the written verdict was flipped but the real number was shown, it followed the number. So it leans on "
      "the model's number, and a manipulated number misleads it.")
    B("Costs are only a \"what if\": they depend on how much a warning helps, which nobody observed. Under our main "
      f"assumption the model beat both \"never warn\" and \"always warn\" only when a warning cost {e['M band']} of a "
      "late sprint. If warnings help nothing, never warning is always cheapest.")
    B(f"Running the agent three times changed very little: all three runs made the same call on about "
      f"{e['same call L'].replace(chr(92), '')} of boards.")
    B("At the start of a sprint, nobody (model, agent or rule) could predict how much work would be left unfinished. "
      "Halfway through, the model and the agents could, but the agents were no better than the model.")
    B(f"A script checked every number the agents wrote: {e['numbers traced']} of {e['numbers']} matched the board. Two "
      "AI raters (the same model family as the agent) then read every message. In the first rating the model lines "
      "were shown only next to the model-aware agent's messages, which gave away which agent wrote them, so two fresh "
      "raters read them all again with the model lines shown for both. Then about 10% of the plain agent's warnings and "
      "7% of the model-aware agent's overstated something when both raters agreed, for example calling every item a "
      "\"story\" when some were bugs. That difference is too small to call.")
    B("About 26% of the plain agent's warnings (and 16% of the other's) told the team to close the sprint on its "
      "planned date and carry leftover work over. That is what Scrum asks for, and it shows the \"late closure\" label "
      "rewards closing on time more than finishing the work.")
    H("What changed late on 15 November 2026")
    P("A second reviewer read the revised paper. We checked every point against the files and then ran more tests: "
      "the raters read all messages again with nothing giving the agent away; the flipped test ran a second time; a "
      "new test flipped only the written verdict; the agent was rerun with the model's accuracy measured on training "
      "data instead of the test set; and the model's thresholds were tuned to each cost assumption. None of this "
      "changed the main findings, but the paper now says them more carefully: no \"hindsight-free\" claim, the "
      "13 sprints shared with the pilot are disclosed (results without them are the same), only six of the eight "
      "projects enter the within-project score, and the planning sample is limited by a fact known only at halfway. "
      "The CCECE paper is now 6 pages (the limit). The Canadian AI paper got the same changes in short form.")
    H("What changed on the evening of 15 November 2026")
    P("An outside reviewer read the Canadian AI version and asked for a narrower claim and a sharper test. This paper "
      "now says it tests one part of the role (a sprint-risk assistant), adds the hidden and flipped model tests, adds "
      "the never-warn and always-warn policies and a scenario where warnings do nothing, writes the cost formula "
      "correctly for sprints with no warning, calls the evidence table illustrative, and no longer calls a lower cost "
      "\"evidence of judgement\". It is still 5 pages.")
    H("What changed earlier on 15 November 2026")
    P("The first pilot (4 November) used 40 sprints, one run and one person coding the messages, and did not check the "
      "agents' numbers. The new study uses 120 sprints spread evenly over the eight projects, three runs per agent, a "
      "second target (the share of planned work left unfinished), an automatic number check and two independent "
      "raters. The two raters' coding matched the original single coder well, so the pilot's finding about closing "
      "on the date holds. The old pilot files are kept in pilot/.")
    H("What the study cannot show")
    P("It is one AI model with one prompt. The raters are the same kind of model as the agents, so they may share blind "
      "spots, and no person has settled the cases where they disagreed. The boards have no issue titles or descriptions, which is where an agent might add most. Public Jira also "
      "hides private issues, about a quarter of the issue numbers in these projects, so the boards may be missing some "
      "work. The study shows the test works and catches both failures. It does not show agents can never add value.")
    d.save(G / "Plain_Explanation.docx")


def manual(e):
    d = docx.Document(G / "Rebuild_Manual.docx")
    clear(d)
    H = lambda t: d.add_paragraph(t, style="Heading 1")
    P = lambda t: d.add_paragraph(t, style="Normal")
    B = lambda t: d.add_paragraph(t, style="List Bullet")
    d.add_paragraph("Grounding the AI Scrum Master (CCECE 2027): how to rebuild the study by hand", style="Title")
    P("Every step to reproduce the 120-sprint replay and the numbers in the paper. Windows and PowerShell. "
      "Last updated 15 November 2026. The older 40-sprint pilot is in pilot/ and is rebuilt the same way with "
      "python -m pilot.build, python -m pilot.score and python -m pilot.rule_baseline.")
    H("1. Before you start")
    B("This folder must sit inside SwEng_202610 next to week-12-method-preliminary-results, "
      "phase-2-explainable-sprint-advisor and phase-3-in-sprint-early-warning, because the study reuses their code and data.")
    B("Python 3.13, scikit-learn 1.9.1, pandas 3.0, scipy and python-docx. Install each phase's requirements.txt first.")
    B("Run Phases 1 to 3 once (see the main README) so their result frames exist.")
    H("2. Build the boards, the models and the prompts")
    P("cd C:\\Project\\course-submission\\SwEng_202610\\ccece-2027-ai-scrum-master\npython -m study.build")
    P("About 15 seconds. It takes the 472 held-out sprints that had committed work and were still open at mid-sprint, "
      "draws 15 from each of the eight projects (seed 2027), rebuilds each board at the start and at halfway, scores the "
      "late-closure model and the unfinished-share model, and writes 72 prompt files (3 runs x 2 arms x 2 checkpoints x "
      "6 batches of 20) to study/prompts/run1-3.")
    H("3. Run the agent arms")
    P("Give each prompt file to a language-model agent that may read only that file. The paper's run used Claude "
      "(claude-opus-5-5) sub-agents in a Claude session on 15 November 2026, each told: \"Read the file <path> with the "
      "Read tool. Do not use any other tool and do not open any other file. Then reply with exactly what that file asks "
      "for.\" The prompt files were copied to a separate folder first, away from the labels.")
    P("python -m study.collect <folder with the sub-agent transcripts>")
    P("collect.py saves each answer word for word in study/agent_outputs/run1-3/ and writes audit.json with every tool "
      "call and the model of each run. It refuses any run that did more than read its own prompt. A rerun gives "
      "slightly different answers; that is expected.")
    H("4. Score")
    P("python -m study.score")
    P("About 3 minutes. Per run, arm and checkpoint: warn rate, sensitivity, specificity, AUC (pooled and within "
      "projects), Brier score, agreement and kappa with M, and Spearman and mean absolute error for the unfinished share. "
      "Then run-to-run agreement, bootstrap intervals (resampling sprints within project) and the cost curve. Writes "
      "study/results/summary.json, arm_metrics_by_run.csv, cost_curve.csv and messages.csv (every warning with its board).")
    H("5. Check what the agents said")
    P("python -m study.factcheck")
    P("Traces every number in every warning to its board (also for the 81 pilot warnings). Writes factcheck_auto.csv "
      "and .json. Read any flagged warning by hand; in our run 2 were flagged, one was advice and one a real error.")
    P("python -m study.rate build")
    P("Writes 40 rater prompt files (raters A and B, 25 warnings each, same warnings in different orders, without arm, "
      "run or outcome). Give each to a separate agent as in step 3, then:")
    P("python -m study.rate collect <transcript folder>\npython -m study.rate score")
    P("Writes ratings.csv and ratings_summary.json: unsupported claims and close-on-date advice per arm, rater agreement, "
      "and agreement with the pilot's original hand coding.")
    H("5b. The model-output ablation and the cost scenarios (added 15 Nov evening)")
    P("python -m study.ablation build")
    P("Writes 24 prompt files to study/ablation/prompts: the grounded agent's boards with the risk-model line changed "
      "only. P shows the probability without the threshold or verdict; F mirrors the probability around the threshold "
      "so the stated verdict is the opposite of the real one. Give each file to an agent as in step 3 (folder "
      "/home/claude/agentbox/ablation in our run, description \"ablation <prompt>\"), then:")
    P("python -m study.ablation collect <transcript folder>\npython -m study.ablation score\npython -m study.ablation intervals")
    P("Writes study/ablation/outputs (answers and audit.json; 24 of 24 clean in our run), ablation_points.csv and "
      "results/ablation_summary.json. Then:")
    P("python -m study.scenarios")
    P("Writes results/cost_scenarios.csv and .json: costs under three effectiveness scenarios (zero, default 0.5/0.25, "
      "strong 0.8/0.5) with never-warn and always-warn, and the cost-ratio band where each arm beats both.")
    H("5c. Second-review checks (added 15 Nov, late)")
    P("python -m study.review2\npython -m study.train_side")
    P("review2 writes results/review2.json: the flipped-call split (followed when the false call was wrong or right), "
      "the 13 sprints shared with the pilot and the results without them, which projects enter the within-project AUC, "
      "and the model's planning AUC on the replay pool against all held-out sprints. train_side writes "
      "results/train_side.json and cost_optimal_M.csv: the model's AUC and error from 5-fold cross-validation on the "
      "training sprints, and M's thresholds tuned to each cost scenario and ratio on training sprints.")
    P("python -m study.sensitivity build")
    P("Writes 36 prompt files to study/sensitivity/prompts: T (performance lines from training data), C (true number, "
      "opposite written verdict) and F2 (a second flipped run). Copy them to a separate folder "
      "(/home/claude/agentbox/sens in our run, description \"sens <prompt>\"), give each to an agent as in step 3, then:")
    P("python -m study.sensitivity collect <transcript folder>\npython -m study.sensitivity score")
    P("Writes study/sensitivity/points.csv and results/sensitivity_summary.json (36 of 36 clean in our run).")
    P("python -m study.rate_masked build")
    P("Writes 32 rater prompt files (raters A2 and B2), with the model lines shown for every warning so the arm cannot "
      "be guessed. Copy them to /home/claude/agentbox/rate2/A2 and B2 (description \"rate2 <prompt>\"), run as in step 3, then:")
    P("python -m study.rate_masked collect <transcript folder>\npython -m study.rate_masked score")
    P("Writes results/ratings_masked.csv and ratings_masked_summary.json, with the first rating's counts and "
      "board-resampled intervals side by side (32 of 32 clean in our run).")
    H("6. Check the numbers against the paper")
    P("python -m study.paper_numbers check")
    P(f"Checks all {len(e)} study numbers quoted in the text and every row of the results table against the result "
      "files. It must report 0 problems. Key numbers:")
    rows = [("Sprints (late)", f"{e['sample sprints']} ({e['sample late']}), pool {e['pool sprints']} ({e['pool late']})"),
            ("G agrees with M", f"{e['A_G percent'].replace(chr(92), '')}, kappa {e['kappa_G']}"),
            ("G extra warnings per run", f"{e['G extra warnings']}, all on on-time sprints"),
            ("G costs more than M", f"{e['cost gap 0.02']} F to {e['cost gap 0.5']} F per sprint"),
            ("L matches the rule at planning", f"{e['rule agree runs']} of 120"),
            ("M minus rule, AUC", f"{e['M-rule']} [{e['M-rule lo']}, {e['M-rule hi']}]"),
            ("Same call in all three runs (L / G)", f"{e['same call L']} / {e['same call G']}".replace(chr(92), "")),
            ("Unfinished share at halfway, Spearman (M / G / L)", f"{e['unf M c50']} / {e['unf G c50']} / {e['unf L c50']}"),
            ("Numbers traced to the board", f"{e['numbers traced']} of {e['numbers']}"),
            ("Unsupported claims, masked rating, both raters", f"{e['masked both L']}; {e['masked both G']}"),
            ("Unsupported claims, first rating", e['first rating both'] + "; " + e['first rating either']),
            ("Close-on-date advice, masked rating", f"{e['masked close L']}; {e['masked close G']}"),
            ("Masked rater agreement", e['masked kappa']),
            ("Verdict hidden: agrees with the model's real call", f"{e['P agree']}, difference from G {e['P-G']} [{e['P-G lo']}, {e['P-G hi']}]".replace(chr(92), "")),
            ("Verdict flipped: follows the wrong call", f"{e['F follow n']}; {e['F-L']} than the agent alone {e['F-L lo']} {e['F-L hi']}"),
            ("Verdict flipped: ranking at planning", f"{e['F auc c0']}, {e['F auc gap']}"),
            ("Cost band where M / G beat both trivial policies", f"{e['M band']} / {e['G band']}"),
            ("G minus M, strong scenario", e['strong gap']),
            ("Second flipped run", e['F2 follow']),
            ("Flipped call followed when wrong / right", e['flip split']),
            ("Only the written verdict flipped", e['C follow']),
            ("Training-only performance lines", e['T agree']),
            ("Without the 13 pilot sprints", e['no pilot'] + "; " + e['no pilot auc'])]
    t = d.add_table(rows=1, cols=2)
    t.style = "Light Grid Accent 1"
    t.rows[0].cells[0].text, t.rows[0].cells[1].text = "Number", "Expected"
    for a, b in rows:
        r = t.add_row().cells
        r[0].text, r[1].text = a, b
    H("7. Numbers that come from the companion sprint-risk work")
    B("Model arm M: AUC 0.751 on all 786 held-out sprints, 0.634 within projects "
      "(ccece-2027-paper/reviewer_checks/late_threshold_checks.csv).")
    B("40% of late closures are one to three days late; sprints planned to end Friday or at the weekend close late 57% "
      "of the time against 28% otherwise (same source).")
    B("Leakage case studies: spillover AUC 0.83 to 1.00 with the sprint's own completion ratio, late-closure AUC 0.64 "
      "to 0.69 with oversampling before the split "
      "(week-12-method-preliminary-results/sprint-risk-prediction/results/leakage_case_studies.json).")
    B("Hidden issues: a median of 26% of issue numbers since each project's first sprint issue are missing from public "
      "Jira (ccece-2027-paper/reviewer_checks/hidden_issues.csv). Quoted in Limits of the replay.")
    H("8. Build the PDF")
    P("Paste CCECE_2027_Grounding_AI_Scrum_Master.tex into the Overleaf project \"Grounding the AI Scrum Master - "
      "CCECE 2027\", compile with IEEEtran, check it is at most 6 pages (exactly 6 on 15 Nov, last page references only) with no warnings, and download the PDF into this "
      "folder.")
    H("9. Push")
    P("cd C:\\Project\\course-submission\\SwEng_202610\ngit add ccece-2027-ai-scrum-master\n"
      "git commit -m \"AI Scrum Master paper: second-review fixes, masked re-rating and sensitivity runs\"\ngit push")
    d.save(G / "Rebuild_Manual.docx")


if __name__ == "__main__":
    e = expected()
    plain(e)
    manual(e)
    print("guides written")
