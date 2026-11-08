@echo off
rem Pushes the 8 Nov 2026 revision of the Grounding paper. Deletes itself when done.
cd /d "%~dp0.."
echo Repo: %CD%
git add ccece-2027-ai-scrum-master/CCECE_2027_Grounding_AI_Scrum_Master.tex ccece-2027-ai-scrum-master/README.md ccece-2027-ai-scrum-master/pilot/rule_baseline.py ccece-2027-ai-scrum-master/pilot/results/rule_baseline.json
git commit -m "Grounding paper: kappa, past-lateness rule, Scrum Guide reframe, reference fixes" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_01GtnMujD8tajLNudb21Eqni"
git push
echo.
echo Done. Check the lines above for errors, then press a key.
pause
(goto) 2>nul & del "%~f0"
