"""Step 2 of the study: save each agent's answer word for word and audit its tool calls.

Each agent run was a Claude sub-agent (model claude-opus-5-5) given one prompt file and told to
read only that file and use no other tool. This script reads the sub-agent transcripts, keeps the
final answer verbatim as agent_outputs/run<k>/<prompt>.jsonl, and writes agent_outputs/audit.json
with every tool call each agent made and the model that answered. Any run whose tool calls are not
exactly one Read of its own prompt file is flagged and must be rerun.

Usage:  python -m study.collect <folder with the sub-agent transcripts>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import study

OUT = study.HERE / "agent_outputs"


def read_transcript(jsonl: Path):
    calls, models, answer = [], set(), None
    for line in jsonl.read_text().splitlines():
        d = json.loads(line)
        m = d.get("message")
        if not isinstance(m, dict):
            continue
        if m.get("model"):
            models.add(m["model"])
        for c in m.get("content") or []:
            if isinstance(c, dict) and c.get("type") == "tool_use":
                if c["name"] == "SubagentHandback":
                    answer = c["input"].get("message")
                else:
                    calls.append({"tool": c["name"], "input": c["input"]})
    return calls, sorted(models), answer


def main(folder: str):
    folder = Path(folder)
    manifest = json.loads((study.HERE / "prompts" / "manifest.json").read_text())
    want = {f"run{m['run']} {m['prompt']}": m for m in manifest}
    audit, problems = [], []
    found = {}
    for meta in sorted(folder.glob("agent-*.meta.json")):
        desc = json.loads(meta.read_text()).get("description", "")
        if desc not in want:
            continue
        tr = meta.with_name(meta.name.replace(".meta.json", ".jsonl"))
        calls, models, answer = read_transcript(tr)
        found.setdefault(desc, []).append((tr.stat().st_mtime, calls, models, answer, meta.stem))
    for desc, m in want.items():
        if desc not in found:
            problems.append(f"missing: {desc}")
            continue
        # if a prompt was run more than once (a rerun after a failed audit), keep the latest clean one
        runs = sorted(found[desc])
        mtime, calls, models, answer, agent = runs[-1]
        expected = f"/home/claude/agentbox/run{m['run']}/{m['prompt']}.txt"
        clean = (len(calls) == 1 and calls[0]["tool"] == "Read"
                 and calls[0]["input"].get("file_path") == expected)
        lines = [l for l in (answer or "").splitlines() if l.strip()]
        try:
            ids = sorted(json.loads(l)["id"] for l in lines)
        except (json.JSONDecodeError, KeyError):
            ids = None
        complete = ids == sorted(m["boards"])
        audit.append({"run": m["run"], "prompt": m["prompt"], "attempts": len(runs), "models": models,
                      "tool_calls": calls, "only_read_own_prompt": clean, "answers_complete": complete})
        if not (clean and complete):
            problems.append(f"bad: {desc} clean={clean} complete={complete}")
            continue
        d = OUT / f"run{m['run']}"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{m['prompt']}.jsonl").write_text("\n".join(lines) + "\n")
    (OUT / "audit.json").write_text(json.dumps(audit, indent=1))
    print(f"{len(audit)} of {len(want)} prompts collected; problems: {len(problems)}")
    for p in problems:
        print(" ", p)


if __name__ == "__main__":
    main(sys.argv[1])
