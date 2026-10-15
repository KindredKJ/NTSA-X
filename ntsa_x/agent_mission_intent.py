"""Narrow English-to-mission proposal; emits JSON only, never executes tools.

This is a deterministic UX shim until language-trained NTSA-X reliably emits
typed plans that pass independent evaluation. It must never grant permissions.
"""
from __future__ import annotations
from argparse import ArgumentParser
from pathlib import Path
import json
import re

from .agent_mission import validate_mission
from .agent_workbench import WorkbenchError


def propose_mission(request: str) -> dict:
    if not isinstance(request, str) or not 1 <= len(request) <= 400:
        raise WorkbenchError("Request must be 1-400 characters")
    value = request.strip()
    if value.casefold() in ("audit repository", "inspect repository health"):
        return validate_mission(
            {"version": 1, "goal": "repository.audit", "inputs": {}})
    if value.casefold() in ("diagnose tests", "run and diagnose tests"):
        return validate_mission(
            {"version": 1, "goal": "tests.diagnose", "inputs": {}})
    options = [
        (r"inspect code\s+(\S+)(?:\s+for\s+([^\r\n]+))?",
         "code.inspect"),
        (r"research page\s+(https://\S+)", "research.inspect"),
        (r"research browser\s+(https://\S+)", "research.inspect"),
    ]
    for pattern, goal in options:
        match = re.fullmatch(pattern, value, re.IGNORECASE)
        if not match:
            continue
        if goal == "code.inspect":
            inputs = {"path": match.group(1)}
            if match.group(2):
                inputs["term"] = match.group(2).strip()
        else:
            inputs = {"url": match.group(1)}
            if value.casefold().startswith("research browser"):
                inputs["browser_fallback"] = True
        return validate_mission(
            {"version": 1, "goal": goal, "inputs": inputs})
    raise WorkbenchError(
        "Unsupported or ambiguous mission. Use a known explicit phrase; "
        "no tool execution was authorized.")


def main():
    p = ArgumentParser(description="Propose a bounded NTSA-X mission from English")
    p.add_argument("--request", required=True)
    p.add_argument("--output", required=True, help="New JSON mission (no overwrite)")
    a = p.parse_args()
    try:
        mission = propose_mission(a.request)
        file = Path(a.output).expanduser()
        file.parent.mkdir(parents=True, exist_ok=True)
        with file.open("x", encoding="utf-8") as output:
            json.dump(mission, output, ensure_ascii=False, indent=2)
            output.write("\n")
        print(f"Mission proposed; NOT EXECUTED: {file}")
    except (OSError, WorkbenchError) as exc:
        p.error(str(exc))


if __name__ == "__main__":
    main()
