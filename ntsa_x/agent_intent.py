"""Conservative English-to-tool router for a few explicit local operator intents.

No LLM required. Ambiguous commands fail closed. This is NOT natural-language
fluency; learned NTSA-X model proposals must satisfy the same plan validator.
"""
from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import json
import re

from .agent_workbench import WorkbenchError, validate_plan


def propose(request: str) -> dict:
    if not isinstance(request, str) or not 1 <= len(request) <= 400:
        raise WorkbenchError("Request must be 1-400 characters")
    phrase = request.strip()
    fixed = {
        "git status": ("git.status", {}),
        "show git status": ("git.status", {}),
        "inspect devops": ("devops.inventory", {}),
        "inspect deployment configs": ("devops.inventory", {}),
        "list files": ("workspace.list", {"path": "."}),
        "run tests": ("python.tests", {"pattern": "test_*.py"}),
        "run all tests": ("python.tests", {"pattern": "test_*.py"}),
    }
    if phrase.casefold() in fixed:
        tool, args = fixed[phrase.casefold()]
    else:
        rules = [
            (r"(?:read file|open file)\s+(.+)", "workspace.read", "path"),
            (r"(?:show diff for|git diff)\s+(.+)", "git.diff", "path"),
            (r"(?:search files for|find text)\s+(.+)", "workspace.search", "pattern"),
            (r"list files in\s+(.+)", "workspace.list", "path"),
            (r"(?:fetch page|research page)\s+(https://\S+)", "web.fetch", "url"),
            (r"(?:inspect website|browser snapshot)\s+(https://\S+)",
             "browser.snapshot", "url"),
        ]
        for rule, selected, key in rules:
            match = re.fullmatch(rule, phrase, flags=re.IGNORECASE)
            if match:
                tool, args = selected, {key: match.group(1).strip()}
                break
        else:
            raise WorkbenchError(
                "Ambiguous request. Use a supported explicit command "
                "or author a reviewed JSON plan.")
    return validate_plan({"version": 1, "steps": [
        {"tool": tool, "args": args}]})


def main():
    parser = ArgumentParser(description="Conservative NTSA-X command-to-tool proposal")
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", required=True,
                        help="New JSON plan file (refuses overwrite)")
    a = parser.parse_args()
    try:
        suggestion = propose(a.request)
        output = Path(a.output).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as destination:
            json.dump(suggestion, destination, ensure_ascii=False, indent=2)
            destination.write("\n")
        print(f"Plan proposed; NOT EXECUTED: {output}")
    except (OSError, WorkbenchError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
