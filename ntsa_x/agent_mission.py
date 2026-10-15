"""NTSA-X Milestone 12: bounded task loop over the Milestone 11 workbench.

This is a deterministic, inspectable controller for a *small set* of task
families, NOT a learned autonomous planning model. It checks actual receipts,
aborts on failed verifications, and only branches to pre-authorized tools.

An explicit mission digest binds goal, parameters, workspace and host allowlist.
Consequential tools also require a matching independent capability flag.
Remote/tool text never gains authority to propose new tools or permissions.
"""
from __future__ import annotations

from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path
import json
import time

from .agent_workbench import (
    Workbench, WorkbenchError, CapabilityFlags, EFFECT_OPS, canonical_bytes,
)

MISSION_VERSION = 1
CONTROLLER_VERSION = "ntsa-m12-deterministic-v1"
MAX_STEPS = 7
GOALS = frozenset(("repository.audit", "code.inspect",
                   "tests.diagnose", "research.inspect"))
INPUT_KEYS = {
    "repository.audit": frozenset(),
    "code.inspect": frozenset(("path", "term")),
    "tests.diagnose": frozenset(("pattern", "timeout_seconds")),
    "research.inspect": frozenset(("url", "browser_fallback")),
}


def validate_mission(mission: object) -> dict:
    if (not isinstance(mission, dict)
            or set(mission) != {"version", "goal", "inputs"}
            or type(mission["version"]) is not int
            or mission["version"] != MISSION_VERSION
            or not isinstance(mission["goal"], str)
            or mission["goal"] not in GOALS
            or not isinstance(mission["inputs"], dict)):
        raise WorkbenchError("Mission must contain version, supported goal and inputs")
    goal, args = mission["goal"], mission["inputs"]
    if not set(args).issubset(INPUT_KEYS[goal]):
        raise WorkbenchError("Unexpected mission input")
    if goal == "code.inspect" and "path" not in args:
        raise WorkbenchError("code.inspect requires path")
    if goal == "research.inspect" and "url" not in args:
        raise WorkbenchError("research.inspect requires url")
    for key, value in args.items():
        if key == "browser_fallback":
            if type(value) is not bool:
                raise WorkbenchError("browser_fallback must be boolean")
        elif key == "timeout_seconds":
            if type(value) is not int or not 1 <= value <= 120:
                raise WorkbenchError("Invalid test timeout")
        elif (not isinstance(value, str) or not 1 <= len(value) <= 1024):
            raise WorkbenchError("Mission inputs must be bounded strings")
    if len(canonical_bytes(mission)) > 4096:
        raise WorkbenchError("Mission exceeds 4 KB")
    return mission


def _step(tool: str, args: dict | None = None) -> dict:
    return {"tool": tool, "args": args or {}}


def potential_steps(mission: dict, wb: Workbench) -> list[dict]:
    """Enumerate every tool the deterministic branch logic could invoke."""
    goal, params = mission["goal"], mission["inputs"]
    if goal == "repository.audit":
        candidate = [_step("git.status"), _step("devops.inventory"),
                     _step("workspace.list", {"path": ".", "limit": 50})]
        if (wb.root / "README.md").is_file():
            candidate.append(_step("workspace.read", {"path": "README.md"}))
        return candidate
    if goal == "code.inspect":
        candidate = [_step("workspace.read", {"path": params["path"]}),
                     _step("git.diff", {"path": params["path"]})]
        if "term" in params:
            candidate.append(_step("workspace.search", {
                "pattern": params["term"], "limit": 12}))
        return candidate
    if goal == "tests.diagnose":
        options = {k: params[k] for k in ("pattern", "timeout_seconds")
                   if k in params}
        return [_step("git.status"), _step("python.tests", options)]
    if goal == "research.inspect":
        result = [_step("web.fetch", {"url": params["url"]})]
        if params.get("browser_fallback", False):
            result.append(_step("browser.snapshot", {"url": params["url"]}))
        return result
    raise WorkbenchError("Unsupported mission goal")


def _check_result(tool: str, value: object) -> tuple[bool, str]:
    """Only structural, observable predicates, never model-made truth claims."""
    if not isinstance(value, dict):
        return False, "Tool did not produce a structured observation"
    if tool in ("git.status", "git.diff", "python.tests"):
        if type(value.get("exit_code")) is not int:
            return False, "Command did not report an exit code"
        if value["exit_code"] != 0:
            return False, f"Command reported exit code {value['exit_code']}"
        if not isinstance(value.get("stdout"), str) or not isinstance(
                value.get("stderr"), str):
            return False, "Missing command output fields"
        return True, "Command exited successfully; semantics not independently verified"
    if tool == "devops.inventory":
        return (isinstance(value.get("configuration_files"), list),
                "Configuration inventory structure checked")
    if tool == "workspace.list":
        return isinstance(value.get("items"), list), "Directory listing structure checked"
    if tool == "workspace.read":
        return (isinstance(value.get("text"), str)
                and isinstance(value.get("path"), str),
                "Bounded file contents were returned")
    if tool == "workspace.search":
        return isinstance(value.get("matches"), list), "Search match structure checked"
    if tool == "web.fetch":
        return (isinstance(value.get("text"), str)
                and value.get("untrusted_external_content") is True,
                "Fetched public text is UNTRUSTED; not task instructions")
    if tool == "browser.snapshot":
        return (isinstance(value.get("text"), str)
                and value.get("read_only_browser_session") is True,
                "Browser text is UNTRUSTED; not task instructions")
    return False, "Unexpected tool in mission"


class MissionController:
    def __init__(self, workbench: Workbench):
        self.wb = workbench

    def prepare(self, mission: object) -> tuple[dict, list[dict], str]:
        data = validate_mission(mission)
        actions = potential_steps(data, self.wb)
        if not 1 <= len(actions) <= MAX_STEPS:
            raise WorkbenchError("Mission exceeds step budget")
        # Fully validate the maximum set of potential tools *before* running.
        for action in actions:
            self.wb.prepare({"version": 1, "steps": [action]})
        digest = sha256(canonical_bytes({
            "controller": CONTROLLER_VERSION, "mission": data,
            "workspace": str(self.wb.root),
            "allowed_hosts": self.wb.allowed_hosts,
            "potential_steps": actions,
        })).hexdigest()
        return data, actions, digest

    def review(self, mission: object) -> dict:
        data, actions, digest = self.prepare(mission)
        return {
            "mode": "review", "controller": CONTROLLER_VERSION,
            "goal": data["goal"], "mission_sha256": digest,
            "workspace": str(self.wb.root),
            "possible_steps": [
                {"tool": row["tool"],
                 "effect": EFFECT_OPS.get(row["tool"], "read-only"),
                 "args": row["args"]} for row in actions],
            "note": "No work was executed; approve this exact mission digest.",
        }

    def execute(self, mission: object, *, approval: str | None = None) -> dict:
        data, actions, digest = self.prepare(mission)
        effects = {EFFECT_OPS[action["tool"]]
                   for action in actions if action["tool"] in EFFECT_OPS}
        if effects:
            if approval != digest:
                raise WorkbenchError("Exact reviewed mission digest approval required")
            for cap in effects:
                if not getattr(self.wb.flags, cap):
                    raise WorkbenchError(f"Capability {cap} disabled")
        receipt = {
            "mode": "executed", "controller": CONTROLLER_VERSION,
            "goal": data["goal"], "mission_sha256": digest,
            "workspace": str(self.wb.root), "steps": [],
            "status": "running", "verified_external_truth": False,
        }
        read_results = {}
        for number, action in enumerate(actions, start=1):
            # Dynamic branch: skip HTML browser if bounded fetch already
            # provided substantive page text. Never follow embedded links.
            if action["tool"] == "browser.snapshot":
                fetched = read_results.get("web.fetch")
                if fetched is not None and len(fetched.get("text", "").strip()) >= 160:
                    receipt["steps"].append({
                        "step": number, "tool": action["tool"],
                        "status": "skipped", "reason": "Static fetch already contained text"})
                    continue
            # After the first step, abort on failures. No auto retry of tests,
            # writes or network effects. Observations cannot change tool args.
            start = time.monotonic()
            try:
                # The whole mission and all effects were independently approved
                # above. Do not self-generate/forge a Workbench plan approval.
                observed = self.wb.dispatch(action["tool"], action["args"])
                ok, rationale = _check_result(action["tool"], observed)
                if not ok:
                    receipt["steps"].append({
                        "step": number, "tool": action["tool"],
                        "status": "failed", "reason": rationale,
                        "observation": observed,
                        "elapsed_ms": round((time.monotonic()-start)*1000)})
                    receipt["status"] = "failed"
                    break
                receipt["steps"].append({
                    "step": number, "tool": action["tool"],
                    "status": "observed", "verification": rationale,
                    "observation": observed,
                    "elapsed_ms": round((time.monotonic()-start)*1000)})
                read_results[action["tool"]] = observed
            except (OSError, ValueError) as exc:
                receipt["steps"].append({
                    "step": number, "tool": action["tool"], "status": "failed",
                    "error_type": type(exc).__name__,
                    "message": str(exc)[:250]})
                receipt["status"] = "failed"
                break
        if receipt["status"] == "running":
            receipt["status"] = "completed"
        if receipt["status"] == "failed":
            failure_tool = receipt["steps"][-1]["tool"]
            receipt["recommended_recovery"] = {
                "python.tests": "Inspect failing test output; propose a NEW reviewed code inspection or repair plan. No automatic edits or reruns.",
                "web.fetch": "Inspect allowed hostname and network error; propose a NEW reviewed mission. No automatic URL changes.",
                "browser.snapshot": "Inspect browser availability and allowed hostname; propose a NEW reviewed mission.",
                "workspace.read": "Check requested workspace path or file access; propose a NEW reviewed mission.",
                "git.diff": "Check local Git configuration or requested source path; propose a NEW reviewed mission.",
                "git.status": "Check Git initialization and repository access; propose a NEW reviewed mission.",
            }.get(failure_tool, "Inspect the recorded observation, then propose a NEW reviewed mission.")
        receipt["observed_steps"] = sum(
            row["status"] == "observed" for row in receipt["steps"])
        receipt["skipped_steps"] = sum(
            row["status"] == "skipped" for row in receipt["steps"])
        receipt["result_integrity_sha256"] = sha256(
            canonical_bytes({k: v for k, v in receipt.items()
                             if k != "result_integrity_sha256"})).hexdigest()
        return receipt


def main():
    p = ArgumentParser(description="NTSA-X bounded plan-observe-verify agent controller")
    p.add_argument("--workspace", required=True)
    p.add_argument("--mission", required=True, help="Existing mission JSON")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--approve-digest")
    p.add_argument("--allow-host", action="append", default=[])
    p.add_argument("--enable-tests", action="store_true")
    p.add_argument("--enable-network", action="store_true")
    p.add_argument("--enable-browser", action="store_true")
    args = p.parse_args()
    try:
        path = Path(args.mission).expanduser()
        if not path.is_file() or path.stat().st_size > 4096:
            raise WorkbenchError("Mission JSON is missing or too large")
        mission = json.loads(path.read_text(encoding="utf-8-sig"))
        wb = Workbench(args.workspace, allowed_hosts=tuple(args.allow_host),
                       flags=CapabilityFlags(
                           tests=args.enable_tests, network=args.enable_network,
                           browser=args.enable_browser))
        controller = MissionController(wb)
        result = (controller.execute(mission, approval=args.approve_digest)
                  if args.execute else controller.review(mission))
        print(json.dumps(result, indent=2, ensure_ascii=False))
        if args.execute and result["status"] != "completed":
            raise SystemExit(1)
    except (ValueError, UnicodeError, OSError) as exc:
        p.error(str(exc))


if __name__ == "__main__":
    main()
