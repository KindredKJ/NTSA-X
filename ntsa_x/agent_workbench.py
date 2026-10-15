"""NTSA-X agent workbench: typed, deny-by-default local tool execution.

Plans are explicit JSON artifacts. A weak or compromised model is NOT granted
unattended tools: high-risk execution requires an exact operator-approved plan
digest AND corresponding opt-in capability flag. This is a research control
plane, not an OS-level sandbox or a guarantee against hostile project code.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time

from .agent_web import public_https_url, fetch_text, browser_snapshot

MAX_PLAN_BYTES = 200_000
MAX_READ_BYTES = 200_000
MAX_WRITE_BYTES = 100_000
MAX_REPLY_CHARS = 12_000
FILE_SUFFIXES = frozenset({
    ".py", ".pyi", ".md", ".txt", ".json", ".jsonl", ".yaml", ".yml",
    ".toml", ".js", ".jsx", ".ts", ".tsx", ".html", ".css", ".ps1",
    ".sh", ".rs", ".go", ".java", ".sql", ".xml", ".ini", ".cfg",
})
DENIED_PARTS = frozenset({
    ".git", ".local", ".kindred", ".venv", "venv", "node_modules",
    "__pycache__", ".ssh", ".aws", ".azure", "secrets", "credentials",
})
DENIED_FILENAMES = frozenset({
    ".env", ".env.local", ".env.production", ".npmrc", ".pypirc",
    "id_rsa", "id_ed25519", "known_hosts", "credentials.json",
    "service-account.json", "token.json", "config.json.secret",
})
SAFE_OPS = frozenset({
    "workspace.list", "workspace.read", "workspace.search",
    "git.status", "git.diff", "devops.inventory",
})
EFFECT_OPS = {
    "workspace.write": "write",
    "python.tests": "tests",
    "web.fetch": "network",
    "browser.snapshot": "browser",
}
SUPPORTED_OPS = SAFE_OPS | set(EFFECT_OPS)
TOOL_PARAMETERS = {
    "workspace.list": {"path", "limit"},
    "workspace.read": {"path"},
    "workspace.search": {"pattern", "limit"},
    "workspace.write": {"path", "content", "expected_sha256"},
    "git.status": set(),
    "git.diff": {"path"},
    "devops.inventory": set(),
    "python.tests": {"pattern", "timeout_seconds"},
    "web.fetch": {"url"},
    "browser.snapshot": {"url"},
}


class WorkbenchError(ValueError):
    """A denied, invalid or unsuccessful workbench action."""


def canonical_bytes(obj: object) -> bytes:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def validate_plan(plan: object) -> dict:
    if not isinstance(plan, dict) or set(plan) != {"version", "steps"}:
        raise WorkbenchError("Plan must contain version and steps only")
    if type(plan["version"]) is not int or plan["version"] != 1:
        raise WorkbenchError("Unsupported plan version")
    steps = plan["steps"]
    if not isinstance(steps, list) or not 1 <= len(steps) <= 20:
        raise WorkbenchError("Plan must have 1-20 steps")
    for step in steps:
        if (not isinstance(step, dict) or set(step) != {"tool", "args"}
                or not isinstance(step["tool"], str)
                or step["tool"] not in SUPPORTED_OPS
                or not isinstance(step["args"], dict)
                or not set(step["args"]).issubset(TOOL_PARAMETERS[step["tool"]])):
            raise WorkbenchError("Unrecognized tool or parameters")
        args = step["args"]
        op = step["tool"]
        required = {
            "workspace.list": set(), "workspace.read": {"path"},
            "workspace.search": {"pattern"}, "workspace.write": {"path", "content"},
            "git.status": set(), "git.diff": {"path"},
            "devops.inventory": set(), "python.tests": set(),
            "web.fetch": {"url"}, "browser.snapshot": {"url"},
        }[op]
        if not required.issubset(args):
            raise WorkbenchError(f"Missing required arguments for {op}")
        for value in args.values():
            if type(value) not in (str, int):
                raise WorkbenchError("Tool arguments must be literal strings or integers")
        if len(canonical_bytes(args)) > MAX_WRITE_BYTES + 400:
            raise WorkbenchError("Oversized tool arguments")
        for key in ("path", "pattern", "url"):
            if key in args and (not isinstance(args[key], str)
                                or not 1 <= len(args[key]) <= 1024):
                raise WorkbenchError(f"Invalid {key}")
        if "limit" in args and (type(args["limit"]) is not int
                                or not 1 <= args["limit"] <= 100):
            raise WorkbenchError("Invalid list/search limit")
        if op == "workspace.write":
            if (not isinstance(args["content"], str)
                    or len(args["content"].encode("utf-8")) > MAX_WRITE_BYTES
                    or (args.get("expected_sha256") is not None and
                        not re.fullmatch(r"[0-9a-f]{64}", str(args["expected_sha256"])))):
                raise WorkbenchError("Invalid write content or expected SHA-256")
        if op == "python.tests":
            if ("pattern" in args and not re.fullmatch(
                    r"test_[A-Za-z0-9_]*\.py", args["pattern"])):
                raise WorkbenchError("Only unittest test_*.py patterns allowed")
            if ("timeout_seconds" in args
                    and (type(args["timeout_seconds"]) is not int or
                         not 1 <= args["timeout_seconds"] <= 120)):
                raise WorkbenchError("Test timeout must be 1-120 seconds")
    if len(canonical_bytes(plan)) > MAX_PLAN_BYTES:
        raise WorkbenchError("Plan too large")
    return plan


def plan_digest(plan: dict, workspace: Path, allowed_hosts: tuple[str, ...]) -> str:
    return sha256(canonical_bytes({
        "plan": plan, "workspace": str(workspace),
        "allowed_hosts": sorted(allowed_hosts),
    })).hexdigest()


@dataclass(frozen=True)
class CapabilityFlags:
    write: bool = False
    tests: bool = False
    network: bool = False
    browser: bool = False


class Workbench:
    def __init__(self, workspace: str | Path, *,
                 allowed_hosts: tuple[str, ...] = (),
                 flags: CapabilityFlags = CapabilityFlags()):
        root = Path(workspace).expanduser().resolve(strict=True)
        if not root.is_dir():
            raise WorkbenchError("Workspace must be an existing directory")
        self.root = root
        self.flags = flags
        for host in allowed_hosts:
            if (not isinstance(host, str)
                    or not re.fullmatch(r"[a-z0-9](?:[a-z0-9.-]{0,251}[a-z0-9])?", host)
                    or host.startswith(".") or host.endswith(".")):
                raise WorkbenchError("Invalid exact allowed hostname")
        self.allowed_hosts = tuple(sorted(set(allowed_hosts)))

    def path(self, relative: str, *, directory: bool = False,
             new_file: bool = False) -> Path:
        if relative == "." and directory:
            return self.root
        if (not isinstance(relative, str) or not 1 <= len(relative) <= 1024
                or Path(relative).is_absolute() or "\x00" in relative
                or re.match(r"^[A-Za-z]:", relative)):
            raise WorkbenchError("Expected relative workspace path")
        # Forbid sensitive path segments before and after resolving symlinks.
        parts = Path(relative.replace("\\", "/")).parts
        if (not parts or any(p in ("..", "") or p.lower() in DENIED_PARTS
                             or p.lower() in DENIED_FILENAMES
                             or p.lower().startswith(".env") for p in parts)):
            raise WorkbenchError("Sensitive or traversal path denied")
        requested = self.root.joinpath(*parts).resolve(strict=False)
        if not requested.is_relative_to(self.root):
            raise WorkbenchError("Outside-workspace path denied")
        checked_parts = requested.relative_to(self.root).parts
        if any(p.lower() in DENIED_PARTS or p.lower() in DENIED_FILENAMES
               or p.lower().startswith(".env") for p in checked_parts):
            raise WorkbenchError("Sensitive resolved path denied")
        if requested.is_symlink():
            raise WorkbenchError("Symlink target denied")
        if directory:
            if not requested.is_dir():
                raise WorkbenchError("Directory does not exist")
        elif new_file:
            if requested.suffix.lower() not in FILE_SUFFIXES:
                raise WorkbenchError("File extension not supported")
        elif not requested.is_file() or requested.suffix.lower() not in FILE_SUFFIXES:
            raise WorkbenchError("File does not exist or extension denied")
        return requested

    def prepare(self, plan: object):
        data = validate_plan(plan)
        # Validate every requested path and host BEFORE any tool runs.
        for step in data["steps"]:
            op, args = step["tool"], step["args"]
            if op == "workspace.list":
                self.path(args.get("path", "."), directory=True)
            elif op in {"workspace.read", "git.diff"}:
                self.path(args["path"])
            elif op == "workspace.write":
                self.path(args["path"], new_file=True)
            elif op in {"web.fetch", "browser.snapshot"}:
                public_https_url(args["url"], self.allowed_hosts)
            elif op == "python.tests" and not (self.root / "tests").is_dir():
                raise WorkbenchError("No tests/ directory for unittest")
        return data

    def digest(self, plan: object) -> str:
        return plan_digest(self.prepare(plan), self.root, self.allowed_hosts)

    def run(self, plan: object, *, execute: bool = False,
            approve_digest: str | None = None) -> dict:
        data = self.prepare(plan)
        digest = plan_digest(data, self.root, self.allowed_hosts)
        if not execute:
            return {"mode": "review", "plan_sha256": digest,
                    "workspace": str(self.root),
                    "steps": [
                        {"tool": row["tool"],
                         "effect": EFFECT_OPS.get(row["tool"], "read-only")}
                        for row in data["steps"]],
                    "note": "Dry-run only: no tools were executed."}
        sensitive = [step["tool"] for step in data["steps"]
                     if step["tool"] in EFFECT_OPS]
        if sensitive:
            if approve_digest != digest:
                raise WorkbenchError("Exact plan digest approval required")
            for tool in sensitive:
                cap = EFFECT_OPS[tool]
                if not getattr(self.flags, cap):
                    raise WorkbenchError(f"Capability {cap} disabled")
        results = []
        for index, step in enumerate(data["steps"]):
            begun = time.monotonic()
            try:
                outcome = self.dispatch(step["tool"], step["args"])
                results.append({"step": index + 1, "tool": step["tool"],
                                "ok": True, "result": outcome,
                                "elapsed_ms": round((time.monotonic() - begun) * 1000)})
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                # Never continue executing later effects after any failure.
                results.append({"step": index + 1, "tool": step["tool"],
                                "ok": False, "error": type(exc).__name__,
                                "message": str(exc)[:250]})
                break
        return {
            "mode": "executed", "workspace": str(self.root),
            "plan_sha256": digest, "results": results,
            "completed": len(results) == len(data["steps"])
                         and all(item["ok"] for item in results),
            "results_are_verified_truth": False,
        }

    def dispatch(self, tool: str, args: dict):
        if tool == "workspace.list":
            folder = self.path(args.get("path", "."), directory=True)
            items = []
            for child in sorted(folder.iterdir(), key=lambda c: c.name):
                try:
                    rel = child.relative_to(self.root).as_posix()
                    if child.is_symlink():
                        continue
                    self.path(rel, directory=child.is_dir())
                except (WorkbenchError, OSError):
                    continue
                items.append({"name": child.name, "kind": "directory"
                              if child.is_dir() else "file"})
                if len(items) >= args.get("limit", 50):
                    break
            return {"items": items}
        if tool == "workspace.read":
            path = self.path(args["path"])
            if path.stat().st_size > MAX_READ_BYTES:
                raise WorkbenchError("File exceeds read limit")
            return {"path": args["path"], "text": path.read_text(encoding="utf-8")[:MAX_REPLY_CHARS],
                    "truncated": path.stat().st_size > MAX_REPLY_CHARS}
        if tool == "workspace.search":
            pattern = args["pattern"]
            if len(pattern) > 128 or "\n" in pattern:
                raise WorkbenchError("Search pattern exceeds bound")
            matches, limit = [], args.get("limit", 20)
            for current, folders, filenames in os.walk(self.root, followlinks=False):
                folders[:] = [f for f in folders if f.lower() not in DENIED_PARTS
                              and not (Path(current) / f).is_symlink()]
                for name in sorted(filenames):
                    if len(matches) >= limit:
                        break
                    rel = (Path(current) / name).relative_to(self.root).as_posix()
                    try:
                        file = self.path(rel)
                        if file.stat().st_size > MAX_READ_BYTES:
                            continue
                        for line_no, line in enumerate(file.read_text(
                                encoding="utf-8").splitlines(), 1):
                            if pattern.casefold() in line.casefold():
                                matches.append({
                                    "path": rel, "line": line_no,
                                    "preview": line[:180]})
                            if len(matches) >= limit:
                                break
                    except (WorkbenchError, UnicodeError, OSError):
                        continue
                if len(matches) >= limit:
                    break
            return {"matches": matches}
        if tool == "workspace.write":
            file = self.path(args["path"], new_file=True)
            parent = file.parent
            if not parent.is_dir():
                raise WorkbenchError("Write parent directory must exist")
            if file.is_dir() or file.is_symlink():
                raise WorkbenchError("Cannot write a directory or symlink")
            if file.exists() and file.stat().st_size > MAX_READ_BYTES:
                raise WorkbenchError("Existing file too large for guarded overwrite")
            existing = file.read_bytes() if file.exists() else None
            if existing is not None and sha256(existing).hexdigest() != args.get("expected_sha256"):
                raise WorkbenchError("Existing file requires matching original SHA-256")
            if existing is None and "expected_sha256" in args:
                raise WorkbenchError("Cannot specify original hash for new file")
            raw = args["content"].encode("utf-8")
            fd, staged = tempfile.mkstemp(prefix=".ntsa-stage-", dir=parent)
            try:
                with os.fdopen(fd, "wb") as stream:
                    stream.write(raw)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(staged, file)
            finally:
                if os.path.exists(staged):
                    os.unlink(staged)
            return {"path": args["path"], "bytes": len(raw),
                    "sha256": sha256(raw).hexdigest()}
        if tool == "git.status":
            return self._command(["git", "status", "--short", "--untracked-files=no"])
        if tool == "git.diff":
            self.path(args["path"])
            return self._command(["git", "diff", "--", args["path"]])
        if tool == "devops.inventory":
            markers = ("Dockerfile", "docker-compose.yml", "compose.yaml",
                       "pyproject.toml", "package.json", "requirements.txt",
                       "vercel.json", "wrangler.toml", "Makefile")
            files = [item for item in markers if (self.root / item).is_file()]
            workflows = self.root / ".github" / "workflows"
            # This is metadata-only: do not read workflow secrets.
            if workflows.is_dir():
                files.extend(".github/workflows/" + p.name
                             for p in sorted(workflows.glob("*.yml"))[:20])
            return {"configuration_files": files,
                    "note": "Inventory only; no cloud credentials or deployments inspected"}
        if tool == "python.tests":
            cmd = [sys.executable, "-m", "unittest", "discover", "-s", "tests",
                   "-p", args.get("pattern", "test_*.py"), "-v"]
            return self._command(cmd, timeout=args.get("timeout_seconds", 60))
        if tool == "web.fetch":
            return fetch_text(args["url"], self.allowed_hosts)
        if tool == "browser.snapshot":
            return browser_snapshot(args["url"], self.allowed_hosts)
        raise WorkbenchError("Unknown tool")

    def _command(self, command: list[str], *, timeout: int = 30):
        try:
            result = subprocess.run(command, cwd=self.root, stdin=subprocess.DEVNULL,
                                    capture_output=True, text=True,
                                    encoding="utf-8", errors="replace",
                                    timeout=timeout, check=False)
        except subprocess.TimeoutExpired:
            raise WorkbenchError("Command timed out") from None
        return {"exit_code": result.returncode,
                "stdout": result.stdout[:MAX_REPLY_CHARS],
                "stderr": result.stderr[:4000],
                "truncated": len(result.stdout) > MAX_REPLY_CHARS}


def main():
    parser = argparse.ArgumentParser(description="NTSA-X capability-gated agent workbench")
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--plan", required=True, help="Explicit JSON step plan")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--approve-digest", help="Exact plan SHA-256 printed by review")
    parser.add_argument("--allow-host", action="append", default=[])
    parser.add_argument("--enable-write", action="store_true")
    parser.add_argument("--enable-tests", action="store_true")
    parser.add_argument("--enable-network", action="store_true")
    parser.add_argument("--enable-browser", action="store_true")
    args = parser.parse_args()
    file = Path(args.plan).expanduser()
    if not file.is_file() or file.stat().st_size > MAX_PLAN_BYTES:
        parser.error("Plan file not found or too large")
    try:
        data = json.loads(file.read_text(encoding="utf-8-sig"))
        wb = Workbench(args.workspace, allowed_hosts=tuple(args.allow_host),
                       flags=CapabilityFlags(
                           write=args.enable_write, tests=args.enable_tests,
                           network=args.enable_network,
                           browser=args.enable_browser))
        receipt = wb.run(data, execute=args.execute,
                         approve_digest=args.approve_digest)
        print(json.dumps(receipt, indent=2, ensure_ascii=False))
        if args.execute and not receipt["completed"]:
            raise SystemExit(1)
    except (WorkbenchError, UnicodeError, json.JSONDecodeError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
