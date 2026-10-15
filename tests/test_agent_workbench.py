"""Agent workbench regression: no autonomous effects, no outside paths."""
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch
import json

from ntsa_x.agent_workbench import (
    CapabilityFlags, Workbench, WorkbenchError, validate_plan)
from ntsa_x.agent_web import WebAccessError, public_https_url, fetch_text


def plan(*steps):
    return {"version": 1, "steps": [
        {"tool": tool, "args": args} for tool, args in steps]}


class WorkbenchTests(TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "readme.md").write_text(
            "The native agent tests a local workspace.\n", encoding="utf-8")
        (self.root / "tests").mkdir()
        (self.root / "tests" / "test_ok.py").write_text(
            "import unittest\nclass Tests(unittest.TestCase):\n"
            "    def test_good(self): self.assertEqual(3, 1 + 2)\n",
            encoding="utf-8")
        (self.root / ".env").write_text("SECRET=do-not-display")
        (self.root / "folder").mkdir()
        self.wb = Workbench(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_review_does_not_execute_any_actions(self):
        p = plan(("workspace.write", {"path": "folder/test.py",
                                     "content": "print('hello')"}))
        report = self.wb.run(p)
        self.assertEqual(report["mode"], "review")
        self.assertEqual(len(report["plan_sha256"]), 64)
        self.assertFalse((self.root / "folder" / "test.py").exists())
        with self.assertRaisesRegex(WorkbenchError, "digest approval"):
            self.wb.run(p, execute=True)
        with self.assertRaisesRegex(WorkbenchError, "disabled"):
            self.wb.run(p, execute=True,
                        approve_digest=report["plan_sha256"])

    def test_explicit_approval_only_for_exact_plan_and_capability(self):
        wb = Workbench(self.root, flags=CapabilityFlags(write=True))
        original = plan(("workspace.write", {
            "path": "folder/hello.py", "content": "print(1)"}))
        hash_value = wb.digest(original)
        changed = plan(("workspace.write", {
            "path": "folder/hello.py", "content": "print(2)"}))
        with self.assertRaisesRegex(WorkbenchError, "digest approval"):
            wb.run(changed, execute=True, approve_digest=hash_value)
        result = wb.run(original, execute=True, approve_digest=hash_value)
        self.assertTrue(result["completed"])
        self.assertEqual((self.root / "folder/hello.py").read_text(),
                         "print(1)")
        duplicate = wb.run(
            original, execute=True, approve_digest=hash_value)
        self.assertFalse(duplicate["completed"])
        self.assertIn("SHA-256", duplicate["results"][0]["message"])
        new = plan(("workspace.write", {
            "path": "folder/hello.py", "content": "print(3)",
            "expected_sha256": sha256(b"print(1)").hexdigest()}))
        result = wb.run(new, execute=True, approve_digest=wb.digest(new))
        self.assertTrue(result["completed"])
        self.assertEqual((self.root / "folder/hello.py").read_text(),
                         "print(3)")

    def test_paths_and_secrets_denied_even_with_symlinks(self):
        for file in ("../etc/passwd", ".env", ".git/config",
                     ".local/model.json", "folder/../../out.py",
                     "C:\\Users\\Public\\file.txt", ".ssh/id_rsa"):
            with self.assertRaises(WorkbenchError, msg=file):
                self.wb.path(file)
        try:
            (self.root / "folder" / "escape").symlink_to(
                self.root.parent, target_is_directory=True)
        except (OSError, NotImplementedError):
            return
        with self.assertRaises(WorkbenchError):
            self.wb.path("folder/escape/out.txt", new_file=True)

    def test_local_read_search_inventory(self):
        p = plan(
            ("workspace.list", {"path": "."}),
            ("workspace.read", {"path": "readme.md"}),
            ("workspace.search", {"pattern": "native agent", "limit": 10}),
            ("devops.inventory", {}))
        receipt = self.wb.run(p, execute=True)
        self.assertTrue(receipt["completed"])
        listed = receipt["results"][0]["result"]["items"]
        self.assertNotIn(".env", [x["name"] for x in listed])
        self.assertIn("native agent", receipt["results"][1]["result"]["text"])
        hits = receipt["results"][2]["result"]["matches"]
        self.assertEqual(hits[0]["path"], "readme.md")
        self.assertNotIn("SECRET", json.dumps(receipt))

    def test_fixed_unittest_command_requires_approval(self):
        p = plan(("python.tests", {"pattern": "test_ok.py",
                                   "timeout_seconds": 20}))
        with self.assertRaises(WorkbenchError):
            self.wb.run(p, execute=True)
        wb = Workbench(self.root, flags=CapabilityFlags(tests=True))
        receipt = wb.run(p, execute=True, approve_digest=wb.digest(p))
        self.assertTrue(receipt["completed"])
        self.assertEqual(receipt["results"][0]["result"]["exit_code"], 0)
        self.assertIn("OK", receipt["results"][0]["result"]["stderr"])

    def test_strict_protocol_rejects_fabricated_tools_and_arguments(self):
        bad = (
            {"version": 1, "steps": []},
            plan(("shell.exec", {"command": "whoami"})),
            plan(("workspace.write", {"path": "a.py", "content": "x",
                                       "command": "rm -rf /"})),
            plan(("python.tests", {"pattern": "../test.py"})),
            plan(("workspace.list", {"limit": True})),
            plan(("workspace.read", {"path": "readme.md", "nested": {}})),
        )
        for item in bad:
            with self.assertRaises(WorkbenchError):
                validate_plan(item)

    def test_web_allowlist_and_gate_no_network_without_approval(self):
        forbidden = (
            "http://example.org/", "https://localhost/",
            "https://127.0.0.1/", "https://example.org:444/",
            "https://example.org.evil.test/", "https://example.org@evil.test/",
            "https://example.org/path#fragment")
        for url in forbidden:
            with self.assertRaises(WebAccessError, msg=url):
                public_https_url(url, ("example.org",))
        self.assertEqual(public_https_url(
            "https://example.org/docs?q=1", ("example.org",)),
            "https://example.org/docs?q=1")
        p = plan(("web.fetch", {"url": "https://example.org/docs"}))
        wb = Workbench(self.root, allowed_hosts=("example.org",),
                       flags=CapabilityFlags(network=True))
        with self.assertRaisesRegex(WorkbenchError, "digest"):
            wb.run(p, execute=True)
        with patch("ntsa_x.agent_workbench.fetch_text",
                   return_value={"text": "public read result"}) as mocked:
            receipt = wb.run(p, execute=True, approve_digest=wb.digest(p))
            self.assertTrue(receipt["completed"])
            mocked.assert_called_once()

    def test_conservative_english_intents_generate_plans_only(self):
        from ntsa_x.agent_intent import propose
        self.assertEqual(propose("git status")["steps"][0]["tool"],
                         "git.status")
        self.assertEqual(propose("run tests")["steps"][0]["tool"],
                         "python.tests")
        self.assertEqual(propose("search files for model")["steps"][0]["args"],
                         {"pattern": "model"})
        self.assertEqual(propose("read file readme.md")["steps"][0]["args"],
                         {"path": "readme.md"})
        for request in ("delete everything", "deploy prod",
                        "send my credentials", "run shell rm -rf ."):
            with self.assertRaises(WorkbenchError):
                propose(request)

    def test_denied_actions_prevalidate_before_prior_effect(self):
        wb = Workbench(self.root, flags=CapabilityFlags(write=True))
        p = plan(
            ("workspace.write", {"path": "folder/hello.txt",
                                 "content": "should-not-appear"}),
            ("workspace.read", {"path": "../outside.txt"}))
        with self.assertRaises(WorkbenchError):
            wb.run(p, execute=True, approve_digest="0" * 64)
        self.assertFalse((self.root / "folder/hello.txt").exists())


if __name__ == "__main__":
    import unittest
    unittest.main()
