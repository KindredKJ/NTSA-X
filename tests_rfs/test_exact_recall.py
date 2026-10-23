"""RFS behavior, isolation, corruption, invalidation, and replay tests."""
from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from rfs import RecallFirstEngine, VerificationError, WorkSpec


def spec(**changes):
    config = {
        "namespace": "kindred.rfs",
        "tenant": "tenant-A",
        "operation": "sum-two",
        "arguments": {"a": 8, "b": 13},
        "context": {"currency": "none", "snapshot": "s1"},
        "implementation_revision": "compute-v1",
        "verifier_revision": "verify-v1",
        "policy_revision": "policy-v1",
        "dependencies": {"data": "sha256:test"},
        "ttl_seconds": 40,
        "pure": True,
    }
    config.update(changes)
    return WorkSpec(**config)


def worker(args):
    return {"value": args["a"] + args["b"]}


def verify(args, value, context):
    return context["snapshot"] in ("s1", "s2") and value == worker(args)


class RecallFirstTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "candidate.rfs.sqlite3"
        self.now = [1000.0]
        self.cache = RecallFirstEngine(self.path, clock=lambda: self.now[0])
        self.addCleanup(self.cache.close)

    def test_miss_then_hit_skips_worker(self):
        calls = []
        def counted(args):
            calls.append(1)
            return worker(args)
        first = self.cache.execute(spec(), counted, verify)
        second = self.cache.execute(spec(), counted, verify)
        self.assertEqual((first.source, second.source), ("computed", "recalled"))
        self.assertEqual(first.value, second.value)
        self.assertEqual(len(calls), 1)
        self.assertEqual(self.cache.metrics()["verifier_invocations"], 2)

    def test_state_isolation(self):
        self.cache.execute(spec(), worker, verify)
        for changed in [
            {"tenant": "tenant-B"},
            {"namespace": "other"},
            {"context": {"snapshot": "s2"}},
            {"implementation_revision": "compute-v2"},
            {"policy_revision": "policy-v2"},
            {"verifier_revision": "verify-v2"},
            {"dependencies": {"data": "sha256:updated"}},
            {"arguments": {"a": 8, "b": 14}},
        ]:
            candidate = self.cache.execute(spec(**changed), worker, verify)
            self.assertEqual(candidate.source, "computed", changed)

    def test_ttl_invalidates(self):
        self.cache.execute(spec(), worker, verify)
        self.now[0] = 1040.0
        response = self.cache.execute(spec(), worker, verify)
        self.assertEqual(response.source, "computed")
        self.assertEqual(self.cache.metrics()["invalidations"], 1)

    def test_verifier_rechecked_after_cache_hit(self):
        self.cache.execute(spec(), worker, verify)
        calls = []
        def changed_truth(args, value, context):
            calls.append(True)
            return len(calls) == 2 and verify(args, value, context)
        # On hit verifier returns false. Fresh recomputation re-verifies true.
        response = self.cache.execute(spec(), worker, changed_truth)
        self.assertEqual(response.source, "computed")
        self.assertEqual(len(calls), 2)

    def test_corrupt_row_recalculated(self):
        old = self.cache.execute(spec(), worker, verify)
        db = sqlite3.connect(self.path)
        try:
            with db:
                db.execute("UPDATE rfs_exact_results SET result_json=? WHERE work_key=?",
                           ('{"value":999}', old.key))
        finally:
            db.close()
        recovered = self.cache.execute(spec(), worker, verify)
        self.assertEqual(recovered.source, "computed")
        self.assertEqual(recovered.value, {"value": 21})

    def test_fresh_failure_not_cached(self):
        with self.assertRaises(VerificationError):
            self.cache.execute(spec(), worker, lambda args, value, ctx: False)
        self.assertEqual(self.cache.execute(spec(), worker, verify).source, "computed")

    def test_worker_failure_not_cached(self):
        def failing(args):
            raise RuntimeError("compute error")
        with self.assertRaisesRegex(RuntimeError, "compute error"):
            self.cache.execute(spec(), failing, verify)
        self.assertEqual(self.cache.execute(spec(), worker, verify).source, "computed")

    def test_restart_preserves_valid_entry(self):
        first = self.cache.execute(spec(), worker, verify)
        with RecallFirstEngine(self.path, clock=lambda: self.now[0]) as reopened:
            value = reopened.execute(spec(), lambda _: self.fail("worker called"), verify)
        self.assertEqual(value.source, "recalled")
        self.assertEqual(value.key, first.key)

    def test_non_pure_and_bad_paths_blocked(self):
        with self.assertRaises(ValueError):
            spec(pure=False)
        with self.assertRaises(ValueError):
            RecallFirstEngine(Path(self.tmp.name) / "ntsa.sqlite3")

    def test_malformed_payload_rejected(self):
        with self.assertRaises(ValueError):
            spec(arguments={"x": float("nan")})
        with self.assertRaises(TypeError):
            spec(dependencies={"x": 12})

    def test_mutating_worker_cannot_mutate_caller_input(self):
        arguments = {"a": 8, "b": 13}
        request = spec(arguments=arguments)
        def mutation(args):
            result = worker(args)
            args["a"] = 999
            return result
        self.cache.execute(request, mutation, verify)
        self.assertEqual(arguments["a"], 8)
        self.assertEqual(self.cache.execute(request, worker, verify).source, "recalled")

    def test_no_cross_namespace_storage_write(self):
        other = Path(self.tmp.name) / "ntsa_original.sqlite3"
        other.write_bytes(b"untouched")
        self.cache.execute(spec(), worker, verify)
        self.assertEqual(other.read_bytes(), b"untouched")

    def test_returned_value_is_not_mutable_cache_state(self):
        self.cache.execute(spec(), worker, verify)
        recalled = self.cache.execute(spec(), worker, verify)
        recalled.value["value"] = 2000
        again = self.cache.execute(spec(), worker, verify)
        self.assertEqual(again.value, {"value": 21})


if __name__ == "__main__":
    unittest.main()
