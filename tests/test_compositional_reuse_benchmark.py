"""SW12 integration tests using the real Nexus and convergence index."""
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest

from examples.benchmark_compositional_reuse import (
    CTX, SCOPE, REV, fixture, reference, solve, valid, component, key,
    run_case, report,
)
from ntsa_x.nexus import NexusStore
from ntsa_x.reuse import ConvergenceLibrary


class CompositionalReuseTests(unittest.TestCase):
    def test_independent_solvers(self):
        for i in (0, 31, 1023):
            b = fixture(i)
            x, n = solve(b)
            y, m = solve(b, True)
            self.assertEqual(x, reference(b))
            self.assertEqual(x, y)
            self.assertGreater(min(n, m), 0)
            self.assertTrue(valid(x, b))
            x[0] += 1
            self.assertFalse(valid(x, b))
            self.assertFalse(report({"states": [x]}, {"states": [x]}, [b]).converged)

    def test_composition_and_partial_reuse(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            bs = [fixture(i) for i in range(4)]
            cold = run_case(root / "cold.sqlite", bs, 0)
            warm = run_case(root / "warm.sqlite", bs, 3)
            full = run_case(root / "full.sqlite", bs, 4)
            self.assertEqual((cold["hits"], warm["hits"], full["hits"]), (0, 3, 4))
            self.assertGreater(cold["row_updates"], warm["row_updates"])
            self.assertEqual(full["row_updates"], 0)
            self.assertEqual(cold["digest"], warm["digest"])
            self.assertEqual(warm["digest"], full["digest"])
            self.assertEqual(max(cold["max_error"], warm["max_error"], full["max_error"]), 0)

    def test_persistent_reuse_and_corrupted_ancestor(self):
        with TemporaryDirectory() as tmp:
            db = Path(tmp) / "state.sqlite"
            b = fixture(0)
            with NexusStore(db) as store:
                node, updates, hits = component(store, ConvergenceLibrary(store), b)
                self.assertGreater(updates, 0)
                self.assertEqual(hits, 0)
                digest = node.digest
            with NexusStore(db) as store:
                old, updates, hits = component(store, ConvergenceLibrary(store), b)
                self.assertEqual((old.digest, updates, hits), (digest, 0, 1))
                lib = ConvergenceLibrary(store)
                self.assertIsNone(lib.get(context_id=CTX, scope=SCOPE, revision="changed", inputs=key(b)))
            with sqlite3.connect(db) as conn:
                conn.execute("UPDATE nodes SET payload='{}' WHERE digest=?", (digest,))
            with NexusStore(db) as store:
                self.assertFalse(store.verify(digest))
                self.assertIsNone(ConvergenceLibrary(store).get(
                    context_id=CTX, scope=SCOPE, revision=REV, inputs=key(b)))


if __name__ == "__main__":
    unittest.main()
