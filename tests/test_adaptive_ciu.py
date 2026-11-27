import tempfile
import unittest
from pathlib import Path

from ntsa_x.adaptive_ciu import learn, NexusRuleLibrary
from ntsa_x.convergence import Check, ConvergenceEngine, REQUIRED_CHECKS
from ntsa_x.nexus import NexusStore


def oracle(s):
    return s ^ (((s & 7) == 7) << 7)


def report():
    return ConvergenceEngine().evaluate([
        Check(c, True, "synthetic test fixture", external=c == "evidence")
        for c in REQUIRED_CHECKS])


class AdaptiveTests(unittest.TestCase):
    def test_escalation_and_probe_reuse(self):
        seen = set()
        def counted(s):
            self.assertNotIn(s, seen)
            seen.add(s)
            return oracle(s)
        result = learn(8, counted, range(256), max_observations=256)
        self.assertEqual(result.degree, 3)
        self.assertTrue(result.exhaustive)
        self.assertEqual(result.observations, 256)
        self.assertEqual(len(seen), 256)
        self.assertTrue(all(result.rule.apply(s) == oracle(s) for s in range(256)))

    def test_partial_candidate_not_proof(self):
        result = learn(8, oracle, [7, 15], max_degree=3)
        self.assertEqual(result.degree, 3)
        self.assertFalse(result.exhaustive)

    def test_abstain(self):
        self.assertIsNone(learn(8, oracle, [7], max_degree=2).rule)
        self.assertIsNone(learn(8, oracle, [7], max_observations=1).rule)

    def test_persistence_and_revision(self):
        rule = learn(8, oracle, range(256), max_observations=256).rule
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'nexus.db'
            with NexusStore(path) as store:
                library = NexusRuleLibrary(store)
                receipt = library.register(rule, context_id='c', revision='1', oracle=oracle, report=report())
                with self.assertRaises(ValueError):
                    library.register(rule, context_id='c', revision='2', oracle=lambda s: s, report=report())
            with NexusStore(path) as store:
                library = NexusRuleLibrary(store)
                self.assertEqual(library.recall(receipt, context_id='c', revision='1'), rule)
                self.assertIsNone(library.recall(receipt, context_id='c', revision='2'))
                self.assertIsNone(library.recall(receipt, context_id='d', revision='1'))
