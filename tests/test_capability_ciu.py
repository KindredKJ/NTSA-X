import tempfile
import unittest
from pathlib import Path
from ntsa_x.capability_ciu import BinaryTask, CapabilityRuntime
from ntsa_x.convergence import Check, ConvergenceEngine, REQUIRED_CHECKS
from ntsa_x.nexus import NexusStore


def report():
    return ConvergenceEngine().evaluate([Check(c, True, 'synthetic fixture', external=c == 'evidence') for c in REQUIRED_CHECKS])


def rule(s):
    return s ^ (((s & 7) == 7) << 7)


class CapabilityTests(unittest.TestCase):
    def test_acquire_and_recall_after_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'nexus.db'
            task = BinaryTask('cubic', 'v1', 8, rule, report())
            with NexusStore(path) as store:
                result = CapabilityRuntime(store).execute(task, 7, acquire=True)
                self.assertEqual((result.source, result.value, result.oracle_calls), ('acquired', 135, 512))
            with NexusStore(path) as store:
                runtime = CapabilityRuntime(store)
                for state in range(256):
                    result = runtime.execute(task, state)
                    self.assertEqual((result.value, result.source, result.oracle_calls), (rule(state), 'recalled', 0))
                result = runtime.execute(BinaryTask('cubic', 'v2', 8, lambda s: s), 7)
                self.assertEqual((result.value, result.source, result.oracle_calls), (7, 'original', 1))

    def test_fallback_without_evidence(self):
        with NexusStore(':memory:') as store:
            task = BinaryTask('cubic', 'v1', 8, rule)
            result = CapabilityRuntime(store).execute(task, 7, acquire=True)
            self.assertEqual((result.value, result.source, result.oracle_calls), (135, 'original', 1))
            self.assertEqual(store.stats()['nodes'], 0)

    def test_high_degree_fallback(self):
        with NexusStore(':memory:') as store:
            task = BinaryTask('degree7', 'v1', 7, lambda s: int(s == 127), report())
            result = CapabilityRuntime(store).execute(task, 127, acquire=True)
            self.assertEqual((result.value, result.source), (1, 'original'))
            self.assertEqual(store.stats()['nodes'], 0)

    def test_corrupted_receipt_fallback(self):
        with NexusStore(':memory:') as store:
            task = BinaryTask('cubic', 'v1', 8, rule, report())
            runtime = CapabilityRuntime(store)
            result = runtime.execute(task, 7, acquire=True)
            store._db.execute('UPDATE nodes SET payload=? WHERE digest=?', ('{}', result.receipt))
            result = runtime.execute(task, 7)
            self.assertEqual((result.value, result.source, result.oracle_calls), (135, 'invalid_receipt_fallback', 1))

    def test_input_bounds(self):
        with NexusStore(':memory:') as store:
            runtime = CapabilityRuntime(store)
            with self.assertRaises(ValueError):
                runtime.execute(BinaryTask('cubic', 'v1', 8, rule), 256)
