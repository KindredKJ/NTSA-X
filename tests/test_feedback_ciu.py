import unittest
from unittest.mock import patch

from ntsa_x.capability_ciu import BinaryTask, CapabilityRuntime
from ntsa_x.convergence import Check, ConvergenceEngine, REQUIRED_CHECKS
from ntsa_x.feedback_ciu import FeedbackRuntime
from ntsa_x.nexus import NexusStore


def oracle(s):
    return s ^ (((s & 7) == 7) << 7)


def report():
    return ConvergenceEngine().evaluate([
        Check(c, True, 'test fixture', external=c == 'evidence') for c in REQUIRED_CHECKS])


def timed(runtime, duration):
    with patch('ntsa_x.feedback_ciu.perf_counter_ns', side_effect=[0, 10, 10, 10 + duration]):
        return runtime.execute([7])


class FeedbackTests(unittest.TestCase):
    def test_online_adaptation(self):
        with NexusStore(':memory:') as store:
            runtime = CapabilityRuntime(store)
            task = BinaryTask('cubic', 'v1', 8, oracle, report())
            runtime.execute(task, 7, acquire=True)
            feedback = FeedbackRuntime(runtime, task, explore_every=2, alpha=1)
            self.assertEqual(timed(feedback, 100).method, 'original')
            self.assertEqual(timed(feedback, 10).method, 'rule')
            self.assertEqual(timed(feedback, 100).method, 'original')
            self.assertEqual(timed(feedback, 200).method, 'rule')
            self.assertEqual(timed(feedback, 20).method, 'rule')
            self.assertEqual(timed(feedback, 20).method, 'rule')
            self.assertEqual(timed(feedback, 5).method, 'original')
            self.assertEqual(timed(feedback, 5).method, 'original')

    def test_table_and_no_duplicate_oracle_calls(self):
        with NexusStore(':memory:') as store:
            calls = []
            def counted(s):
                calls.append(s)
                return oracle(s)
            task = BinaryTask('cubic', 'v1', 8, counted, report())
            runtime = CapabilityRuntime(store)
            runtime.execute(task, 7, acquire=True)
            calls.clear()
            feedback = FeedbackRuntime(runtime, task, table_budget_bytes=2048)
            states = range(256)
            results = [feedback.execute(states) for _ in range(3)]
            self.assertEqual([r.method for r in results], ['original', 'rule', 'table'])
            self.assertEqual(len(calls), 256)
            self.assertEqual([r.oracle_calls for r in results], [256, 0, 0])
            self.assertTrue(all(r.values == tuple(map(oracle, states)) for r in results))

    def test_invalidated_receipt_discards_table(self):
        with NexusStore(':memory:') as store:
            runtime = CapabilityRuntime(store)
            task = BinaryTask('cubic', 'v1', 8, oracle, report())
            receipt = runtime.execute(task, 7, acquire=True).receipt
            feedback = FeedbackRuntime(runtime, task, table_budget_bytes=2048)
            for _ in range(3):
                feedback.execute([7])
            store._db.execute('UPDATE nodes SET payload=? WHERE digest=?', ('{}', receipt))
            result = feedback.execute([7])
            self.assertEqual((result.method, result.values, result.oracle_calls), ('original', (135,), 1))
            self.assertEqual(set(feedback.estimates), {'original'})

    def test_no_capability_and_input_limits(self):
        with NexusStore(':memory:') as store:
            runtime = CapabilityRuntime(store)
            feedback = FeedbackRuntime(runtime, BinaryTask('cubic', 'v2', 8, oracle))
            self.assertEqual(feedback.execute([7]).method, 'original')
            for states in ([], [256], range(65537)):
                with self.assertRaises(ValueError):
                    feedback.execute(states)
            with self.assertRaises(ValueError):
                FeedbackRuntime(runtime, feedback.task, alpha=float('nan'))
