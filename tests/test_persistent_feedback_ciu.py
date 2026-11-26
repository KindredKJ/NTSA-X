import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ntsa_x.capability_ciu import BinaryTask, CapabilityRuntime
from ntsa_x.convergence import Check, ConvergenceEngine, REQUIRED_CHECKS
from ntsa_x.feedback_ciu import FeedbackRuntime
from ntsa_x.nexus import NexusStore


def oracle(s):
    return s ^ (((s & 7) == 7) << 7)


def task():
    report = ConvergenceEngine().evaluate([
        Check(c, True, 'test fixture', external=c == 'evidence') for c in REQUIRED_CHECKS])
    return BinaryTask('cubic', 'v1', 8, oracle, report)


def timed(runtime, duration, states=(7,)):
    with patch('ntsa_x.feedback_ciu.perf_counter_ns', side_effect=[0, 10, 10, 10 + duration]):
        return runtime.execute(states)


class PersistentFeedbackTests(unittest.TestCase):
    def test_restart_and_environment_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'nexus.db'
            with NexusStore(path) as store:
                runtime = CapabilityRuntime(store)
                runtime.execute(task(), 7, acquire=True)
                feedback = FeedbackRuntime(runtime, task(), environment='host-a')
                self.assertEqual(timed(feedback, 100).method, 'original')
                self.assertEqual(timed(feedback, 10).method, 'rule')
            with NexusStore(path) as store:
                runtime = CapabilityRuntime(store)
                feedback = FeedbackRuntime(runtime, task(), environment='host-a')
                result = timed(feedback, 10)
                self.assertEqual((result.method, result.values, result.oracle_calls), ('rule', (135,), 0))
                self.assertEqual(timed(FeedbackRuntime(runtime, task(), environment='host-b'), 100).method, 'original')
                self.assertEqual(timed(feedback, 100, (7, 15)).method, 'original')
                other = BinaryTask('cubic', 'v2', 8, oracle)
                self.assertEqual(timed(FeedbackRuntime(runtime, other, environment='host-a'), 100).method, 'original')

    def test_expired_future_and_invalid_timings(self):
        with NexusStore(':memory:') as store:
            runtime = CapabilityRuntime(store)
            runtime.execute(task(), 7, acquire=True)
            feedback = FeedbackRuntime(runtime, task(), environment='a', max_age_s=1)
            timed(feedback, 100)
            timed(feedback, 10)
            for stamp in (0, 9223372036854775807):
                store._db.execute('UPDATE ciu_timing_profile SET updated_ns=?', (stamp,))
                self.assertEqual(timed(FeedbackRuntime(runtime, task(), environment='a', max_age_s=1), 100).method, 'original')
            store._db.execute('UPDATE ciu_timing_profile SET ns=-1')
            self.assertEqual(timed(FeedbackRuntime(runtime, task(), environment='a'), 100).method, 'original')

    def test_receipt_invalidation_and_storage_failure(self):
        with NexusStore(':memory:') as store:
            runtime = CapabilityRuntime(store)
            receipt = runtime.execute(task(), 7, acquire=True).receipt
            feedback = FeedbackRuntime(runtime, task(), environment='a', table_budget_bytes=2048)
            for duration in (100, 10, 1):
                timed(feedback, duration)
            store._db.execute('UPDATE nodes SET payload=? WHERE digest=?', ('{}', receipt))
            result = timed(feedback, 100)
            self.assertEqual((result.method, result.values), ('original', (135,)))
            store._db.execute('DROP TABLE ciu_timing_profile')
            self.assertEqual(timed(feedback, 100).values, (135,))

    def test_warm_policy_still_explores(self):
        with NexusStore(':memory:') as store:
            runtime = CapabilityRuntime(store)
            runtime.execute(task(), 7, acquire=True)
            feedback = FeedbackRuntime(runtime, task(), environment='a', alpha=1)
            timed(feedback, 100)
            timed(feedback, 10)
            feedback = FeedbackRuntime(runtime, task(), environment='a', alpha=1, explore_every=2)
            self.assertEqual(timed(feedback, 200).method, 'rule')
            self.assertEqual(timed(feedback, 5).method, 'original')
            self.assertTrue(timed(feedback, 5).exploratory)
