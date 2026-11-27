import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ntsa_x.batch_ciu import HotChainRuntime
from ntsa_x.capability_ciu import BinaryTask, CapabilityRuntime
from ntsa_x.nexus import NexusStore
from test_compose_ciu import setup, report, a, b


class BatchTests(unittest.TestCase):
    def test_hot_chain_and_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'nexus.db'
            with NexusStore(path) as store:
                rt, tasks, _ = setup(store)
                hot = HotChainRuntime(rt, min_inputs=6)
                for i in range(3):
                    r = hot.execute(tasks, [3, 7], report=report())
                    self.assertEqual(r.values, (b(a(3)), b(a(7))))
                    self.assertEqual(r.source, 'stages' if i < 2 else 'fused')
                self.assertTrue(r.promotion_attempted)
                digest = r.receipt
                self.assertEqual(store.stats()['nodes'], 3)
            with NexusStore(path) as store:
                hot = HotChainRuntime(CapabilityRuntime(store))
                r = hot.execute(tasks, range(256))
                self.assertEqual(r.values, tuple(b(a(s)) for s in range(256)))
                self.assertEqual((r.source, r.receipt, r.promotion_attempted), ('fused', digest, False))

    def test_batch_verification_not_per_input(self):
        with NexusStore(':memory:') as store:
            rt, tasks, receipts = setup(store)
            hot = HotChainRuntime(rt, min_batches=2, min_inputs=2)
            hot.execute(tasks, [3], report=report())
            hot.execute(tasks, [3], report=report())
            with patch.object(store, 'read', wraps=store.read) as read:
                r = hot.execute(tasks, range(256))
                self.assertEqual(r.source, 'fused')
                self.assertEqual(read.call_count, 1)
            store._db.execute('UPDATE nodes SET payload=? WHERE digest=?', ('{}', receipts[0]))
            r = hot.execute(tasks, [3])
            self.assertEqual((r.source, r.values), ('original', (b(a(3)),)))

    def test_no_report_and_budget_fallback(self):
        with NexusStore(':memory:') as store:
            rt, tasks, _ = setup(store)
            hot = HotChainRuntime(rt, min_batches=2, min_inputs=2, max_oracle_calls=1)
            for _ in range(3):
                self.assertEqual(hot.execute(tasks, [3]).source, 'stages')
            r = hot.execute(tasks, [3], report=report())
            self.assertTrue(r.promotion_attempted)
            self.assertEqual(r.source, 'original')
            r = hot.execute(tasks, [3], report=report())
            self.assertFalse(r.promotion_attempted)
            self.assertEqual(r.source, 'original')
            self.assertEqual(store.stats()['nodes'], 2)

    def test_missing_revision_and_input_validation(self):
        with NexusStore(':memory:') as store:
            rt, tasks, _ = setup(store)
            hot = HotChainRuntime(rt)
            changed = (BinaryTask('alpha', 'v2', 8, lambda s: s), tasks[1])
            self.assertEqual(hot.execute(changed, [3]).values, (b(3),))
            for states in ([], [256]):
                with self.assertRaises(ValueError):
                    hot.execute(tasks, states)
