import tempfile
import unittest
from pathlib import Path

from ntsa_x.capability_ciu import BinaryTask, CapabilityRuntime
from ntsa_x.compose_ciu import compose
from ntsa_x.convergence import Check, ConvergenceEngine, REQUIRED_CHECKS
from ntsa_x.nexus import NexusStore


def report():
    return ConvergenceEngine().evaluate([Check(c, True, 'synthetic fixture', external=c == 'evidence') for c in REQUIRED_CHECKS])


def a(s):
    return s ^ (((s & 3) == 3) << 2)


def b(s):
    return s ^ (((s & 4) != 0) << 1)


def setup(store):
    rt = CapabilityRuntime(store)
    tasks = (BinaryTask('alpha', 'v1', 8, a, report()), BinaryTask('beta', 'v1', 8, b, report()))
    receipts = tuple(rt.execute(t, 3, acquire=True).receipt for t in tasks)
    return rt, tasks, receipts


class ComposeTests(unittest.TestCase):
    def test_order_repeat_and_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'nexus.db'
            with NexusStore(path) as store:
                rt, tasks, receipts = setup(store)
                chain = compose(rt, tasks + tasks[:1], report=report())
                reverse = compose(rt, tasks[::-1], report=report())
                node = store.read(chain.receipt)
                self.assertEqual(node.parents, receipts)
                self.assertEqual(node.payload['composition_order'], [*receipts, receipts[0]])
                self.assertNotEqual(chain.task.revision, reverse.task.revision)
                self.assertEqual(chain.oracle_calls, 768)
            with NexusStore(path) as store:
                rt = CapabilityRuntime(store)
                for s in range(256):
                    r = rt.execute(chain.task, s)
                    self.assertEqual((r.value, r.source, r.oracle_calls), (a(b(a(s))), 'recalled', 0))

    def test_recursive_composition_and_invalid_ancestry(self):
        with NexusStore(':memory:') as store:
            rt, tasks, receipts = setup(store)
            first = compose(rt, tasks, report=report())
            with self.assertRaises(ValueError):
                compose(rt, (first.task, tasks[1]), report=report(), max_oracle_calls=512)
            second = compose(rt, (first.task, tasks[1]), report=report())
            self.assertEqual(second.oracle_calls, 768)
            self.assertEqual(store.read(second.receipt).parents, (first.receipt, receipts[1]))
            for s in range(256):
                self.assertEqual(rt.execute(second.task, s).value, b(b(a(s))))
            store._db.execute('UPDATE nodes SET payload=? WHERE digest=?', ('{}', receipts[0]))
            r = rt.execute(second.task, 3)
            self.assertEqual((r.source, r.value), ('invalid_receipt_fallback', a(3)))

    def test_missing_and_budget(self):
        with NexusStore(':memory:') as store:
            rt, tasks, _ = setup(store)
            for args, kwargs in [(tasks, {'max_oracle_calls': 511}), (tasks, {'max_terms': 1}),
                ((tasks[0], BinaryTask('unknown', 'v1', 8, b)), {})]:
                with self.assertRaises(ValueError):
                    compose(rt, args, report=report(), **kwargs)
            self.assertEqual(store.stats()['nodes'], 2)

    def test_changed_oracle_rejected(self):
        with NexusStore(':memory:') as store:
            rt, tasks, _ = setup(store)
            changed = BinaryTask('alpha', 'v1', 8, lambda s: s, report())
            with self.assertRaises(ValueError):
                compose(rt, (changed, tasks[1]), report=report())
            self.assertEqual(store.stats()['nodes'], 2)
