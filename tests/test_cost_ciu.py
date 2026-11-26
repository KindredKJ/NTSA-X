import unittest
from ntsa_x.cost_ciu import Cost, select
from ntsa_x.capability_ciu import BinaryTask, CapabilityRuntime
from ntsa_x.convergence import Check, ConvergenceEngine, REQUIRED_CHECKS
from ntsa_x.nexus import NexusStore


def report():
    return ConvergenceEngine().evaluate([
        Check(c, True, 'test fixture', external=c == 'evidence') for c in REQUIRED_CHECKS])


def oracle(s):
    return s ^ (((s & 7) == 7) << 7)


class CostTests(unittest.TestCase):
    def test_cost_horizon(self):
        direct = Cost('original', 0, 100)
        rule = Cost('rule', 1000, 10)
        self.assertLess(direct.total(1), rule.total(1))
        self.assertLess(rule.total(100), direct.total(100))

    def test_no_rule_original(self):
        with NexusStore(':memory:') as store:
            runtime = CapabilityRuntime(store)
            plan = select(runtime, BinaryTask('cubic', 'v1', 8, oracle), expected_calls=100)
            self.assertEqual(plan.method, 'original')
            self.assertEqual(plan.apply(7), 135)
            self.assertEqual(store.stats()['nodes'], 0)

    def test_verified_choices_and_memory(self):
        with NexusStore(':memory:') as store:
            runtime = CapabilityRuntime(store)
            task = BinaryTask('cubic', 'v1', 8, oracle, report())
            runtime.execute(task, 7, acquire=True)
            for budget, names in [(0, {'original', 'rule'}), (2048, {'original', 'rule', 'table'})]:
                plan = select(runtime, task, expected_calls=100000, table_budget_bytes=budget)
                self.assertEqual({c.method for c in plan.costs}, names)
                self.assertTrue(all(c.extra_table_bytes <= budget for c in plan.costs))
                self.assertEqual([plan.apply(s) for s in range(256)], [oracle(s) for s in range(256)])
                with self.assertRaises(ValueError):
                    plan.apply(256)

    def test_mismatched_revision_and_corrupt_receipt(self):
        with NexusStore(':memory:') as store:
            runtime = CapabilityRuntime(store)
            task = BinaryTask('cubic', 'v1', 8, oracle, report())
            receipt = runtime.execute(task, 7, acquire=True).receipt
            other = BinaryTask('cubic', 'v2', 8, lambda s: s)
            self.assertEqual(select(runtime, other, expected_calls=100).apply(7), 7)
            store._db.execute('UPDATE nodes SET payload=? WHERE digest=?', ('{}', receipt))
            self.assertEqual(select(runtime, task, expected_calls=100).method, 'original')

    def test_budget(self):
        with NexusStore(':memory:') as store:
            with self.assertRaises(ValueError):
                select(CapabilityRuntime(store), BinaryTask('cubic', 'v1', 8, oracle), expected_calls=0)
