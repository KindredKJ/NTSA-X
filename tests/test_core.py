import unittest
from ntsa_x import MemoryChain, Scheduler

class TestMemoryChain(unittest.TestCase):
    def test_verified_commit_and_parent(self):
        chain = MemoryChain()
        first = chain.append({"state": 1}, verified=True)
        second = chain.append({"state": 2}, parents=(first.digest,), verified=True)
        self.assertIn(first.digest, chain.blocks)
        self.assertEqual(second.parents, (first.digest,))
        self.assertEqual(first.digest, chain.append({"state": 1}, verified=True).digest)

    def test_reject_unverified_and_missing_parent(self):
        chain = MemoryChain()
        with self.assertRaises(ValueError):
            chain.append({"state": 1})
        with self.assertRaises(ValueError):
            chain.append({"state": 1}, parents=("missing",), verified=True)

    def test_gating(self):
        scheduler = Scheduler()
        self.assertFalse(scheduler.gate(verified_recall=True, estimated_cost=1, remaining_budget=10).enabled)
        self.assertFalse(scheduler.gate(verified_recall=False, estimated_cost=11, remaining_budget=10).enabled)
        self.assertTrue(scheduler.gate(verified_recall=False, estimated_cost=2, remaining_budget=10).enabled)

if __name__ == "__main__":
    unittest.main()
