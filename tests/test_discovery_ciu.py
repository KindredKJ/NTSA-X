import random
import unittest

from ntsa_x.discovery_ciu import audit, discover


class DiscoveryTests(unittest.TestCase):
    def test_quadratic_unseen(self):
        def oracle(s):
            return s ^ (((s & 3) == 3) << 7) ^ 16
        result = discover(8, oracle)
        self.assertEqual(result.probes, 37)
        self.assertFalse(result.exhaustive)
        self.assertEqual(audit(result, oracle, range(256)), (256, 0))

    def test_exhaustive_arbitrary(self):
        rng = random.Random(8)
        table = [rng.randrange(32) for _ in range(32)]
        result = discover(5, table.__getitem__, degree=5)
        self.assertTrue(result.exhaustive)
        self.assertEqual(audit(result, table.__getitem__, range(32)), (32, 0))

    def test_hidden_cubic_rejected(self):
        def oracle(s):
            return s ^ (((s & 7) == 7) << 7)
        result = discover(8, oracle, degree=2)
        self.assertEqual(result.checked_apply(7, oracle), (135, False))
        self.assertEqual(audit(result, oracle, range(256)), (256, 32))
        self.assertEqual(audit(discover(8, oracle, degree=3), oracle, range(256)), (256, 0))

    def test_budget_before_query(self):
        def oracle(s):
            self.fail('budget should be checked before oracle invocation')
        with self.assertRaises(ValueError):
            discover(32, oracle, max_probes=528)

    def test_invalid_output(self):
        with self.assertRaises(ValueError):
            discover(8, lambda s: 256)
