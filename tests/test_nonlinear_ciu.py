import random
import unittest

from ntsa_x.nonlinear_ciu import BooleanTransition, ExpansionLimit


def sequential(state, gates):
    for op, target, *controls in gates:
        delta = 1 if op == 'not' else (state >> controls[0]) & 1
        if op == 'and_xor':
            delta &= (state >> controls[1]) & 1
        state ^= delta << target
    return state


class NonlinearCIUTests(unittest.TestCase):
    def test_nonlinear_and_exact_all_inputs(self):
        gates = [('and_xor', 2, 0, 1), ('and_xor', 3, 2, 1), ('not', 1), ('xor', 0, 3)]
        rule = BooleanTransition.compile(8, gates)
        self.assertGreaterEqual(rule.degree, 2)
        for s in range(256):
            self.assertEqual(rule.apply(s), sequential(s, gates))

    def test_compose_noncommuting_rules(self):
        a = [('and_xor', 2, 0, 1), ('not', 0)]
        b = [('and_xor', 3, 2, 0), ('xor', 1, 3)]
        x, y = BooleanTransition.compile(4, a), BooleanTransition.compile(4, b)
        self.assertEqual(x.then(y), BooleanTransition.compile(4, a+b))
        self.assertNotEqual(x.then(y), y.then(x))
        for s in range(16):
            self.assertEqual(x.then(y).apply(s), sequential(s, a+b))

    def test_cubic_growth(self):
        rule = BooleanTransition.compile(5, [('and_xor', 3, 0, 1), ('and_xor', 4, 3, 2)])
        self.assertEqual(rule.degree, 3)

    def test_budgets(self):
        with self.assertRaises(ExpansionLimit):
            BooleanTransition.compile(4, [('and_xor', 3, 0, 1)], max_terms=4)
        with self.assertRaises(ExpansionLimit):
            BooleanTransition.compile(4, [('and_xor', 3, 0, 1)], max_work=1)
        with self.assertRaises(ExpansionLimit):
            BooleanTransition.compile(4, []).then(BooleanTransition.compile(4, []), max_terms=3)

    def test_reject_invalid(self):
        for gate in [('and_xor', 0, 1), ('xor', 0, 8), ('not', True), ('add', 0)]:
            with self.assertRaises(ValueError):
                BooleanTransition.compile(8, [gate])
        with self.assertRaises(ValueError):
            BooleanTransition.compile(8, []).apply(256)

    def test_unseen(self):
        rng = random.Random(93)
        gates = [('and_xor', rng.randrange(8,16), rng.randrange(8), rng.randrange(8)) for _ in range(512)]
        rule = BooleanTransition.compile(16, gates)
        for s in rng.sample(range(65536), 512):
            self.assertEqual(rule.apply(s), sequential(s, gates))
