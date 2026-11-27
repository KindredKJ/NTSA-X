import random
import unittest

from ntsa_x.affine_ciu import AffineTransition
from ntsa_x.xor_ciu import XorTransition


def sequential(width, state, program):
    limit = (1 << width) - 1
    for op, arg in program:
        if op == 'xor':
            state ^= arg
        elif op == 'rotl':
            state = ((state << arg) | (state >> (width - arg))) & limit
        elif op == 'xorshl':
            state ^= (state << arg) & limit
        elif op == 'xorshr':
            state ^= state >> arg
        else:
            raise ValueError(op)
    return state


class AffineCIUTests(unittest.TestCase):
    def test_all_eight_bit_inputs(self):
        program = [('xor', 135), ('rotl', 3), ('xorshl', 2), ('xorshr', 5)]
        rule = AffineTransition.compile(8, program)
        for s in range(256):
            self.assertEqual(rule.apply(s), sequential(8, s, program))

    def test_recursive_composition(self):
        a = [('xor', 123), ('xorshl', 3)]
        b = [('rotl', 2), ('xorshr', 4)]
        c = [('xor', 37), ('rotl', 1)]
        x, y, z = (AffineTransition.compile(8, p) for p in (a, b, c))
        self.assertEqual(x.then(y).then(z), x.then(y.then(z)))
        self.assertEqual(x.then(y).then(z), AffineTransition.compile(8, a+b+c))

    def test_fixed_xor_compatibility(self):
        base = XorTransition.compose(8, [12, 99, 32])
        affine = AffineTransition.from_xor(base)
        self.assertEqual(affine.count, base.count)
        for s in range(256):
            self.assertEqual(affine.apply(s), base.apply(s))

    def test_unseen_inputs(self):
        rng = random.Random(92)
        program = [('xor', rng.getrandbits(64)), ('rotl', 11),
                   ('xorshl', 7), ('xorshr', 19)] * 128
        rule = AffineTransition.compile(64, program)
        for _ in range(512):
            s = rng.getrandbits(64)
            self.assertEqual(rule.apply(s), sequential(64, s, program))

    def test_reject_outside_algebra(self):
        for program in [[('and_state', 1)], [('add', 1)], [('xor', 256)],
                        [('rotl', 8)], [('xorshr', 0)]]:
            with self.assertRaises(ValueError):
                AffineTransition.compile(8, program)
        with self.assertRaises(ValueError):
            AffineTransition.identity(257)
        with self.assertRaises(ValueError):
            AffineTransition.identity(8).then(AffineTransition.identity(16))
        with self.assertRaises(ValueError):
            AffineTransition.identity(8).apply(-1)
