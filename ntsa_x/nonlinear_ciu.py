"""Bounded exact Boolean polynomial compilation, not a learned neural model."""
from dataclasses import dataclass
from typing import Iterable

from .xor_ciu import _bits


class ExpansionLimit(ValueError):
    """Symbolic expansion exceeded its configured resource budget."""


def _width(width):
    if type(width) is not int or not 1 <= width <= 64:
        raise ValueError("Boolean width must be 1..64")
    return width


class _Budget:
    def __init__(self, terms, work):
        if type(terms) is not int or not 1 <= terms <= 65536:
            raise ValueError("invalid term limit")
        if type(work) is not int or not 1 <= work <= 100_000_000:
            raise ValueError("invalid work limit")
        self.terms, self.work = terms, work

    def spend(self, amount):
        self.work -= amount
        if self.work < 0:
            raise ExpansionLimit("symbolic work limit exceeded")

    def check(self, amount):
        if amount > self.terms:
            raise ExpansionLimit("symbolic term limit exceeded")

    def multiply(self, a, b):
        self.spend(len(a) * len(b))
        result = set()
        for x in a:
            for y in b:
                m = x | y  # x_i squared = x_i for Boolean inputs.
                if m in result:
                    result.remove(m)
                else:
                    result.add(m)
                    self.check(len(result))
        return result


@dataclass(frozen=True)
class BooleanTransition:
    """Canonical ANF: (input monomial bitmask, output XOR bitmask) pairs."""
    width: int
    terms: tuple[tuple[int, int], ...]
    count: int = 0

    def __post_init__(self):
        _width(self.width)
        if type(self.terms) is not tuple or len(self.terms) > 65536:
            raise ValueError("invalid immutable term list")
        last = -1
        for pair in self.terms:
            if type(pair) is not tuple or len(pair) != 2:
                raise ValueError("invalid term pair")
            m, output = pair
            _bits(m, self.width)
            _bits(output, self.width)
            if m <= last or not output:
                raise ValueError("terms must be sorted, unique and nonzero")
            last = m
        if type(self.count) is not int or self.count < 0:
            raise ValueError("invalid operation count")

    @property
    def degree(self):
        return max((m.bit_count() for m, _ in self.terms), default=0)

    def apply(self, state):
        _bits(state, self.width)
        result = 0
        for m, output in self.terms:
            if state & m == m:
                result ^= output
        return result

    @classmethod
    def _pack(cls, width, polys, count, budget):
        result = {}
        for i, poly in enumerate(polys):
            for m in poly:
                result[m] = result.get(m, 0) | (1 << i)
                budget.check(len(result))
        return cls(width, tuple(sorted(result.items())), count)

    @classmethod
    def compile(cls, width, gates: Iterable[tuple], *,
                max_terms=4096, max_work=2_000_000):
        """Sequential NOT, XOR and controlled AND-XOR bit instructions."""
        width = _width(width)
        budget = _Budget(max_terms, max_work)
        budget.check(width)
        polys = [{1 << i} for i in range(width)]
        count = 0
        for gate in gates:
            budget.spend(1)
            if not isinstance(gate, tuple) or not gate:
                raise ValueError("invalid gate")
            op, *args = gate
            arity = {"not": 1, "xor": 2, "and_xor": 3}.get(op) if isinstance(op, str) else None
            if arity is None or len(args) != arity or any(
                type(i) is not int or not 0 <= i < width for i in args
            ):
                raise ValueError("invalid or unsupported gate")
            target = args[0]
            if op == "not":
                delta = {0}
            elif op == "xor":
                delta = polys[args[1]]
            else:
                delta = budget.multiply(polys[args[1]], polys[args[2]])
            budget.spend(len(polys[target]) + len(delta))
            polys[target] = polys[target] ^ delta
            budget.check(sum(map(len, polys)))
            count += 1
        return cls._pack(width, polys, count, budget)

    def then(self, other, *, max_terms=4096, max_work=2_000_000):
        """Compose already compiled rules by bounded polynomial substitution."""
        if not isinstance(other, BooleanTransition) or other.width != self.width:
            raise ValueError("incompatible Boolean rules")
        budget = _Budget(max_terms, max_work)
        inputs = [set() for _ in range(self.width)]
        outputs = [set() for _ in range(self.width)]
        for m, mask in self.terms:
            for i in range(self.width):
                if mask & (1 << i):
                    inputs[i].add(m)
        budget.check(sum(map(len, inputs)))
        for monomial, mask in other.terms:
            product = {0}
            while monomial and product:
                bit = monomial & -monomial
                product = budget.multiply(product, inputs[bit.bit_length() - 1])
                monomial ^= bit
            for i in range(self.width):
                if mask & (1 << i):
                    budget.spend(1 + len(outputs[i]) + len(product))
                    outputs[i] ^= product
                    budget.check(sum(map(len, outputs)))
        return self._pack(self.width, outputs, self.count + other.count, budget)
