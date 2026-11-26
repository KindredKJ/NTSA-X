"""Active Boolean rule discovery with explicit evidence scope."""
from dataclasses import dataclass
from itertools import combinations
from math import comb
from typing import Callable

from .nonlinear_ciu import BooleanTransition, _width
from .xor_ciu import _bits


@dataclass(frozen=True)
class DiscoveryResult:
    rule: BooleanTransition
    probes: int
    exhaustive: bool
    degree_bound: int

    def checked_apply(self, state: int, oracle: Callable[[int], int]) -> tuple[int, bool]:
        """Return authoritative output and candidate agreement, not a speedup."""
        predicted = self.rule.apply(state)
        actual = _bits(oracle(state), self.rule.width)
        return actual, predicted == actual


def discover(width: int, oracle: Callable[[int], int], *, degree: int = 2,
             max_probes: int = 4096) -> DiscoveryResult:
    """Interpolate a candidate ANF using low-weight inputs.

    Exact on the entire domain only when exhaustive, or under an independently
    established degree bound. The oracle must be deterministic and side-effect free.
    """
    width = _width(width)
    if not callable(oracle) or type(degree) is not int or not 0 <= degree <= min(width, 6):
        raise ValueError("invalid oracle or degree")
    if type(max_probes) is not int or not 1 <= max_probes <= 65536:
        raise ValueError("invalid probe budget")
    required = sum(comb(width, k) for k in range(degree + 1))
    if required > max_probes:
        raise ValueError("probe budget exceeded")
    coefficients = {}
    for k in range(degree + 1):
        for indices in combinations(range(width), k):
            mask = sum(1 << i for i in indices)
            value = _bits(oracle(mask), width)
            sub = mask
            while sub:
                sub = (sub - 1) & mask
                value ^= coefficients.get(sub, 0)
            if value:
                coefficients[mask] = value
    rule = BooleanTransition(width, tuple(sorted(coefficients.items())))
    return DiscoveryResult(rule, required, required == (1 << width), degree)


def audit(candidate: DiscoveryResult, oracle: Callable[[int], int], states) -> tuple[int, int]:
    """Return (checked, mismatched). Passing finite samples is not proof."""
    checked = mismatched = 0
    for state in states:
        _, match = candidate.checked_apply(state, oracle)
        checked += 1
        mismatched += not match
    return checked, mismatched
