"""Bounded adaptive discovery and exact, revision-scoped Nexus rule reuse."""
from dataclasses import dataclass
from math import comb
from typing import Callable, Iterable

from .convergence import ConvergenceReport
from .discovery_ciu import discover
from .nexus import NexusStore
from .nonlinear_ciu import BooleanTransition, _width
from .xor_ciu import _bits, _label

KIND = "xor_ciu/exhaustive_rule/v1"


@dataclass(frozen=True)
class AdaptiveResult:
    rule: BooleanTransition | None
    degree: int | None
    observations: int
    mismatches: int
    exhaustive: bool


def learn(width: int, oracle: Callable[[int], int], states: Iterable[int], *,
          max_degree: int = 3, max_observations: int = 4096) -> AdaptiveResult:
    """Escalate degree after counterexamples, reusing all observed outputs.

    Audit states are diagnostic, not independent after model selection. Only
    complete domain coverage can establish exactness here. No side effects.
    """
    width = _width(width)
    if not callable(oracle) or type(max_degree) is not int or not 0 <= max_degree <= min(width, 6):
        raise ValueError("invalid oracle or degree")
    if type(max_observations) is not int or not 1 <= max_observations <= 65536:
        raise ValueError("invalid observation budget")
    observed = {}
    audit_states = set()
    for index, state in enumerate(states):
        if index >= max_observations:
            raise ValueError("audit input budget exceeded")
        audit_states.add(_bits(state, width))
    if not audit_states:
        raise ValueError("nonempty audit required")

    def query(state):
        if state not in observed:
            observed[state] = _bits(oracle(state), width)
        return observed[state]

    mismatches = 0
    for degree in range(max_degree + 1):
        required = sum(comb(width, k) for k in range(degree + 1))
        # Conservative bound: account for overlap without materializing probes.
        if required + len(audit_states) > max_observations + min(len(audit_states), required):
            break
        # Low-weight probe set is exactly the set used by discover().
        overlap = sum(s.bit_count() <= degree for s in audit_states)
        if required + len(audit_states) - overlap > max_observations:
            break
        candidate = discover(width, query, degree=degree, max_probes=max_observations)
        for state in sorted(audit_states):
            query(state)
        mismatches = sum(candidate.rule.apply(s) != y for s, y in observed.items())
        if not mismatches:
            return AdaptiveResult(candidate.rule, degree, len(observed), 0,
                                  len(observed) == 1 << width)
    return AdaptiveResult(None, None, len(observed), mismatches, False)


class NexusRuleLibrary:
    """Store only exhaustively checked rules using the existing verified ledger."""
    def __init__(self, store: NexusStore):
        if not isinstance(store, NexusStore):
            raise TypeError("NexusStore required")
        self.store = store

    def register(self, rule: BooleanTransition, *, context_id: str, revision: str,
                 oracle: Callable[[int], int], report: ConvergenceReport) -> str:
        _label(context_id)
        _label(revision)
        if not isinstance(rule, BooleanTransition) or rule.width > 12:
            raise ValueError("exhaustive registration supports width <= 12")
        if not isinstance(report, ConvergenceReport) or not report.converged or not report.external_evidence:
            raise ValueError("independent convergence report required")
        for state in range(1 << rule.width):
            if rule.apply(state) != _bits(oracle(state), rule.width):
                raise ValueError("rule failed exhaustive verification")
        payload = {"kind": KIND, "context_id": context_id, "revision": revision,
                   "width": rule.width, "terms": [[hex(m), hex(v)] for m, v in rule.terms],
                   "count": rule.count, "verified_inputs": 1 << rule.width}
        return self.store.append(payload, report=report).digest

    def recall(self, receipt: str, *, context_id: str, revision: str) -> BooleanTransition | None:
        _label(context_id)
        _label(revision)
        node = self.store.read(receipt)
        if node is None:
            return None
        p = node.payload
        if p.get("kind") != KIND or p.get("context_id") != context_id or p.get("revision") != revision:
            return None
        width = _width(p["width"])
        if width > 12 or p.get("verified_inputs") != 1 << width:
            return None
        return BooleanTransition(width, tuple((int(m, 16), int(v, 16)) for m, v in p["terms"]), p["count"])
