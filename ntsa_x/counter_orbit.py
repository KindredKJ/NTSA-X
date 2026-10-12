"""Paired forward/reverse computational orbits with shared abstract budget.

Agreement between counter-orbits is NOT external truth. An independent verifier
must supply all convergence checks; this is a prototype, not a security sandbox.
"""
from copy import deepcopy
from dataclasses import dataclass
from math import isfinite
from typing import Callable, Mapping, Sequence

from .convergence import Check, ConvergenceEngine, ConvergenceReport
from .core import Scheduler
from .events import EventStream
from .nexus import NexusStore


@dataclass(frozen=True)
class PairedOperator:
    name: str
    forward: Callable[[dict], Mapping]
    reverse: Callable[[dict], Mapping]
    axis: tuple[int, int, int] = (1, 0, 0)
    estimated_cost: float = 2.0
    priority: int = 0
    eligible: Callable[[Mapping, Mapping], bool] | None = None
    external_effects: bool = False

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Operator name must be nonempty")
        if not callable(self.forward) or not callable(self.reverse):
            raise TypeError("Both trajectory operators must be callable")
        if len(self.axis) != 3 or any(type(x) is not int for x in self.axis):
            raise ValueError("axis must be three integer deltas")
        if not isfinite(self.estimated_cost) or self.estimated_cost <= 0:
            raise ValueError("Estimated cost must be finite and positive")


@dataclass(frozen=True)
class CounterOrbitResult:
    forward_state: dict
    reverse_state: dict
    resolved: dict | None
    verified: bool
    committed_digest: str | None
    steps: int
    spent: float
    position: tuple[int, int, int]
    reason: str
    report: ConvergenceReport | None


class CounterOrbitEngine:
    """Runs both trajectories per admitted step; no external effects admitted."""

    def __init__(
        self, *, verifier: Callable[[Mapping], Sequence[Check]],
        project_forward: Callable[[Mapping], Mapping],
        project_reverse: Callable[[Mapping], Mapping],
        store: NexusStore | None = None,
        stream: EventStream | None = None,
        contradictions: Callable[[Mapping, Mapping], Sequence[str]] | None = None,
        max_steps: int = 16,
        max_cost: float = 100.0,
    ):
        if not all(callable(x) for x in (verifier, project_forward, project_reverse)):
            raise TypeError("verifier and both projections are required")
        if type(max_steps) is not int or max_steps < 1:
            raise ValueError("max_steps must be positive")
        if not isfinite(max_cost) or max_cost <= 0:
            raise ValueError("max_cost must be positive and finite")
        self.verifier, self.project_forward, self.project_reverse = verifier, project_forward, project_reverse
        self.store = store
        self.stream = stream if stream is not None else EventStream()
        self.contradictions = contradictions
        self.max_steps, self.max_cost = max_steps, max_cost
        self.scheduler = Scheduler()
        self.convergence = ConvergenceEngine()

    def _check(self, forward: Mapping, reverse: Mapping):
        forward_view = self.project_forward(deepcopy(dict(forward)))
        reverse_view = self.project_reverse(deepcopy(dict(reverse)))
        if not isinstance(forward_view, Mapping) or not isinstance(reverse_view, Mapping):
            raise TypeError("Projections must return mappings")
        forward_view, reverse_view = deepcopy(dict(forward_view)), deepcopy(dict(reverse_view))
        conflicts = tuple(self.contradictions(deepcopy(forward_view), deepcopy(reverse_view))) if self.contradictions else ()
        checks = tuple(self.verifier(deepcopy(forward_view)))
        report = self.convergence.evaluate(
            checks, contradictions=conflicts, forward=forward_view, reverse=reverse_view,
        )
        self.stream.emit(
            "counter_orbit.compare", agreement=report.counter_orbit_agreement,
            score=report.score, contradiction_count=len(report.contradictions),
            external_evidence=report.external_evidence,
        )
        self.stream.emit(
            "convergence.check", score=report.score, check_count=len(checks),
            missing_count=len(report.missing), failed_count=len(report.failed),
            contradiction_count=len(report.contradictions),
            counter_orbit_agreement=report.counter_orbit_agreement,
            external_evidence=report.external_evidence, verified=report.converged,
        )
        self.stream.emit("verify.pass" if report.converged else "verify.fail", verified=report.converged)
        return forward_view, reverse_view, report

    def run(
        self, forward_initial: Mapping, reverse_initial: Mapping,
        operators: Sequence[PairedOperator], *, context_id: str,
        parents: Sequence[str] = (), recall_digest: str | None = None,
    ) -> CounterOrbitResult:
        if not isinstance(forward_initial, Mapping) or not isinstance(reverse_initial, Mapping):
            raise TypeError("Initial trajectories must be mappings")
        if not isinstance(context_id, str) or not context_id.strip() or len(context_id) > 256:
            raise ValueError("context_id must be a nonempty bounded string")
        forward, reverse = deepcopy(dict(forward_initial)), deepcopy(dict(reverse_initial))
        position, spent, report = (0, 0, 0), 0.0, None
        self.stream.emit("observe", available=len(operators), max_steps=self.max_steps)
        if recall_digest is not None:
            if self.store is not None:
                try:
                    node = self.store.read(recall_digest)
                except ValueError:
                    node = None
                if node and node.payload.get("context_id") == context_id and isinstance(node.payload.get("forward"), dict) and isinstance(node.payload.get("reverse"), dict):
                    recalled_forward = node.payload["forward"]
                    recalled_reverse = node.payload["reverse"]
                    forward_view, reverse_view, report = self._check(recalled_forward, recalled_reverse)
                    if report.converged and node.payload.get("resolved") == forward_view:
                        self.stream.emit("recall.hit", digest=recall_digest, verified=True)
                        self.stream.emit("orbit.close", reason="verified_recall", closure=True)
                        return CounterOrbitResult(
                            recalled_forward, recalled_reverse, forward_view, True,
                            recall_digest, 0, 0.0, position, "verified_recall", report,
                        )
            self.stream.emit("recall.miss", reason="missing, invalid, stale, or unverified")
        self.stream.emit("orbit.start", max_steps=self.max_steps, remaining_budget=self.max_cost)
        for step in range(1, self.max_steps + 1):
            selected = None
            remaining = max(0.0, self.max_cost - spent)
            for operator in sorted(operators, key=lambda op: (-op.priority, op.name)):
                if operator.external_effects:
                    self.stream.emit("plugin.denied", operator=operator.name, reason="external effects disabled")
                    continue
                if operator.eligible and not operator.eligible(deepcopy(forward), deepcopy(reverse)):
                    continue
                gate = self.scheduler.gate(
                    verified_recall=False, estimated_cost=operator.estimated_cost, remaining_budget=remaining,
                )
                self.stream.emit(
                    "operator.gate", operator=operator.name,
                    enabled=gate.enabled, reason=gate.reason, remaining_budget=remaining,
                )
                if gate.enabled:
                    selected = operator
                    break
            if selected is None:
                self.stream.emit("budget.exhausted", spent=spent, reason="no eligible affordable pair")
                self.stream.emit("orbit.close", closure=False, reason="no_operator")
                return CounterOrbitResult(forward, reverse, None, False, None, step-1, spent, position, "no_operator", report)
            # Compute both candidates without changing prior state until both succeeded.
            new_forward = selected.forward(deepcopy(forward))
            new_reverse = selected.reverse(deepcopy(reverse))
            if not isinstance(new_forward, Mapping) or not isinstance(new_reverse, Mapping):
                raise TypeError("Both operators must return mappings")
            forward, reverse = deepcopy(dict(new_forward)), deepcopy(dict(new_reverse))
            spent += selected.estimated_cost
            position = tuple(a+b for a,b in zip(position,selected.axis))
            self.stream.emit(
                "counter_orbit.step", step=step, operator=selected.name,
                axis_x=position[0], axis_y=position[1], axis_z=position[2],
                spent=spent, remaining_budget=max(0.0, self.max_cost-spent),
            )
            forward_view, reverse_view, report = self._check(forward, reverse)
            if report.converged:
                digest = None
                if self.store is not None:
                    node = self.store.append(
                        {"context_id": context_id, "resolved": forward_view,
                         "forward": forward, "reverse": reverse},
                        parents=parents, report=report,
                    )
                    digest = node.digest
                    self.stream.emit(
                        "nexus.commit", digest=digest, parent_count=len(node.parents),
                        verified=True, node_count=self.store.stats()["nodes"],
                    )
                    self.stream.emit("commit.append", digest=digest, verified=True, parent_count=len(node.parents))
                self.stream.emit("orbit.close", closure=True, reason="verified_convergence")
                return CounterOrbitResult(
                    deepcopy(forward), deepcopy(reverse), forward_view, True,
                    digest, step, spent, position, "verified_convergence", report,
                )
        self.stream.emit("budget.exhausted", spent=spent, reason="step limit")
        self.stream.emit("orbit.close", closure=False, reason="step_limit")
        return CounterOrbitResult(forward, reverse, None, False, None, self.max_steps, spent, position, "step_limit", report)
