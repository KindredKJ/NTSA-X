"""Budget-bounded adaptive orbit executor with evidence-gated closure."""
from copy import deepcopy
from dataclasses import dataclass
from math import isfinite
from typing import Callable, Mapping, Sequence

from .convergence import Check, ConvergenceEngine, ConvergenceReport
from .core import MemoryChain, Scheduler
from .events import EventStream


@dataclass(frozen=True)
class Operator:
    name: str
    apply: Callable[[dict], Mapping]
    axis: tuple[int, int, int] = (1, 0, 0)
    estimated_cost: float = 1.0
    priority: int = 0
    eligible: Callable[[Mapping], bool] | None = None
    external_effects: bool = False

    def __post_init__(self):
        if not self.name or not self.name.strip():
            raise ValueError("Operator must have a name")
        if len(self.axis) != 3 or any(type(v) is not int for v in self.axis):
            raise ValueError("Axis delta must contain three integers")
        if not isfinite(self.estimated_cost) or self.estimated_cost <= 0:
            raise ValueError("estimated_cost must be finite and positive")
        if type(self.priority) is not int:
            raise TypeError("priority must be integer")


@dataclass(frozen=True)
class OrbitResult:
    state: dict
    verified: bool
    committed_digest: str | None
    steps: int
    spent: float
    position: tuple[int, int, int]
    reason: str
    report: ConvergenceReport | None


class OrbitEngine:
    """Pure in-process operator execution; not a security sandbox.

    The injected verifier is independent of the candidate operator. Its evidence
    checks are attestations supplied by the integrator, not cryptographic proof.
    """

    def __init__(
        self, *, verifier: Callable[[Mapping], Sequence[Check]],
        chain: MemoryChain | None = None,
        stream: EventStream | None = None,
        max_steps: int = 16,
        max_cost: float = 100.0,
    ):
        if not callable(verifier):
            raise TypeError("Explicit verifier required")
        if type(max_steps) is not int or max_steps < 1:
            raise ValueError("max_steps must be positive")
        if not isfinite(max_cost) or max_cost <= 0:
            raise ValueError("max_cost must be finite and positive")
        self.verifier = verifier
        self.chain = chain if chain is not None else MemoryChain()
        self.stream = stream if stream is not None else EventStream()
        self.max_steps = max_steps
        self.max_cost = max_cost
        self.scheduler = Scheduler()
        self.convergence = ConvergenceEngine()

    def _report(self, state: Mapping):
        checks = tuple(self.verifier(deepcopy(dict(state))))
        report = self.convergence.evaluate(checks)
        self.stream.emit(
            "convergence.check", score=report.score,
            check_count=len(checks), missing_count=len(report.missing),
            failed_count=len(report.failed),
            contradiction_count=len(report.contradictions),
            external_evidence=report.external_evidence,
            verified=report.converged,
        )
        self.stream.emit("verify.pass" if report.converged else "verify.fail", verified=report.converged)
        return report

    def run(self, initial_state: Mapping, operators: Sequence[Operator], *, context_id: str, recall_digest: str | None = None):
        if not isinstance(initial_state, Mapping):
            raise TypeError("initial_state must be a mapping")
        if not isinstance(context_id, str) or not context_id.strip():
            raise ValueError("context_id must identify the validation context")
        if len(context_id) > 256:
            raise ValueError("context_id must be <= 256 chars")
        state = deepcopy(dict(initial_state))
        position = (0, 0, 0)
        spent = 0.0
        report = None
        self.stream.emit("observe", available=len(operators), max_steps=self.max_steps)
        if recall_digest is not None:
            if self.chain.verify(recall_digest):
                block = self.chain.blocks[recall_digest]
                if block.payload.get("context_id") == context_id and isinstance(block.payload.get("state"), dict):
                    candidate = deepcopy(block.payload["state"])
                    report = self._report(candidate)  # Revalidate in the current context.
                    if report.converged:
                        self.stream.emit("recall.hit", digest=recall_digest, verified=True)
                        self.stream.emit("orbit.close", closure=True, reason="verified_recall")
                        return OrbitResult(candidate, True, recall_digest, 0, 0.0, position, "verified_recall", report)
            self.stream.emit("recall.miss", reason="missing, invalid, changed, or unverified")
        self.stream.emit("orbit.start", max_steps=self.max_steps, remaining_budget=self.max_cost)
        for step in range(1, self.max_steps + 1):
            selected = None
            remaining = self.max_cost - spent
            # Re-evaluate predicates every orbit. Stable priority + name tie-break.
            for operator in sorted(operators, key=lambda op: (-op.priority, op.name)):
                if operator.external_effects:
                    self.stream.emit("plugin.denied", operator=operator.name, reason="external effects disabled")
                    continue
                if operator.eligible is not None and not operator.eligible(deepcopy(state)):
                    continue
                gate = self.scheduler.gate(verified_recall=False, estimated_cost=operator.estimated_cost, remaining_budget=remaining)
                self.stream.emit("operator.gate", operator=operator.name, enabled=gate.enabled, reason=gate.reason, remaining_budget=remaining)
                if gate.enabled:
                    selected = operator
                    break
            if selected is None:
                self.stream.emit("budget.exhausted", spent=spent, reason="no eligible affordable operator")
                self.stream.emit("orbit.close", closure=False, reason="no_operator")
                return OrbitResult(state, False, None, step - 1, spent, position, "no_operator", report)
            # Do not mutate the committed/earlier state if an operator fails.
            candidate = selected.apply(deepcopy(state))
            if not isinstance(candidate, Mapping):
                raise TypeError("Operator must return a mapping")
            state = deepcopy(dict(candidate))
            spent += selected.estimated_cost
            position = tuple(p + delta for p, delta in zip(position, selected.axis))
            self.stream.emit(
                "orbit.step", step=step, operator=selected.name, axis_x=position[0],
                axis_y=position[1], axis_z=position[2], spent=spent,
                remaining_budget=max(0.0, self.max_cost - spent),
            )
            report = self._report(state)
            if report.converged:
                block = self.chain.append({"context_id": context_id, "state": state}, verified=True)
                self.stream.emit("commit.append", digest=block.digest, verified=True, parent_count=0)
                self.stream.emit("orbit.close", closure=True, reason="verified_convergence")
                return OrbitResult(deepcopy(state), True, block.digest, step, spent, position, "verified_convergence", report)
        self.stream.emit("budget.exhausted", spent=spent, reason="step limit")
        self.stream.emit("orbit.close", closure=False, reason="step_limit")
        return OrbitResult(deepcopy(state), False, None, self.max_steps, spent, position, "step_limit", report)
