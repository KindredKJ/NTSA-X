"""Indexed extension of CounterOrbitEngine, preserving the original runtime API."""
from copy import deepcopy

from .counter_orbit import CounterOrbitEngine, CounterOrbitResult
from .reuse import ConvergenceLibrary


class IndexedCounterOrbitEngine(CounterOrbitEngine):
    """Reuses exact-input convergences, always revalidating current evidence."""

    def __init__(self, *, library: ConvergenceLibrary, **kwargs):
        if not isinstance(library, ConvergenceLibrary):
            raise TypeError("ConvergenceLibrary required")
        if kwargs.get("store") is not library.store:
            raise ValueError("Library and orbital engine must use the same NexusStore")
        super().__init__(**kwargs)
        self.library = library

    def run(self, forward_initial, reverse_initial, operators, *,
            context_id, reuse_scope, verifier_revision,
            parents=(), recall_digest=None):
        """Scope identifies algorithm, constraints, operator versions, and goal.

        Revision must identify the independent verifier's evidence policy.
        Caller is responsible for changing these when applicability changes.
        """
        inputs = (deepcopy(dict(forward_initial)), deepcopy(dict(reverse_initial)))
        # Validate all key parts before executing operators.
        self.library._key(context_id, reuse_scope, verifier_revision, inputs)
        found = self.library.get(
            context_id=context_id, scope=reuse_scope,
            revision=verifier_revision, inputs=inputs,
        )
        if found is not None:
            payload = found.payload
            forward, reverse = payload["forward"], payload["reverse"]
            forward_view, reverse_view, report = self._check(forward, reverse)
            if report.converged and payload["resolved"] == forward_view:
                self.stream.emit("recall.hit", digest=found.digest,
                                 verified=True, reason="indexed_convergence")
                self.stream.emit("orbit.close", closure=True, reason="indexed_convergence")
                return CounterOrbitResult(
                    deepcopy(forward), deepcopy(reverse), forward_view, True,
                    found.digest, 0, 0.0, (0, 0, 0),
                    "indexed_convergence", report,
                )
        self.stream.emit("recall.miss", reason="no verified indexed convergence")
        result = super().run(
            forward_initial, reverse_initial, operators, context_id=context_id,
            parents=parents, recall_digest=recall_digest,
        )
        if result.verified and result.reason == "verified_convergence" and result.committed_digest and result.report:
            self.library.register(
                context_id=context_id, scope=reuse_scope,
                revision=verifier_revision, inputs=inputs,
                node_digest=result.committed_digest, report=result.report,
            )
            self.stream.emit("convergence.promote", digest=result.committed_digest,
                             verified=True)
        return result
