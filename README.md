# NTSA-X

**Nexus Transforming Stream Architecture** — an experimental architecture and reference kernel for stateful, evidence-gated, adaptive computation.

> Research-stage project. This is **not** yet a trained foundational AI model or a verified transformer replacement. Do not treat claimed performance, energy savings, external proof, or security boundaries as established.

## Research threads

NTSA-01 Token-Stream · NTSA-02 Byte-to-Byte · NTSA-03 Nexus-Byte · NTSA-04 Byte/Bit-Stream · NTSA-05 Transforming Byte-Scaler · NTSA-06 Chain-Graph / Graph-to-Graph · **NTSA-X** independent recurrent stream machine.

Full lineage and scientific hypotheses: [Architecture](docs/ARCHITECTURE.md).

## Implemented in the reference kernel

- **Orbital scheduler** selects pure in-process operators using priority, changing predicates, remaining abstract cost budget, and bounded steps.
- **Three-axis position** tracks X duality, Y abstraction, and Z lineage coordinates as signed integer orbit deltas.
- **Convergence engine** requires explicit check categories (state, semantic, causal, constraint, evidence, delta), rejects missing checks and contradictions, and can veto on mismatched paired trajectories.
- **Recall-before-compute** rechecks canonical SHA-256 lineage, context matching, and current verification before reuse.
- **Verification-gated commits** only append to an in-memory hash-linked chain after a passing verifier report.
- **Materializer event stream** emits bounded metadata-only trace events through a non-authoritative consumer callback.

This is a prototype, not an authorization sandbox: arbitrary Python callbacks must be trusted. Verification check labels are attestations from an injected oracle, not cryptographically established external facts. Persistent graph storage, learned scheduling, full paired trajectory execution, production Materializer service, and empirical benchmarks are future work.

## Run locally

Python 3.12+; no third-party dependencies required:

```bash
python -m unittest discover -s tests -v
PYTHONPATH=. python examples/demo_orbit.py
```

See [Execution prototype](docs/EXECUTION.md), [Materializer integration contract](docs/INTEGRATIONS.md), and [Validation gates](docs/VALIDATION.md).

## Public repository notice

The repository is public; a publishing action alone does not grant a software license. See [LICENSE-NOTICE.md](LICENSE-NOTICE.md). Avoid adding secrets or unpublished proprietary assets.
