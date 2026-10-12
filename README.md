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

This is a prototype, not an authorization sandbox: arbitrary Python callbacks must be trusted. Verification check labels are attestations from an injected oracle, not cryptographically established external facts. The foundation above is complemented by later milestones, but trained-model validation, authenticated external evidence, production Materializer connectivity, and empirical benchmarks remain future work.

## Milestone 02 — Paired orbits + Nexus graph + live local Materializer

**Implemented in the research prototype:**

- Two coordinated forward/reverse computational paths, agreement checks, contradiction veto and a **shared bounded abstract compute budget**.
- **Persistent SQLite Nexus DAG** with multiple parents, canonical SHA-256 lineage verification, cross-session recall and transactional append-only application operations.
- **Local Materializer browser dashboard** with Server-Sent Events, read-only orbit traces, convergence/commit events and Nexus counters. The viewer is hosted only on loopback and is not a deployed integration with an existing Kindred Materializer system.

**Watch locally:**

```bash
python -m unittest discover -s tests -v
python -m examples.demo_counter_orbit --watch
```

Open the localhost URL printed by the process and follow the terminal prompt. See [Milestone 02 instructions](docs/MILESTONE-02.md).

This remains a research prototype; external-proof labels are verifier assertions, not independent attestation, and in-process callbacks are not sandboxed. Do not store secrets in this prototype.

## Milestone 03 — Verified reuse and capability-bounded workers

- `ConvergenceLibrary` builds an **additive**, immutable SHA-linked lookup index for exact-input, context-scoped and verifier-revision-scoped convergence reuse.
- `IndexedCounterOrbitEngine` checks the current evidence and both trajectory projections again before accepting prior results. It preserves the existing `CounterOrbitEngine` API.
- Higher-order convergence composition explicitly links multiple verified source nodes; composite results must undergo their own verification.
- Versioned fixed-function `PluginWorker` operations run in separate time-limited Python processes with registered capability manifests. These are **not hardened sandboxes** for untrusted code.
- `KindredContracts` defines local-only, metadata-based adapter packets for Zero Point, UMR/uDECA, Strata, Kindred Root, Brainstem, and Materializer. **No remote Kindred services have been connected.**

Run:

```bash
python -m unittest discover -s tests -v
python -m examples.demo_milestone03
```

Read [Milestone 03 documentation](docs/MILESTONE-03.md) for architecture, limitations, and security requirements.

## Run locally

Python 3.12+; no third-party dependencies required:

```bash
python -m unittest discover -s tests -v
PYTHONPATH=. python examples/demo_orbit.py
```

See [Execution prototype](docs/EXECUTION.md), [Materializer integration contract](docs/INTEGRATIONS.md), and [Validation gates](docs/VALIDATION.md).

## Public repository notice

The repository is public; a publishing action alone does not grant a software license. See [LICENSE-NOTICE.md](LICENSE-NOTICE.md). Avoid adding secrets or unpublished proprietary assets.
