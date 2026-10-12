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

## Milestone 04 — ×100 verified production-capacity initiative

- **Factory scheduler:** bounded, dependency-aware work batches; a 100-task example and up to 100 explicitly configured local worker slots (default 8).
- **No blind promotion:** each fixed-function worker result is independently recalculated by a local arithmetic oracle before an append-only, parent-linked Nexus commit. Failed tasks block dependent work.
- **Factory Materializer:** read-only local dashboard for queue progress, verified completions, worker peaks and measured task rate.
- **Measured benchmarks:** JSON reports compare fresh sequential and parallel 100-task batches. Windows and Linux CI compile source, run tests/demos, capture benchmark artifacts, and enforce a combined test gate.

**×100 is a goal, not a measured speedup.** Worker throughput varies by machine and workload; this prototype is neither a trained AI model nor a production plugin sandbox.

```bash
python -m examples.demo_factory --tasks 100 --workers 8 --watch
python -m examples.benchmark_factory --tasks 100 --workers 8 --compare
```

See [Milestone 04: Factory ×100](docs/MILESTONE-04-FACTORY-X100.md).

## Milestone 05 — Persistent Scalar

**Persistent Scalar** keeps a bounded number of trusted scalar worker processes warm, reuses them across tasks, verifies request/response IDs, and recycles or replaces failing workers. The existing one-shot API remains intact. It plugs into the verified factory without changing the Nexus ledger contract.

The 100-task benchmark compares **1, 4, 8, and 16 persistent workers** against the one-shot single-worker baseline, reporting absolute rates and variability on Windows and Linux. This is an optimization of synthetic fixed-function task processing, **not** a claim of 100× AI-model acceleration.

```bash
python -m examples.benchmark_persistent_scalar --tasks 100 --workers 1,4,8,16 --repeats 3
```

See [Persistent Scalar technical specification](docs/MILESTONE-05-PERSISTENT-SCALAR.md).

## Run locally

Python 3.12+; no third-party dependencies required:

```bash
python -m unittest discover -s tests -v
PYTHONPATH=. python examples/demo_orbit.py
```

See [Execution prototype](docs/EXECUTION.md), [Materializer integration contract](docs/INTEGRATIONS.md), and [Validation gates](docs/VALIDATION.md).

## Public repository notice

The repository is public; a publishing action alone does not grant a software license. See [LICENSE-NOTICE.md](LICENSE-NOTICE.md). Avoid adding secrets or unpublished proprietary assets.
