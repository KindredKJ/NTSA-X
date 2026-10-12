# NTSA-X Milestone 05 — Persistent Scalar

**Performance flagship:** Persistent Scalar — supervised, reusable, trusted, fixed-function scalar workers, avoiding Python interpreter startup on every synthetic task. This is a measured *systems optimization*, not a demonstrated 100× increase in model intelligence or AI inference.

## Suped-up capabilities

- **Warm-pool execution:** 1–100 reusable subprocess slots; each leased worker handles only one request at a time, supporting controlled parallel execution.
- **Bounded JSONL protocol:** explicit versioned handshake, unique request IDs, newline framing and a maximum message size. Unsupported operations never enter the worker.
- **Recovery and isolation:** on a failed response, timeout, process crash or request-ID mismatch, fail the *individual* task, recycle its worker, and allow independent jobs to proceed. Dependents remain blocked until their parents are verified.
- **Lifecycle control:** configurable process recycling after a maximum number of tasks, bounded startup/response timeouts, explicit pool close, and observable process-start/restart/request/failure counts.
- **Existing integrity guarantees:** the factory still validates a DAG, independently recomputes each scalar result, and performs verification-gated, parent-linked Nexus commits. Persistent Scalar never overwrites existing verified records.
- **Evidence-driven performance:** a repeatable 100-job comparison across 1, 4, 8 and 16 worker slots on Windows and Linux, versus the existing one-shot/single-worker baseline. Benchmarks report rates and variability, plus worker startup overhead and Nexus verification.

## Run from repository root

```powershell
py -3.12 -m unittest discover -s tests -v
py -3.12 -m examples.benchmark_persistent_scalar --tasks 100 --workers 1,4,8,16 --repeats 3 --output persistent-scalar-benchmark.json
```

Use `PersistentPluginWorker` as a context manager with `ProductionFactory(plugin_worker=pool)`. The old `PluginWorker` remains available for correctness comparisons and compatibility.

## Scientific and security boundaries

This implementation executes only the repository's bundled arithmetic operations: increment, decrement, identity and negate. It **does not** load arbitrary third-party plugin code. Subprocess separation is not a hardened OS security sandbox, and it does not enforce memory/CPU quotas or prevent inherited local filesystem access. Values are verified against a separate local arithmetic oracle, not independently authenticated real-world facts.

A faster *task factory* does not make a trained language model faster. The ×100 target remains unproven until repeated, reproducible, correctness-equivalent throughput results reach it.

## Beyond this milestone

Next performance tracks: batched fixed-function RPC, reduced message serialization, dedicated worker lifecycle probes, supervised process-tree termination, backpressure/fairness, stable resource telemetry, and actual-model workloads with comparable baseline accuracy.
