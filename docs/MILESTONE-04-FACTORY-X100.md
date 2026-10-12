# NTSA-X Milestone 04 — ×100 Production Capacity Initiative

**Interpretation:** ×100 is an ambitious *engineering target* for verified work output per unit time, **not** a factual 100× speedup, 100× compute efficiency, or trained-model inference claim. This milestone provides a safe 100-work-item demonstration and instrumentation for measuring real gains.

## Production factory

`ntsa_x.factory.ProductionFactory` accepts a bounded directed acyclic task graph. Each `FactoryTask` specifies a stable ID, a registered pure scalar operation, input value, amount, dependencies and an optional dependency-output sum. The coordinator:

1. Validates uniqueness, valid parent references, a DAG (reject cycles), bounds and supported operation types **before any worker starts**.
2. Schedules ready independent tasks onto a bounded pool of up to **100 configurable worker slots**. The default is **8**, avoiding a sudden 100-process load on a laptop.
3. Invokes the existing `PluginWorker` fixed-function Python child process. There is **no arbitrary untrusted code execution or external action**. The worker timeout still applies.
4. Recomputes the expected scalar result independently in the host and evaluates six convergence checks plus forward/reverse scalar comparison. An invalid worker result is never committed.
5. Records successful results through existing `NexusStore.append`, with dependency digest parents and a deterministic record including the batch/task identity. This preserves cryptographic ancestry and existing database records.
6. Marks failed tasks as `failed`, their descendants `blocked`; unrelated jobs can finish. Exceptions are summarized without spilling input or system details into telemetry.
7. Emits metadata-only `factory.start/dispatch/commit/fail/blocked/close` events for the existing loopback Materializer viewer.

**Trust scope:** This is a bounded fixed-function *job factory*. The local arithmetic oracle is independent of the child-process computation but is **not authenticated external evidence of real-world facts**. Operators and their Python subprocess environment are **not a hostile-code sandbox**. The factory has no network, USD transfer, token minting, external deployment, or autonomous external-effect capabilities.

## Important limitations

- 100 queued or even 100 *simultaneous worker slots* is **not** proof of 100× throughput. Python subprocess startup, CPU, SSD, process memory, OS scheduler, laptop thermals, and disk contention can reduce scalability.
- `ThreadPoolExecutor` coordinates **trusted fixed-function child processes** and records results; threads do not magically bypass the Python GIL for CPU-bound Python callbacks.
- Dependency management is local and synchronous per batch; no persistent distributed job broker, lease/retry protocol, multi-device workers, job auth, signed manifests, cancellation, resource metering or scheduler fairness SLA exists yet.
- Every new successful task creates a Nexus state record. No existing ledger/cache or original source data is overwritten or auto-reconciled.
- The factory currently uses fixed deterministic scalar operations only. It is **not yet a model training factory, code generation pipeline or production deployment system**.

## Run 100-job batch on Windows

From the repo root after `git pull origin main`:

```powershell
py -3.12 -m unittest discover -s tests -v
py -3.12 -m examples.demo_factory --tasks 100 --workers 8
py -3.12 -m examples.demo_factory --tasks 100 --workers 8 --watch
```

The `--watch` command prints the loopback Materializer URL. Open it on the same computer, press Enter to begin, then press Enter again to stop. Set `--workers` to match your actual system capacity rather than automatically selecting 100.

Use an explicit `--db ./ntsa-factory.sqlite3` for persistent verified results. Omit it for an ephemeral, throwaway demo store.

## Measure actual rate

```powershell
py -3.12 -m examples.benchmark_factory --tasks 100 --workers 8 --compare --output factory-bench.json
```

This performs **two real fresh-store batches**: one with 1 worker and one with 8 workers. It records elapsed time, verified completions per second, observed peak in-flight workers, and `sequential_elapsed / parallel_elapsed`. A ratio of 4.0 means 4× for that fixed test workload on that machine, **not 100×**. Ratios may vary and may fall below 1.0. A single run is only a sanity check; rigorous results require repeated runs, stabilized warmup, host metrics, distribution/variance and comparable correctness constraints.

GitHub Actions performs concurrent Windows/Linux suites, compiles Python source, executes all existing demos, and measures a 100-task baseline/parallel pair on **both** OS families. It uploads the two JSON reports as retained workflow artifacts. A `verified-gate` job fails if any test or benchmark job fails; no performance-number threshold is asserted without an evidence-backed baseline.

## Throughput improvement roadmap

1. **Now:** separate correctness, safety and throughput measurements; cap concurrent workers; preserve lineage.
2. **Next:** collect benchmark samples across Windows/Linux and target the measured bottleneck (worker startup, hashing/SQLite, OS process overhead).
3. Replace one-subprocess-per-job with **supervised persistent workers**, explicit process resource limits and per-worker auth, while preserving deterministic contracts and no unknown code loading.
4. Add typed dependency streaming and sealed task manifests, bounded disk queues, retries with idempotency keys, reconciliation and failure-injection tests.
5. Add rate-aware schedulers, batch coalescing, vectorized deterministic operations and reduced serialization overhead. For model tasks, measure real GPU kernels separately.
6. Only claim ×100 against a **specified, reproducible baseline** after measured correctness-equivalent throughput and fault-recovery evidence.

## CI and operational safety

The workflow cancels superseded runs for the same Git reference to reduce wasted runner minutes. Matrix jobs execute in parallel, not 100 concurrent GitHub deployments. Public CI artifacts contain synthetic benchmark metadata, not personal/user data or wallet credentials. No paid cloud infrastructure has been provisioned.
