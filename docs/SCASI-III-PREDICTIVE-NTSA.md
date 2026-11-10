# SCASI III × NTSA-X — predictive intelligence research extension

**Status:** Additive, testable research runtime. Not a trained SCASI III foundation model and not evidence of physical negative latency, superintelligence, or a validated Effective Intelligence Multiplier (EIM).

## Mechanism

1. **Forward projection:** Operator-defined signals and recent session history rank likely upcoming objectives.
2. **Reverse constraint projection:** For each objective, reason backward from success to explicit prerequisite knowledge.
3. **NTSA-X recall:** Use the existing `MemoryFabric.recall` port to inspect source-associated knowledge. Require all prerequisite terms for a candidate record; partial matches are missing.
4. **Knowledge-gap mapping:** Label each required item `missing`, `stale`, `unverified`, or `attested` based on record metadata. The current NTSA-X write API does not create verified records; observed data remains unverified.
5. **Condensation:** Perform exact normalized deduplication of recalled snippets, preserving source attribution. No unsupported semantic fusion, automatic truth promotion, or weight updates.
6. **Ahead-of-time warm:** Call `warm()` to compute candidate requirements and recall prior to anticipated inputs. Later matching `resolve()` uses a bounded TTL cache. Prefetches are read-only; potential savings are workload- and cache-dependent, never guaranteed.
7. **Recursive measurement:** `record_outcome` maintains a short success history and changes subsequent objective ranking. This is a policy heuristic, not self-rewriting executable code or neural parameter training.

## Existing NTSA-X integration

The new module `ntsa_x/scasi_iii.py` is completely additive and preserves checkpoint formats, the native X/Y/Z recurrent model, `MemoryFabric`, `UnifiedCognitiveMechanism`, and deployed routes.

Use the adapter with existing native cognition:

```python
from ntsa_x.scasi_iii import Objective, PredictiveSuperCycle, native_scasi_extension
from ntsa_x.memory_fabric import MemoryFabric
from ntsa_x.unified_cognition import UnifiedCognitiveMechanism

memory = MemoryFabric("scasi_memory.sqlite", writable=False)
objectives = (Objective("research", ("knowledge gap", "train model"),
                        ("training data rights", "held out evaluation")),)
scasi = PredictiveSuperCycle(objectives, memory=memory, session="research")
scasi.warm()
# Inject native_scasi_extension(scasi) into the existing, operator-constructed
# extensions tuple passed to UnifiedCognitiveMechanism(model, ...).
```

No automatic shell, web, network or live execution happens in this adapter. Production retrieval requires a separate authenticated evidence provider with source trust verification, a tenant-aware memory policy, and a consented update mechanism.

## Running

From the root of the existing NTSA-X repository:

```bash
python -m unittest tests.test_scasi_iii -v
python -m examples.demo_scasi_iii
```

The tests use deterministic in-process fixtures. The demo uses the actual `MemoryFabric` SQLite interface and does not require model weights, GPU access, credentials, or network connectivity.

## Benchmark design (EIM remains unmeasured)

Benchmark paired cold and prewarmed requests with identical objectives and evidence. Report hit rate, median/p95 request latency, total preparation+request compute, factual evidence coverage, false-positive gap closure, accuracy on held-out outcomes, RAM, and net benefit across realistic request probabilities. Time spent before input arrival is **not zero computational cost**. In the next experiment, compare baseline NTSA-X, NTSA-X with recall, and SCASI III warmed NTSA-X at fixed budgets and equal held-out sets. Neither fewer parameters nor frontier-level reasoning has been established by this prototype.

## SCASI III.1: knowledge-gap discovery, native hook and benchmarks

The additive native-scasi evolution coordinator extracts only explicitly requested unknowns, probes session-aware NTSA-X memory, and preserves unverified status. It ranks existing objectives using observed previous requests and prewarms a bounded shortlist. A new, unmatched question remains missing or unverified, never magically solved. The evidence-age policy is separate from the prefetch cache TTL. After memory changes, call controller.invalidate() before relying on cached context. No model weights or checkpoints change.

Use coordinator.attach_to_model(loaded_native_model) to create the real UnifiedCognitiveMechanism with the additive SCASI extension; it can now surface previously unregistered explicit knowledge questions. The integration test uses actual NTSA-X SQLite MemoryFabric and cognition runtime with a deterministic stand-in model, not a trained model accuracy evaluation.

Run: python -m unittest tests.test_scasi_iii tests.test_scasi_iii_native_integration -v
Run: python -m examples.benchmark_scasi_iii --requests 300 --repeats 7

The synthetic lexical-recall benchmark compares direct recall, cold SCASI, and warmed SCASI and includes all prewarming costs in total time. It measures orchestration and memory retrieval ONLY, not neural inference, truth verification, or Effective Intelligence Multiplier. Held-out neural model benchmarks remain future work.
