# Recall-First Systems (RFS) — Independent Research Category

**Category ID:** `KINDRED-RFS-001`
**Status:** Isolated research fork; architecture exploration, not performance-verified
**Parent snapshot:** `KindredKJ/NTSA-X` on `main`, copied into `category/recall-first-systems-v1`
**Original preservation rule:** `main` and the historical NTSA-X architecture, code, checkpoints, datasets, tests, receipts, and runtime state must remain unchanged.

## Purpose
Explore **recall-before-work** beyond inference: use trusted, context-matched, version-aware, provenance-checked prior computation and state to avoid unnecessary model execution, training, tool invocations, network I/O, orchestration, materialization, and repeated verification.

This category is an independent candidate, not an NTSA-X upgrade or a replacement for NTSA-X. No changes from this branch are to be merged into `main` without a separately authorized decision.

## Required architectural boundaries
1. **No destructive modifications** to inherited `ntsa_x/`, original architecture documents, source lineage, datasets, checkpoints, evidence, or build assets. Preserve originals byte-for-byte when exploring alternatives.
2. **Separate category namespace:** build new candidate modules under `rfs/` and tests under `tests_rfs/`. Integrate existing features only through explicit, versioned, read-only adapters until an integration is independently verified.
3. **Independent state and artifacts:** separate namespaces, SQLite stores, indices, caches, experiment runs, model weights, and signed receipts; do not write to existing NTSA-X databases or artifacts.
4. **Provenance and invalidation:** successful recall requires input, context, dependency versions, policy, evidence freshness, authorization scope, and output validity checks. Failed validation routes to fresh bounded computation.
5. **Separation of claims:** distinguish designed, implemented, locally tested, externally validated, and production-proven. No speed, energy, accuracy, or savings claim without a reproducible benchmark.
6. **External effects:** deny-by-default. Recall may never substitute for current authorization, live environmental checks, or transaction-specific validation.

## Exploration tracks
| Track | Gate before expensive work | Candidate savings |
| --- | --- | --- |
| RFS-01 Reasoning | Verified answer and derivation recall | Fewer inference steps |
| RFS-02 Learning | Verified gradient/update, sample utility, and delta recall | Fewer redundant training operations |
| RFS-03 Scheduling | Reuse approved execution plans and dependency closure | Fewer planner invocations |
| RFS-04 Tools | Revalidate tool result provenance and freshness | Fewer duplicate calls |
| RFS-05 I/O and Storage | Content-addressed delta/materialization recall | Fewer data transfers and writes |
| RFS-06 Agents | Scope-bound shared task memory and verified handoffs | Less duplicated multi-agent work |
| RFS-07 Verification | Reuse valid proof fragments, not expired authority | Less repeated checking |
| RFS-08 Adaptive Runtime | Cost-aware confidence gating and graceful cache miss | Less compute while maintaining correctness |

## Baseline experiment contract
- Record input dataset hashes, hardware/runtime/software versions, measured energy if instrumentation exists, response quality, correctness, cache-hit integrity, latency (p50/p95/p99), peak memory, CPU/GPU utilization, network bytes, and storage bytes.
- Compare **uncached baseline**, **exact-match recall**, **contextual recall**, **partial reuse**, **invalidated/stale recall**, and **adversarial recall**, with identical task sets and correctness gates.
- Measure end-to-end cost including indexing, verification, memory pressure, warmup, invalidation, and retrieval failures. Report failures and negative results.
- Default to correctness-preserving fallback; fail closed for authorization-sensitive operations.

## Promotion and version policy
- Branch line: `category/recall-first-systems-v1`.
- Experimental code and artifacts belong to this category, not canonical NTSA-X.
- A future separate repository can be established with complete Git history once repository-creation access is available. Until then, this Git branch is a full source snapshot with independent subsequent history.
- No PR, merge, publication claim, or replacement of original NTSA-X is authorized by establishing this category alone.

## Cross-project independence
JETTISON, NTSA-D, and other distinct programs remain separate. This category must not silently import their weights, state, checkpoints, governance, or architecture as canonical dependencies.