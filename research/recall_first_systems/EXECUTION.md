# RFS-001 — Executable Recall-Before-Work Kernel

**Experimental branch:** category/recall-first-systems-v1. Canonical NTSA-X main is not modified.

## Architectural flow

Typed deterministic work -> canonical identity -> tenant+context+dependency gate -> isolated SQLite index -> expiry/integrity/current verifier gate -> (reuse | fresh worker) -> fresh verifier -> cache pure result.

Scope identity includes tenant, namespace, operation, arguments, explicit context snapshot, implementation/verifier/policy revisions, dependency digests, and TTL. Neither lexical similarity nor an embedding match can establish correctness. Raw cached outputs are never authorization tokens, grants, payment approvals, or external actions.

This module creates an isolated .rfs.sqlite3 database, and never touches the inherited NTSA-X Nexus or learning stores. A SHA-256 checksum detects accidental corruption but is not cryptographic authentication or a defense against attackers who can rewrite the database. Callers must securely handle tenant identifiers and code/policy revisions, and use independent verifier logic for their domain.

## Run (Python 3.12+, no dependencies)

    python -m unittest discover -s tests_rfs -v
    python -m rfs.benchmark --unique 12 --repeats 4 --iterations 50000 --output rfs-benchmark.json

The benchmark compares end-to-end repeated deterministic CPU work, with a separately reported fixture setup cost. It reports measured p50/p95/p99, invocation counts, and relative speedup for that workload. It **does not** measure LLM tokens, joules, real-world energy, security guarantees, or training speed.

## Operational contracts

- Require explicitly pure, deterministic worker operations. Arbitrary callables are not sandboxed; callers must enforce purity through audited adapters.
- A mandatory current verifier runs on **every** cache read and every fresh result. Cache verifier failure invalidates the entry and triggers fresh computation; fresh verifier failure is raised and not cached.
- TTL uses the configured clock; implementation, policy and external state revisions **must** change on relevant changes. No automatic external freshness detection is claimed.
- No semantic or approximate replay in RFS-001. These require a distinct probabilistic policy, independent evaluation sets, abstention and correctness gates.
- Isolation within one SQLite file is logical, not encryption. Never store credentials in arguments/context. Tenant-sensitive workloads require OS protections or stronger isolation.
- No proof of authorization, clock synchronization, cross-host consistency or hostile code isolation is provided.
- Worker exceptions propagate; nothing is cached. Callers own deadlines, retries and per-worker sandboxing.
- The initial kernel serializes in-process operations with a lock. Profile real contention before changing concurrency.

## Independent research tracks

1. **RFS-02:** compiler-level graph-node memoization with a versioned dependency DAG, subtree invalidation and differential execution.
2. **RFS-03:** immutable agent-plan recall and handoffs with tenant-specific delegation boundaries.
3. **RFS-04:** read-only prefix/KV checkpoint adapters for modern inference engines; require per-tenant cache isolation.
4. **RFS-05:** active data selection and evidence-gated learning-update reuse for native recurrent training, including repetitive-character collapse diagnostics.
5. **RFS-06:** streaming multimodal deltas and materializer receipt reconstruction under corruption, crash and cold-boot tests.
6. **RFS-07:** adaptive memory tiers and compute-vs-recall policy with cache-miss costs and measured joules.
7. **RFS-08:** interoperable but separate interfaces to SDPN and other Kindred runtimes through versioned read-only contracts.

**Scientific positioning:** exact-prefix KV caching exists in production inference systems already. Novelty and superiority require comparisons against strong published baselines and independently reproducible results.
