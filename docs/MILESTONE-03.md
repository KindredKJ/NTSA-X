# NTSA-X Milestone 03 — Indexed Convergence Reuse, Workers and Kindred Contracts

**State:** experimental, runnable Python reference implementation, not production-secured autonomous AI and not a trained post-Transformer model.

## Verified convergence reuse

`ConvergenceLibrary` adds a new `convergence_index` table to the existing `NexusStore` SQLite database. It does **not** migrate, rewrite, clear, or reconcile existing `nodes` or `edges` tables. A lookup key binds:

- `context_id`: exact validation target
- `scope`: algorithm/operator versions, inputs' meaning, goal, policy and constraints
- `revision`: independent verifier and evidence-policy version
- `input_digest`: canonical SHA-256 of *both* initial trajectories (raw input not stored in the index)

The indexed value points to an additional, append-only Nexus binding node whose payload commits the complete lookup key, whose parent points to the original verified convergence node, and whose hash-linked ancestry is checked on retrieval. A mutated/redirected SQL index row must still match its binding's content before a result may be considered.

`IndexedCounterOrbitEngine` wraps the previous `CounterOrbitEngine` without replacing it. It looks up an exact key, then re-runs forward and reverse projections, contradiction checks, the **current** independent verifier, and evidence gates. If these checks fail, it computes instead of accepting recalled results. Only a freshly calculated convergence is promoted into the index; an explicit legacy digest recall is not promoted under arbitrary new inputs.

A verifier's `external=True` check is only a caller-supplied attestation. No cryptographic source authentication, temporal freshness proof, trusted witness signatures or domain-truth guarantee is implemented. A caller MUST change `scope` or `revision` if requirements, operators, external facts, or validation rules have materially changed.

### Higher-order convergence composition

`ConvergenceLibrary.compose(..., members=[digest1,digest2], ...)` creates a parent-linked macro convergence node, keeping the verified source nodes traceable. It demands a *new* converged report that includes agreement across both projected paths and an externally marked evidence check. This is **not** automatic synthesis, rule-learning, or optimization; composition is explicitly authorized by the caller and must be independently validated as a whole. The caller must check that the `forward` and `reverse` states project to `resolved`; the raw store cannot infer domain semantics.

## Plugin workers

`PluginRegistry` accepts explicit immutable manifests with `plugin_id`, `version`, `capability=compute.scalar/v1` and an allow-list from **increment, decrement, identity and negate**.

`PluginWorker` sends a strict JSON `value,amount` command to the repository's fixed arithmetic worker in a separate Python process. It uses an argument vector (not a shell), a bounded timeout and input/output ranges; malformed requests, unregistered names/versions and invalid outputs fail closed. `paired_operator` wraps two registered operations into a `PairedOperator` for the existing orbital runtime.

**Security boundary:** these are **trusted fixed-function workers**, not arbitrary third-party plugins. Subprocess separation and timeouts do NOT stop a malicious subprocess from using inherited OS privileges or its environment, nor provide filesystem, network, memory, process-tree or CPU quotas. Full sandboxing requires container/VM/OS-level restricted identities, filesystem and network policy, independent signing and audit, and platform-specific termination of descendant processes. Do not execute or register untrusted source in this prototype. This build performs no live financial, network, blockchain, or external effects.

## Kindred contracts

`KindredContracts` packages versioned, metadata-only **local envelopes** for `kindred.zero-point`, `kindred.umr-udeca`, `kindred.strata`, `kindred.root`, `kindred.brainstem`, and `kindred.materializer`. It enforces a read-only digest reference for Zero Point and observation-only event metadata for other ports.

**Not connected:** these contract definitions do not yet authenticate to, call, synchronize with or deploy the actual Kindred services. Before external routing, implement mutual authentication, deny-by-default authorization, encryption, replay protection, schema negotiation, backpressure, proper content redaction, and receipts.

## Run

From the project root with Python 3.12+:

```bash
python -m unittest discover -s tests -v
python -m examples.demo_milestone03
```

For persistent indexed reuse across processes:

```bash
python -m examples.demo_milestone03 --db ./my-research-nexus.sqlite3
```

The demo's verifier is a test oracle. To establish actual research evidence, provide trusted independent validators, real problem suites and comparable baseline measurements. GitHub CI runs unit/integration tests and smoke demos on Windows and Linux.

## Subsequent gates

1. Formal independent evidence record/provenance and signature validation, freshness-aware invalidation.
2. Transactional commit-plus-index publication and crash-recovery reconciliation without touching canonical records.
3. Real isolated third-party plugin runtime with OS-level capabilities and platform-specific process trees.
4. Authorized, user-approved adapters to existing Kindred installations using their actual schemas.
5. Higher-order macro candidate discovery and quality/cost benchmarks against equivalent baselines.
