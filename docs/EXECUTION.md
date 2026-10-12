# NTSA-X execution prototype

**Status:** Python 3.12+ research prototype. In-memory only; not a trained language model, autonomous production system, cryptographic attestation authority, sandbox, or demonstrated transformer replacement.

## Run
From repository root:

```bash
python -m unittest discover -s tests -v
PYTHONPATH=. python examples/demo_orbit.py
```

## Core execution

`OrbitEngine.run(initial_state, operators, context_id=..., recall_digest=...)`:

1. Emits metadata-only `observe` event.
2. If supplied, validates hash-linked recall and exact `context_id` match, then **reruns the current verifier**. Successful recall avoids operator cycles.
3. Otherwise begins a bounded orbit. At each step, sorts operators by descending priority/name, re-evaluates eligibility predicates and budgets, and executes the first admitted in-process operator.
4. Advances (X,Y,Z) position by the selected operator's integer `axis` delta and emits `operator.gate` and `orbit.step`.
5. Calls the independently supplied verifier; `ConvergenceEngine` requires state, semantic, causal, constraint, evidence and delta checks all passing, including at least one check marked as external evidence. Contradictions or opposing trajectory disagreement veto closure when supplied.
6. Only after convergence does it append canonical SHA-256 verified lineage and emit `commit.append` then `orbit.close`. Missing evidence or exhausted budgets leave memory unchanged.

**Important:** A check marked `external=True` is an *attestation supplied by the verifier*. The runtime does not independently establish whether its source is authentic. Use a separately trusted validator with verifiable evidence before any consequential deployment.

The `estimated_cost` and `max_cost` values are abstract budget units, **not joules, dollars or measured latency**. Adaptive operator selection is a deterministic priority/predicate/budget policy, not a learned controller. Counter-orbit comparison is available at the `ConvergenceEngine` level; paired trajectory scheduling is future work. In-process Python operator functions are **not sandboxed**; an `external_effects=True` declaration is denied but untrusted callbacks could still cause side effects.

## Materializer event contract

Pass `EventStream(sink=callback)` to `OrbitEngine`. The callback receives a dictionary:

```json
{
  "schema_version": "1.0",
  "event_id": "<trace_id>:<sequence>",
  "trace_id": "<trace_id>",
  "timestamp": "<UTC ISO-8601>",
  "type": "orbit.step",
  "payload": {
    "step": 1, "operator": "increment",
    "axis_x": 1, "axis_y": 1, "axis_z": 0,
    "spent": 1.0, "remaining_budget": 4.0
  },
  "provenance": "ntsa-x",
  "redaction_level": "metadata"
}
```

The schema forbids unrecognized payload fields and raw states. Consumers should reject unknown schema versions and never treat telemetry as trusted instructions or verified ground truth. Sink exceptions are swallowed so rendering failure does not affect runtime decisions. The in-memory event buffer drops oldest entries after `max_events` and counts dropped entries. There is no remote streaming endpoint, auth layer, stable delivery guarantee, persistence, or connected Materializer deployment yet.

## Test coverage

Unittest cases cover successful 3-orbit closure, operator gating, budget exhaustion, denial of declared external effects, failed/missing verification, counter-orbit mismatch veto, contradictory evidence veto, verified recall and context mismatch, tamper detection, bounded event history, and broken renderer isolation.

## Next gates

- Explicit trusted verifier adapters with check provenance and replayable evidence.
- Graph persistence and multi-parent lineage with durable storage; existing `MemoryChain` only stores in process.
- Full forward/reverse paired orbit scheduler with metered resources and contradiction reporting.
- Sandboxed plugin execution and authorization outside the in-process reference implementation.
- Real Materializer connector and end-to-end integration tests.
- Reproducible performance/accuracy comparison against Transformer and SSM baselines.
