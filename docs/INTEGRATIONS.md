# Materializer and Kindred plugin integration contracts

## Materializer
Materializer is an observability/visualization consumer, not a dependency of the inference kernel. Stream typed events for orbit steps, axes and state positions, active operator gates, Nexus traversals, memory recalls, convergence checks, verification decisions, commits, latency and energy measurements or explicitly marked estimates. The renderer must not gain implicit authority to mutate the kernel.

Event envelope: schema_version, event_id, trace_id, timestamp, type, payload, provenance, redaction_level. Sensitive memory and credentials must be redacted before emission.

## Plugin network
Registry entries specify plugin ID, semantic version, capability manifest, accepted input/output schemas, resource budgets, isolation model, provenance and permission requirements. Operators are selected through capability routing, with deny-by-default external effects and audit logging. A plugin's reported success does not establish verified correctness.

## Existing Kindred components
Adapters may connect UMR/uDECA orchestration, Strata transport, Zero Point verified-state reuse, Kindred root lineage, and Materializer visualization. Keep NTSA-X core independent and version each adapter contract. Do not overwrite canonical ledgers or claim a deployment without runtime verification.

## Minimal event vocabulary
observe, recall.hit, recall.miss, nexus.traverse, orbit.start, orbit.step, operator.gate, convergence.check, verify.pass, verify.fail, commit.append, orbit.close, budget.exhausted, plugin.invoke, plugin.denied.
