# NTSA-X Milestone 02 — Paired Orbits, Durable Nexus and Live Materializer

**Maturity:** experimental reference implementation, not a trained post-Transformer model or deployed Kindred Materializer network.

## Paired counter-orbital execution

`ntsa_x.counter_orbit.CounterOrbitEngine` runs two separately supplied operators (forward and reverse trajectories) per admitted orbit step, sharing a total abstract cost budget and the three-axis coordinate. The integrator supplies two projection functions into one comparable state representation. The convergence engine checks those views for equality (not proof), independently asks a verifier for six required categories, and optionally checks explicit contradictions. Only conjunction of all required checks, marked external evidence, agreement, and no contradictions can close and commit.

An independent verifier is injected; an `external=True` label remains a caller-supplied claim, **not cryptographically validated evidence**. Python operator callbacks are not sandboxed; untrusted plugins MUST NOT be used. Operator eligibility, selection, and orbital cost are heuristic/prototype behavior, not a learned optimizer, formal energy meter, or safety containment boundary.

## Persistent Nexus graph

`ntsa_x.nexus.NexusStore(path)` uses built-in SQLite in WAL mode (for disk files), foreign-key edges, append-only application operations, SHA-256 canonical JSON hashes, transactional insertion, duplicate handling, multi-parent DAG relationships, recursive ancestry integrity checks, reopening/replay, and lineage traversal. `append` requires a converged evidence-marked `ConvergenceReport`. DB rows are not overwritten by the API; a failed parent check aborts the whole transaction.

**Threat boundary:** SQLite with hashes is not immutable against a database administrator, and verification metadata does not prove the facts in an attestation. There is no encryption-at-rest, key management, signatures, trusted timestamps, independent anchoring, remote replication, authorization, or backup/restore guarantee. Do not store secrets or sensitive user state in the demo.

## Live local Materializer

`LocalMaterializer` serves a browser visualization and Server-Sent Events at a *random loopback-only port*. The read-only dashboard charts paired orbit movement, convergence state, commit count and Nexus node count, and a bounded event feed. Its source is in `ntsa_x/materializer.*`. The event stream passes only approved metadata scalar fields and has a separate bounded buffer; slow viewers cannot block the inference callback.

A live view is available when the demo is running. This **does not** connect to an existing Kindred Materializer installation or cloud service. It is a local adapter and browser renderer.

Start:

```bash
python -m unittest discover -s tests -v
python -m examples.demo_counter_orbit --watch
```

Open the printed `http://127.0.0.1:<port>/` URL on the **same computer**, then press Enter in the terminal to start the paired computation. Once done, press Enter again to shut down the server. Without `--watch`, it runs through automatically.

Keep your existing Nexus data with `--db PATH`; omit `--db` to use an ephemeral demonstration database. Never expose the loopback listener through a public proxy or host. There is no authentication; other processes on the same computer could read metadata while it is running.

## Files and API

- `ntsa_x/counter_orbit.py`: paired scheduler, projection, convergence and verified commit.
- `ntsa_x/nexus.py`: transactional persistent hash-linked multi-parent DAG.
- `ntsa_x/materializer.py`: localhost HTTP/SSE bridge.
- `ntsa_x/materializer.html`, `materializer.js`, `materializer.css`: live dashboard.
- `tests/test_nexus_counter_materializer.py`: persistence, tamper rejection, contradiction veto, dual budget, contextual recall, SSE and telemetry filtering.
- `examples/demo_counter_orbit.py`: watchable local workflow.

## Next scientific and engineering gates

Real independent evidence retrieval/verification, isolation of plugin code in worker processes, durable event-log replay, authorized cross-device visualization, secure backup/recovery, performance & electrical-energy measurement, higher-order convergence reuse, and reproducible comparison to baseline models.
