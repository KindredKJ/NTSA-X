# NTSA-X — Nexus Transforming Stream Architecture

Status: research architecture specification, not a demonstrated replacement for Transformers.

## Purpose
NTSA-X explores a transformer-independent recurrent, stateful stream machine whose operators are dynamically selected. Attention, MLP, state-space, graph, and external plugin operations are callable operators, not assumed to be the governing architecture.

## Research lineage
| Track | Hypothesis |
|---|---|
| NTSA-01 Token-Stream Transformer | Persistent streaming state, recall-before-compute, conditional attention |
| NTSA-02 Byte-to-Byte Transformer | Tokenizer-free bytes and adaptive segmentation |
| NTSA-03 Nexus-Byte Transformer | Byte substrate with emergent structure and persistent relational Nexus graph |
| NTSA-04 Byte/Bit-Stream Transformer | Resolution changes across bit, byte, patch, structure, concept, graph |
| NTSA-05 Transforming Byte-Scaler | Dynamic granularity, parameters, recurrence, depth, memory, precision, attention and compute |
| NTSA-06 Chain-Graph / Graph-to-Graph | Temporal graph transition models for agent and blockchain state |
| NTSA-X | Nexus Transforming Stream Machine synthesizing these lines |

## Core primitives
1. **Orbit**: bounded recurrent traversal of computational state.
2. **Block**: typed observation, state delta, or evidence unit.
3. **Chain**: hash-linked lineage of transitions.
4. **Nexus**: graph of relationships, reusable solutions, and provenance.
5. **Recall**: retrieval of relevant, verified prior work before expensive inference.
6. **Convergence**: evidence-based agreement across state, constraints, and opposing trajectories.
7. **Commit**: append-only persistence of verified, reusable state transitions.

## Execution cycle
Observe → Recall → Connect → Compute → Converge → Verify → Commit.

Modes:
- Known: retrieve verified result, check applicability and provenance.
- Partially known: retrieve, adapt, then verify.
- Novel: compute within budget, verify, then optionally commit.

Never equate a recalled record, a confidence score, or agreement between internal paths with external truth.

## Super-Cyclitic Adaptive Computation (SCAC)
The orbital scheduler chooses how many and which state transitions are needed. Easy or previously verified tasks may require very few transitions; hard tasks may need many. An orbit budget is bounded by latency, energy, and compute constraints. A closure decision terminates further optional computation only when verification passes.

## Three-axis orbit model
Represent computational position as q=(x,y,z):
- X: horizontal duality (prediction/reconstruction, hypothesis/falsification).
- Y: abstraction resolution (bit ↔ byte ↔ structure ↔ concept ↔ graph/world).
- Z: temporal lineage (present ↔ historical state).
Scheduler actions Δq may move on individual axes or diagonally, through Nexus junctions.

## Counter-Orbital Computation
Paired forward/reverse trajectories may share a compute drive: predict/reconstruct, cause/effect-to-cause, encode/decode, abstract/resolve, compose/decompose, and provenance/forecast. A common-space convergence metric can prioritize verification but cannot substitute for external evidence.

Conceptual recurrence:
z⁺ₖ₊₁ = F(z⁺ₖ,N,M)
z⁻ₖ₊₁ = R(z⁻ₖ,N,M)

## Binary Control Spine
A binary gate gᵢ∈{0,1} determines whether an optional operator runs. Internal reasoning may remain continuous or probabilistic. Closure cₖ and a separate commit authorization gate must not be conflated. When gᵢ=0, skip execution entirely.

Conceptual budget: E_total = E_router + ΣgᵢEᵢ.

## Verification and convergence
Verification examines state consistency, semantic applicability, causal support, constraints, external evidence, unresolved contradictions, and the proposed state delta. Failures retain useful diagnostic traces without promoting them to verified truth. Require explicit thresholds and bounded retries.

## Memory
Tiers: immediate, active orbit, episodic, semantic Nexus, immutable archive, and anchored provenance/identity. Verified computation paths can be promoted into higher-order reusable convergence nodes while retaining original evidence and proof links.

## Orbital Memory Chain Fabric (OMCF)
Append-only, hash-linked blocks; optional multi-parent Merkle DAG; parent references, content hash, confidence, utility, verification metadata, and state delta. Illustrative block bᵢ=(xᵢ,Pᵢ,sᵢ,cᵢ,uᵢ,vᵢ). Use canonical serialization, deterministic hashes, and explicit parent validation. External blockchain anchoring is optional; do not impose consensus or proof-of-work overhead on inference.

## Computational Energy Recovery
Means informational reuse: saved paths, failures, shortcuts, cached verified results, and indices. It does not recover consumed electrical energy. Claims about latency, joules, or throughput require instrumentation and measurements.

## Safety and external effects
Deny-by-default external actions, explicit authorization, provenance, resource limits, isolated plugins, auditable decisions, and human approval where required. The architecture is a research specification until runnable benchmarks substantiate performance claims.
