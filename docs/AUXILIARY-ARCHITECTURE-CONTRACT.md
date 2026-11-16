# Auxiliary Architecture Contract

**Status:** source-of-truth design contract for the Auxiliary / NTSA-X PDC
research track. This contract supersedes interpretations in the initial
Milestone 17 prototype that put a projection denoiser downstream of the
existing recurrent encoder.

## Locked architecture

Auxiliary is one native NTSA-X cognitive system. Its intelligence core scans
raw experience directly through its native recurrent X/Y/Z state transitions;
there is no detached semantic encoder before cognition. The projection-diffusion
workspace (PDC) is integrated with that scanner as the input membrane and
internal work environment: it receives the scanner's live experience-bearing
state, refines/resolves projections, and shares the conductive state through
which cognition operates. Perception reads those projections. A separate
perception model or independent assistant must not translate the experience
first.

The same trainable recurrent X/Y/Z cognition attends over the projections,
recalls relevant persistent states before inference, reasons across live and
recalled evidence, updates memory proposals, and applies learned algebraic
operators. X/Y/Z are soft specialization priors: each axis may process live
input, memory, alternatives, synthesis, and temporal lineage. Memory storage,
scheduling, and externally authorized execution remain framework services;
the framework is not a second cognitive planner.

Byte-oriented external I/O may remain for compatibility. That boundary does
not establish token-free internal cognition. PDC projections are cognitive
state, not visualizations or copies of weights. A diffusion residual alone is
not evidence that the cognitive state or task has converged.

## NTSA-X capability carry-forward inventory

“Carry forward” means preserve the capability's purpose and measurable
contract, not blindly embed every existing Python subsystem inside the neural
network. Cognitive selection and state updates should become learned model
behavior where indicated. Durable storage, deterministic verification,
resource enforcement, and authorized side effects remain host responsibilities.

| NTSA-X capability or trait | Disposition | Current source of truth and Auxiliary status | Carry-forward requirement |
|---|---|---|---|
| Transformer-independent recurrent stream; persistent X/Y/Z state | Integrate and evolve | `native_model.py`, `cog_fu.py`; inherited as COG-FU cortex, but current PDC scaffold still uses recurrence before/alongside PDC | Preserve state tuple compatibility and recurrence; teach the same network to attend over PDC projections. Keep parity and continuation tests. |
| X/Y/Z soft axes and 3D orbit coordinates | Integrate and evolve | `docs/ARCHITECTURE.md`, `orbit.py`, native cells; historical axis semantics are not yet demonstrated by learned lanes | Preserve soft specialization priors, cross-axis access, coordinate deltas and introspection; test each axis on live input, memory, synthesis and temporal tasks. |
| Byte-level boundary and adaptive resolution lineage | Integrate/evolve resolution; preserve exact I/O contract | `native_model.py`, `cog_fu.py`, architecture lineage; byte IDs persist, dynamic bit/byte/structure/concept granularity is absent | Retain exact input/output boundary behavior; add learned resolution transitions within PDC without claiming tokenless internal cognition until measured. |
| Direct native input scanner, PDC membrane, and conductive workspace | Integrate and evolve | Added in Auxiliary v0.2 as `scan_experience`; shares the recurrent X/Y/Z cells and sends their live state into PDC. Further behavioral evidence remains pending. | Core reads raw input directly; scanner state and PDC projections stay within one trainable intelligence core. No detached semantic encoder or second model. |
| Recall-before-compute; episodic/semantic/procedural/relational memory | Integrate/evolve native retrieval; preserve storage interface | `memory_fabric.py`, `unified_cognition.py`; host memory exists, unified runtime currently retrieves/assembles text before model inference; no learned latent recall | Preserve tiering, bounded retrieval and session scope. Add trainable queries, compatibility scoring, latent reinjection and exact-source links; measure recall, interference, stale-memory rejection and cost. |
| Nexus, OMCF, hash-linked lineage, multi-parent state, verified commits | Preserve as host invariant | `nexus.py`, `core.py`, `orbit.py`; separate durable host subsystem, not part of Auxiliary model/checkpoint | Keep provenance, integrity, parent lineage and promotion rules. Neural outputs remain proposals; external verifier controls verified commits. |
| Evidence-based convergence and contradiction handling | Preserve verifier as host invariant; integrate learned critique | `convergence.py`, `counter_orbit.py`; host verifier is not coupled to learned PDC convergence; PDC residual is only numeric | Preserve distinct state/semantic/causal/constraint/evidence/delta checks. Report projection residual and task correctness separately; never use residual as proof. |
| Paired forward/reverse counter-orbits | Integrate and evolve after parity baseline | `counter_orbit.py`; host-side deterministic engine, absent from Auxiliary learned trajectory | Add model-native opposing-trajectory/counterexample capability. Keep external contradiction/evidence verification independent. Measure added quality and latency. |
| SCAC bounded adaptive compute and binary optional-operator gates | Integrate and evolve; preserve host hard caps | `docs/ARCHITECTURE.md`, `orbit.py`; prototype has fixed PDC bounds/residual stop, not learned expected-value control | Learn whether another cognition/PDC/operator cycle is useful. Measure physical FLOPs, quality-per-cost and actual skipped work. |
| K-ACT adaptive recurrence, W working memory, cross-lane coupling | Adapt only after ablation | `kact.py`, `docs/KACT-V0.1.md`; separate candidate model, not inherited by Auxiliary | Compare integration or compatible state adapter against base recurrence and PDC ablations. Preserve zero-impact baseline, halt metrics and physical-compute accounting. |
| Operators, typed actions, bounded execution and failure recovery | Integrate learned proposals; preserve host authorization | `plugins.py`, `persistent_workers.py`, `agent_workbench.py`, `agent_mission.py`, `unified_cognition.py`; host extensions execute, model does not learn general tool choice/execution | Train internal typed operation proposals and learned algebraic composition; preserve host capability grants, audit, resettable evaluation and recovery scoring. |
| SCASI cycles, anchors, identity and goal return | Integrate and evolve after state parity/recall | NTSA-X continuum blueprint; current model has no native anchor/identity module | Add versioned identity/mission/objective/exploration state as trainable conditioning. Measure goal return, identity consistency, branch value and resource use. |
| Informational reuse / convergence cache | Adapt only after ablation; preserve verified reuse contract | `reuse.py`, `indexed_orbit.py`; host exact/context-scoped verified reuse, not model-native learning | Preserve scoped cache validity and re-verification. Compare against fresh inference at equal correctness; report avoided FLOPs. |
| Web/evidence intake and provenance | Preserve as host evidence adapter | `web_recall.py`, `unified_cognition.py`; host retrieval can supply sourced context, no native multimodal/browser perception in Auxiliary | Preserve source attribution and stale/contradictory evidence handling; route evidence through PDC as input data, never as trusted instructions. |
| Runtime, streams, checkpointing, monitoring, durable state | Preserve as host invariant; adapt interfaces as needed | `native_server.py`, `cog_fu_server.py`, `live_gateway.py`, `events.py`, `nexus.py`; Auxiliary has a minimal local checkpoint only | Preserve compatible APIs and versioned checkpoint lineage. Add typed PDC/cognition telemetry with bounded metadata and resume/parity tests. |
| Data rights, held-out evaluation, checkpoint immutability | Preserve as host/training invariant | `learning.py`, `cog_fu_train.py`, training tests; some protections are inherited in prototype trainer | Preserve rights checks, prompt/document disjointness, no overwrite, replay provenance and independent holdouts across Auxiliary training. |

This inventory is the architecture-accounting checklist. A capability marked
“design considered” or “separate host subsystem” is **not** considered carried
into Auxiliary behavior until an adapter/module exists and a test demonstrates
its contract. Every future architecture change should update this table and
add a corresponding ablation or integration test.

## Inheritance and evolution rules

“Equal or greater” is a behavioral bar, not a naming or diagram bar. Auxiliary
may change the implementation, state shape, or scheduling when the change is
shown to preserve the old contract and improve at least one target dimension
without hiding a regression in quality, stability, retention, provenance, or
resource cost.

Every inherited capability receives exactly one disposition:

1. **Integrate and evolve:** capability belongs in Auxiliary's learned
   cognition; keep its useful behavior and demonstrate an improvement.
2. **Preserve as a host invariant:** capability is deterministic storage,
   provenance, verification, authorization, scheduling, or transport; keep the
   established interface and make Auxiliary consume or produce its typed data.
3. **Adapt only after ablation:** capability may be useful, but is not assumed
   core. Compare it against a simpler Auxiliary control and retain it only when
   it improves quality-per-resource or reliability.
4. **Defer or exclude from the cognitive core:** capability has no direct use
   in the locked PDC cognition, duplicates another mechanism, or adds cost
   without measured value. Exclusion from Auxiliary does not delete or break
   the existing NTSA-X host subsystem or its callers.

No item is “inherited” merely because a Python class is imported. Completion
requires the named behavioral test and a baseline comparison. For any legacy
capability Auxiliary cannot preserve, record the exact regression, affected
use case, reason, and replacement before accepting that version.

### Required equivalence and upgrade gates

| Capability | Required Auxiliary form | Minimum acceptance evidence |
|---|---|---|
| Native recurrent stream and X/Y/Z | Same continuing trainable X/Y/Z cognition reads evolving PDC projections; lanes remain cross-functional with soft priors | With Auxiliary additions disabled, checkpoint/state API parity within recorded numerical tolerance. With additions enabled, held-out task score is non-inferior to the pinned native baseline; report per-axis ablations and full recurrent state continuity. |
| Direct scanner and PDC projection perception | Raw experience is read by the integrated core scanner; its state enters the PDC workspace and projections are the perceptual substrate and conductive state for cognition | Trace proves there is no detached encoder or legacy `cortex.forward` call before PDC in Auxiliary inference; ablation of projection-to-cognition path changes measured task behavior; report task quality at every PDC refinement cycle. |
| External I/O and resolution | Preserve exact bytes, order, boundaries, and recoverable source references while learning internal resolution changes | Exact round-trip tests for supported boundary formats; controlled noise and length sweeps; document which internal transitions are continuous latent, discrete, or decoded. No token-free claim from byte I/O alone. |
| Recall-before-inference and memory | Query relevant persistent state before expensive inference; fuse selected latent state into ongoing cognition while retaining exact source links | Delayed recall, distractor, stale/contradictory memory and interference tests. At matched compute, recall accuracy and false-recall rate are no worse than the declared host baseline; report memory bytes and retrieval latency. |
| X/Y/Z synthesis and counter-orbits | Learn complementary alternatives, forward/reverse prediction, abstraction/resolution, and temporal/lineage synthesis within the same model | Held-out compositional and counterexample tasks beat or match single-trajectory control at equal FLOPs; independent evidence verification still governs factual promotion. |
| SCAC / adaptive compute | A learned value controller allocates PDC, recurrent, recall, and operator work under a hard host budget | Against fixed-small and fixed-large controls, report actual FLOPs, wall latency, quality-versus-compute, physical skipped work, and confidence intervals. Logical step counts alone do not pass. |
| Self-scaling capacity | Capacity grows, reuses, or prunes in response to measured task complexity, then stabilizes/contracts when demand falls | Fixed-small, fixed-large, no-growth, no-pruning, and linearized controls; active versus allocated capacity; matched total FLOPs; held-out accuracy, retention, state stability, and memory/energy curves. Parameter growth alone fails. |
| Memory/provenance/convergence/commit | Learned cognition proposes relevance and state changes; host Nexus/OMCF and independent verification preserve source, integrity, contradictions, and verified-commit rules | Corrupt, stale, missing-parent and contradictory evidence tests must retain their existing outcomes. No learned residual/confidence or internal agreement can bypass external checks. |
| Learned operators and execution | Neural cognition selects/composes typed internal operations; host remains sole authority for external capabilities and effects | Resettable environment evaluation with independent end-state inspection, allowed-tool and side-effect checks, injected failures, recovery metrics, and calls-per-success. Registered scalar demos alone are insufficient. |
| Persistent checkpoints, serving, telemetry and data rights | Versioned model/state continuity, compatible serving interfaces, bounded receipts, rights-aware training and disjoint evaluation | Save/restore/resume parity; restart/replay tests; bounded telemetry; dataset rights and train/eval/replay disjointness tests; no overwrite of source checkpoints. |

### Deliberate host-only or optional capabilities

- **Keep outside learned cognition:** authentication, capability grants,
  secrets, deterministic external execution, independent evidence checks,
  cryptographic hashes, Nexus transactions, durable queues and hard resource
  ceilings. Auxiliary can emit typed requests; it cannot grant itself
  authority.
- **Optional candidate, requiring matched ablation:** K-ACT's W state and
  halting controller, particular SCASI cycle decomposition, exact convergence
  cache/index, specialized operator schedulers, or a specific attention/state
  kernel. Integrate a compatible idea when it improves the locked model; do
  not duplicate mechanisms solely to preserve names.
- **Defer from the core:** browser/desktop providers, network protocols,
  dashboards, QR visualization/transport and blockchain anchoring. They may
  remain adapters or product integrations, but are not required for native
  projection cognition. Do not spend model complexity on them before core
  cognition passes its tests.

The carry-forward table and these gates are binding for the next Auxiliary
implementation and its test plan. A later feature may change a disposition,
but only with a documented use case, comparison, and measured reason.

## Implementation status correction

The first Milestone 17 commit (`17d2001`) was a bounded PDC denoising scaffold
that pooled embeddings before PDC and reused the legacy COG-FU answer path.
Auxiliary v0.2 replaces that pooled-condition handoff with `scan_experience`:
the core's own byte embedding and recurrent X/Y/Z cells scan input directly,
and their live state seeds/conditions PDC. The resolved projection supplies
the cached cognitive context to the integrated response denoiser, avoiding a
second recurrent input-encoding pass during ordinary Auxiliary drafting.
This is the first executable direct-scanner slice, not evidence of learned
general cognition, full PDC convergence, endogenous capacity growth, or a
latency improvement. Existing v0.1 state dictionaries remain loadable; their
interpretation follows the v0.2 direct-scanner forward path.

Do not build a separate PDC service or second model and call that Auxiliary.
“Parallel PDC” means concurrent state updates within the one model: PDC
continues refining the shared projection workspace while X/Y/Z cognition
processes available projections. Synchronization points are explicit and
measured. A host may schedule kernels and enforce budgets, but may not make
cognitive decisions on behalf of the neural system.

## Required implementation sequence

1. **Parity and traceability:** pin a real trained NTSA-X checkpoint; preserve
   byte I/O and baseline behavior; prove save/restore/continuation parity.
2. **Native membrane:** keep raw experience scanning inside the model's
   recurrent intelligence core; route its live X/Y/Z state into the integrated
   PDC workspace. Expose typed projection state and receipts. No detached
   semantic encoder or duplicate recurrent input scan in the inference path.
3. **Projection perception and cognition:** train recurrent X/Y/Z attention
   and state updates to read the evolving projections directly. Keep external
   byte decoding only at the output boundary.
4. **Recall-before-inference:** add trainable retrieval queries, relevance
   gates, bounded state-cell recall, provenance-preserving fusion, and explicit
   stale/contradictory memory tests.
5. **Learned operators and stability:** add trainable algebraic composition,
   state norm controls, perturbation/recovery tasks, and value-based compute
   allocation. Residual-based stopping must also track task quality.
6. **Parallel execution ablation:** compare serial PDC then cognition,
   synchronous coupled PDC/cognition, overlapped PDC and X/Y/Z updates with
   synchronization only at required state barriers, and no-PDC control.
   All PDC arms remain inside the same learned architecture.
7. **Self-scaling:** only after quality and stability baselines, implement a
   learned capacity controller that can add, reuse, or prune projection
   channels/units within fixed resource caps. Measure active and allocated
   capacity separately; more parameters alone do not count as useful scaling.
8. **Skill and execution transfer:** evaluate held-out compositions and
   resettable tool environments. The model proposes typed operations; the
   host enforces capabilities and independently scores final environment
   state.

## Parallel experiment

Use one pinned checkpoint, identical task stream, initialization, seed set,
training FLOPs, precision, hardware, and output budget. Include at least five
seeds when feasible. Evaluate each task at every projection cycle and report
both quality and cost.

| Arm | PDC and cognition schedule | Purpose |
|---|---|---|
| No-PDC | X/Y/Z uses the preserved baseline path | Baseline quality and latency |
| Serial | PDC resolves projections, then X/Y/Z consumes them | Sequential cost reference |
| Synchronous | PDC and X/Y/Z exchange state at each defined cycle | Coupled quality reference |
| Overlapped | PDC and X/Y/Z advance concurrently; synchronize at explicit dependency barriers | Test latency reduction without changing model ownership |

Current overlap prototype seeds PDC from the direct scanner's first short
prefix (up to 16 symbols), runs all but the final PDC update while the same
recurrent scanner consumes the remaining input, then merges at a barrier and
performs the last update. It preserves one model and an equal PDC update count,
but the early PDC branch sees a prefix snapshot; measure task quality and
projection change alongside latency. Current CPU measurements show no speedup.

Primary measurements: end-task success; quality-versus-projection-cycle; p50/p95
latency; throughput; training and inference FLOPs; peak activation and durable
memory; energy per verified success when measurable; residual; state norms;
perturbation recovery; and barrier/queue wait time. Report confidence intervals
per seed. A latency gain only counts if quality is non-inferior within a
pre-registered 2-point margin, recall and retention remain within 2 points,
and the apparent gain is not caused by omitted work or extra hardware.

## Endogenous scaling claim

Compare adaptive-capacity Auxiliary with fixed-small, fixed-large,
no-growth, no-pruning, and linearized controls on seeded compositional tasks.
Vary rule count (2–64), compositional depth, sequence length, delay,
distractors, interference, and revisits independently. Match total training
FLOPs for the primary comparison and test on unseen rule compositions and
surface forms.

The claim passes only when active capacity rises with measured complexity,
stabilizes or contracts on repeated/easier phases, improves held-out learning or
accuracy per equal FLOPs, stays within pre-registered retention and stability
limits, and transfers to independent skill and resettable execution tests. The
existing evaluation protocol remains binding for metrics and reporting:
`Auxiliary_Research_Evaluation_Protocol_v1.md`.
