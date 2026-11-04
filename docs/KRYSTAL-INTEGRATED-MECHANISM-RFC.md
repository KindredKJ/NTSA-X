# KRYSTAL — Integrated Mechanism RFC

**Kindred Labs / NTSA-X** — 2026-10-17  
**Status:** Design-only draft; implementation awaits the founder's explicit mark.  
**Scope:** Model architecture and integration plan, not verified benchmark claims.

## Decision

KRYSTAL is **one multi-mechanism**, not three independent engines stacked in series and not a detached cognition processor. Cognition, memory, and diffusion contribute different functions to dynamically constructed, linked graph representations of a problem and its solution. The original non-Transformer, UTF-8 byte-native NTSA-X model and prior checkpoints must remain usable.

Graph *views* in the current experimental branch are **execution provenance**, not yet trained graph-state processing. This distinction must be maintained in logs, documentation, and benchmark claims.

## Intended single-cycle execution contract

1. **Zero point / input**: establish the problem, request context, trusted/untrusted source boundaries, checkpoint identity, and bounded inference budget.
2. **Driver**: activate and schedule work in response to need, novelty, uncertainty, and progress; it is an activation/controller function, **not a fourth graph**.
3. **Generative diffusion**: propose and transform native internal representations for cognition to examine; do not equate these proposals with validated answers.
4. **Cognition**: work over the diffusion contributions to form relationships, hypotheses, intermediate operations and evidence requirements, adjusting shared working state.
5. **Memory weaving**: relevant recall is available throughout, not permanently pinned to an initial stage. Source provenance and trust labels accompany recalled information. Recall may be skipped when useless or too costly.
6. **Tools, as needed**: the driver/cognition route only through registered, permission-scoped tools with typed inputs/outputs, timeout, receipts, and clear provenance. No unconstrained model-authored shell/network writes.
7. **Return diffusion + convergence**: refine the same evolving representation; supported information should persist and unsupported or irrelevant influence may decay. Unresolved contradictions remain marked as uncertainty rather than silently disappearing.
8. **Bounded continuation or answer**: stop based on measurable criteria (budget, uncertainty, incremental improvement) or continue within a finite limit. Emit answer plus auditable lineage/metrics. Preserve original user request and checkpoint weights unless an explicitly authorized training job runs.

The terms **zero point**, **counter-orbital**, and **convergence** describe implementable computational dynamics to be specified and experimentally tested, not demonstrated physical or quantum processes.

## Actual repository inventory as of RFC

- `ntsa_x/cog_fu.py`: recurrent cortex plus masked-byte diffusion with fused log-probability decoding; output decisions influence subsequent recurrent state.
- `ntsa_x/krystal_cycle.py` on PR #23: one memory recall before generation, per-byte head comparisons, three typed graph **trace views**, and agreement diagnostics. These views do not yet create live state or execute a learned convergence refinement.
- `ntsa_x/unified_cognition.py`: recall and explicitly enabled extensions contribute bounded observations to a native fused decoding path, but are not a multi-step graph-state learner.
- `ntsa_x/plugins.py` and `ntsa_x/plugin_worker.py`: versioned bounded fixed-function scalar operations; research controls, not adversarial sandboxing.
- `ntsa_x/agent_workbench.py`: audited and capability-gated developer tools, separate from unrestricted model autonomy.
- `ntsa_x/cog_fu_server.py`: supports selectable decoding and opt-in extensions; current Krystal controller is not the active public serving path by default.

## Proposed engineering increments (NOT AUTHORIZED TO START YET)

**I. State interface and invariants.** Introduce a typed `KrystalWorkingState` with representation identifiers, directed provenance edges, uncertainty estimates, budgets, and immutable user inputs. State updates from cognition, memory, and diffusion must be measurable, bounded and replayable.

**II. Interleaved mechanism.** Implement trainable/ablatable diffusion-to-cognition-to-diffusion exchanges with memory lookup at decision points. Avoid replacing learned model behavior with arbitrary deterministic 'reasoning' rules. Test gradient flow through trainable exchange parameters.

**III. Driver and tool contracts.** Activate optional compute based on thresholds learned or calibrated against measured error/cost; issue typed tool requests through a deny-by-default capability registry; carry tool source receipts into the state. Keep high-impact side effects behind explicit approval.

**IV. Convergence.** Use a measurable learned/refined gating mechanism that retains corroboration, dampens irrelevant state, and preserves uncertainty; bound iteration count and quantify information lost. Never equate agreement between native heads with factual proof.

**V. Training.** Preserve current checkpoints; version initialization/resumption provenance, dataset hashes, optimizer state, seeds, architecture settings, validation splits, and measured losses. Do not train on benchmark answers used for evaluation.

**VI. Qualification.** On identical checkpoint/hardware/eval protocol, compare native recurrence, present COG-FU, existing PR #23 Krystal, and integrated proposed mechanism. Use unseen reasoning tasks plus loss, calibration, decoding quality, runtime, peak memory, tool accuracy, and per-graph contribution ablations. Do not claim frontier performance before independent tests.

## Explicit constraints and hold

- Retain non-Transformer and conventional tokenizer-free byte-native pathways.
- Preserve main branch, original weights, and current functional endpoints.
- No fabricated metrics, training claims, tools, or graph cognition.
- Design-only documentation may be reviewed now; **start implementation only on founder's mark**.
- Existing checkpoint training is a distinct permitted operational activity if separately authorized; it does not mean this RFC has been implemented.
