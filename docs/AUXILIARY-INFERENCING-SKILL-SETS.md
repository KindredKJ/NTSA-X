# Auxiliary Recall-Before-Inference / Inferencing Skill Sets v0.1

**Source:** \`KindredKJ/NTSA-X\`, branch \`feat/auxiliary-pdc-foundation\`
at \`a317863526a02f3ae1a630b2e633005abd67f09e\`.

**Status:** Integration adapter and evidence-ledger research. No trained Auxiliary
skill checkpoint, no measured held-out accuracy improvement, and no learned
neural applicability head. The contextual Beta(1,1) posterior is an explicit
online statistical reliability estimate, not a model-weight update.

## Architecture map

\`\`\`
Existing AuxiliaryModel.scan_experience / resolve_projections
  -> X/Y/Z projection (same recurrent cortex)
  -> stable evidence trace candidate recall
  -> operation/condition/architecture/attribution compatibility
  -> structural + latent fit, contradiction checks
  -> clipped bounded delta in the SAME X/Y/Z projection
  -> existing projection_readout
  -> existing CogFuModel.unified_draft (same response head)
  -> append prediction / IDs / fit / confidence / PDC receipt
  -> independent observed outcome and matched baseline comparison
  -> provisional | contextual positive | contextual negative update
\`\`\`

The operation and condition metadata are typed boundary observations for
applicability; they are not alternate response-generation instructions.
Retrieval itself is explicit host logic; no new neural model is instantiated.

### Data schema (version 1)

\`ExperienceTrace\` stores stable hash ID, architecture, 3xwidth PDC
state, observed transition vector, operation/context, action, outcome, evidence,
provenance, event sequence, confidence, skill identifier, related and
counterexample references. The SQLite ledger stores versioned event records,
immutable source traces and per-skill/per-context evidence counters.
Snapshot/restore uses SQLite online backup and preserves IDs and chronology.

Evidence is not a skill summary. Skill reliability is a contextual posterior;
the source trace remains addressable. Output-specific reinforcement requires
an independent ground-truth outcome and an unassisted counterfactual prediction.
If both arms succeed/fail, outcome is confounded, or multiple skills contribute,
updates remain provisional. Repeated \`observe\` calls for one inference are
rejected. Future outcomes are never accepted as inputs to \`infer\`.

### Bounded guidance

Eligible traces must match operation, all stored conditions, model architecture
and state width, and must have attributable original evidence. The
applicability score combines exact structural compatibility with bounded
latent cosine similarity and evidence reliability. Competing contradictory
traces from identical prior situations are disqualified. Guidance is a
clipped perturbation (max 12% state norm by default) to the original PDC
projection, followed by the existing Auxiliary readout and unified denoiser.
No trace can bypass current inference to supply a cached output.

This is a measured *integration interface*, **not** proof that the state
deltas correspond to transferable skills. Learning useful latent deltas,
training a relevance head, confidence calibration and causal contribution
ablations require a trained checkpoint and independent held-out evaluation.

### Commands

\`\`\`bash
python -m unittest discover -s tests -p 'test_inferencing_skills.py' -v
python -m unittest discover -s tests -v
python -m examples.benchmark_inferencing_skills --output .local/rbi-results --seeds 213 311 701
\`\`\`

The benchmark intentionally evaluates synthetic retrieval/applicability
mechanics only. It produces \`aggregate.json\` and per-seed JSONL inferences;
\`neural_exact_transition_accuracy\` is **null**. Delay is represented
by 0/100/500 intervening inference events. Demonstration source states are
separate from query states. It covers operation mismatch, condition mismatch,
held-out compositions, late recall, and no-recall control.
Do not compare its oracle labels to model outputs; the oracle is never a solver
used by Auxiliary inference.

### Required milestone before an effectiveness claim

1. Supply a versioned trained Auxiliary checkpoint and independent
   benchmark dataset, including source-rights and disjoint demo/test splits.
2. Match checkpoint, seeds, PDC cycles, inference steps and generation
   budgets across off/raw/consolidated arms; measure full cost including recall.
3. Train and ablate applicability and delta operators *in the existing
   recurrent X/Y/Z model*, without creating a second model.
4. Run delayed/contradictory/held-out compositions and retention across seeds,
   with ground-truth outcome attribution and calibration.
5. Publish per-case accuracy, recall gain, false/missed recall, interference,
   model FLOPs/latency, bytes, attribution calibration and regression analysis.

### Known limitations

- No trained Auxiliary foundation weights are present in the inspected repo.
- The current operation tags are provided by the caller; autonomous native
  operation recognition is not demonstrated.
- Stored transition vectors require trusted, measured source capture.
- A probabilistic reliability counter does not constitute learned neural skill.
- The synthetic benchmark tests selection, not skill-guided neural correctness.
- No production integration or deployment is implied by the existence of code.
