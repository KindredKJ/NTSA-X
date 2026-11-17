# Auxiliary Neural Inferencing Skill Sets — v0.2

## Status

Research-stage neural applicability training. The synthetic held-out selection test succeeded at larger demonstration counts, but checkpoint-backed Auxiliary task accuracy has NOT been established.

Foundation revision: a317863526a02f3ae1a630b2e633005abd67f09e on feat/auxiliary-pdc-foundation.
Development branch: feat/auxiliary-recall-before-inference; draft PR #39.

## Single-model dataflow

Current input -> Auxiliary native X/Y/Z scanner and PDC -> addressable trace recall -> explicit condition/operation/provenance/contradiction checks -> gradient-trained applicability score -> bounded native projection guidance -> existing projection_readout -> existing COG-FU recurrent/denoising decoder -> inference receipt -> independently observed outcome -> per-trace attribution -> eligible gradient-training pairs.

No second cognition and no stored answer replay. The neural component is registered inside the actual AuxiliaryModel parameter tree under inferencing_skill_head. Base model weights are preserved; only head parameters are optimized by the supplied training routine.

Untrained fallback uses the prior explicit relevance scorer; it does not apply random neural coefficients. Provisional/confounded outcomes and multi-skill attribution do not create positive or negative head labels. Operation and condition metadata are currently supplied by the caller, not learned by Auxiliary.

## Source files

- ntsa_x/neural_inferencing.py: neural head, optional registered integration, evidence pair extraction, gradient training, checkpoint save/restore.
- ntsa_x/inferencing_skills.py: versioned evidence ledger, ordered inference events and optional pre-outcome feature staging hook.
- examples/train_neural_skill_head.py: outcome-trained head CLI.
- examples/benchmark_neural_inferencing.py: synthetic seeded selection benchmark.
- tests/test_neural_inferencing.py: gradients, hard negatives, deterministic reruns, attributable evidence, checkpoint roundtrip, and native-path integration.

## Run

python -m unittest discover -s tests -p test_neural_inferencing.py -v
python -m examples.benchmark_neural_inferencing --output .local/neural-skills --seeds 213 311 701 --epochs 20

With a real version-compatible Auxiliary checkpoint and evidence ledger containing independently attributed positive AND negative cases:

python -m examples.train_neural_skill_head --checkpoint .local/auxiliary-checkpoint --ledger .local/evidence.sqlite --output .local/neural-skill-checkpoint --seed 213 --epochs 24

The trainer refuses to overwrite an existing output directory. The skill checkpoint sidecar is bound to the SHA-256 of base weights.pt. It does not replace or overwrite the foundation checkpoint.

## Measured locally — synthetic state selection ONLY

Three seeds, demonstration groups 24/120/360, intervention delays 0/64/256, 60 held-out groups/seed, six candidate types, 20 epochs; 1,849 trainable parameters at width 8/hidden 48.

With 24 demonstration groups: learned useful recall 66.1%, learned false application 0%; raw scorer useful recall 100%, raw false application approximately 0.56%.
With 120 groups: learned useful recall 100%, false application 0%.
With 360 groups: learned useful recall 100%, false application 0%.

This demonstrates gradient learning of the synthetic applicability labels but shows a LOW-DATA REGRESSION with 24 demonstrations. It does not measure actual Auxiliary response or transition accuracy. The standalone local tests had two native tests skipped, because Auxiliary source and a trained checkpoint were not mounted. No claim of native learned efficacy is made.

## Remaining dependencies

- A real trained Auxiliary checkpoint plus independent held-out transition tasks;
- run the model-path integration and full existing Auxiliary test suite;
- matched-compute no-recall/raw/consolidated comparisons, causal per-skill ablations;
- retention/forgetting, actual neural FLOPs, calibrated applicability, and latency with real states;
- transaction-safe production ledger concurrency and privacy controls.

The SQLite ledger contains experience-derived states and is not encrypted.