# Auxiliary Recall-Before-Inference — Verification v0.3

**Status: Not yet end-to-end verified. Do not merge as a proven learned cognitive improvement.**

Base model: KindredKJ/NTSA-X, `feat/auxiliary-pdc-foundation`, commit `a317863526a02f3ae1a630b2e633005abd67f09e`. Feature work is additive in draft PR #39; existing native X/Y/Z/PDC model and checkpoint file formats are not replaced.

## What changed

1. `ntsa_x/inferencing_skills.py` now optionally runs a genuine no-guidance decode through the **same** pre-outcome Auxiliary projection and decoder, with identical seed and prompt. It writes that paired prediction to the inference receipt. This adds validation compute; record it separately.
2. `SkillLedger.observe` never credits a caller-supplied baseline string as independent evidence. It requires a pre-outcome paired receipt plus independently validated outcome (`outcome_verified=True`). Confounded/multi-skill cases remain provisional. The flag represents host trust in an external gold/outcome verifier; it is not cryptographic proof.
3. `ntsa_x/neural_inferencing.py` stages inputs before outcomes, excludes unverified updates from gradient labels, and does not allow a trained-but-uncertified head to influence inference.
4. `ntsa_x/neural_calibration.py` selects a threshold on a disjoint calibration set and requires a further disjoint blind set with sufficient positive/negative support, at least 80% useful selection and at most 2% 95%-Wilson upper false-application bound. Opposing labels with identical features cannot activate the head. Real datasets must establish independent labels and disjointness; source training loss alone cannot.
5. Test cases cover caller-supplied baseline rejection, missing independent outcome verification, high-confidence deceptive near-matches, collision abstention, deterministic replay, snapshot preservation and gradient training.

## Executed local evidence

CPU-only PyTorch 2.10.0, three seeds 213/311/701. The 19-test local suite yielded **17 passed and 2 skipped**, the skips requiring native Auxiliary model code. **No trained Auxiliary weights were accessible.** GitHub Actions test jobs failed before executing steps, so CI adds no model evidence.

Independent synthetic calibration and blind sets: each 60 positives + 300 negatives per seed. Original synthetic distribution: 100% useful selection and 0% observed false application in all three blind sets (false-application 95% Wilson upper 1.264%). **High-confidence misleading near-matches:** 60/60 such near-matches were incorrectly selected per seed at the original threshold, or 20% of all negatives in that scenario. Calibration refused activation on every adversarial seed.

These are NeuralSkillHead relevance-classification measurements, **not** Auxiliary held-out task success, recurrent-learning efficacy, or causal improvement. The calibration gate rejects an unproven model; it does not cure the neural classifier's inability to distinguish information-theoretically identical inputs.

## Native verification requirements still missing

- Accessible real `ntsa_x/auxiliary.py` and `ntsa_x/cog_fu.py` checkout on the execution machine.
- A genuinely trained Auxiliary `config.json` and `weights.pt` from the pinned architecture.
- A trained `skill-head.pt` bound to the SHA-256 of that base checkpoint, independently calibrated.
- A source-addressed ledger with trace IDs, evidence and verified outcomes.
- Independent demonstration/calibration/blind task splits and several seed repetitions.

Local test command: `python -m unittest discover -s tests -p 'test_*.py' -v`.
Local calibration audit: `python -m examples.benchmark_calibration_v03 --output .local/calibration-audit --seeds 213 311 701`.
The standalone audit bundle includes a strict native verifier and 224 bit-operation fixtures. It exits nonzero rather than create a mock when source/weights/ledger are missing.

## Promotion bar

Compare no-recall, raw trace, and calibrated learned-head predictions from the same checkpoint and task order; record ground-truth exact-transition accuracy, false application, missed recall, paired counterfactual attribution, real delayed recall/retention, calibration, per-family p50/p95 latency and additional compute. Keep all seeds, configurations and per-inference evidence. Only a reproducible held-out improvement supports the claim **end-to-end learned inferencing skill**.
