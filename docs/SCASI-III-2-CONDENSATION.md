# SCASI III.2 — Adaptive Condensation and Held-Out Transfer

Status: research implementation and isolated test execution. Not a trained foundation model or verified capability-per-parameter improvement.

## Modules
- `ntsa_x/scasi_iii_condensation.py`: exact evidence dedup and conflict reporting, reviewed-example affine skill induction, disjoint validation, versioned SQLite skill bank, native cognitive extension and matched inference evaluator.
- `tests/test_scasi_iii_condensation.py`: focused tests for source lineage, conflict handling, skill transfer, persisted revisions and native API.
- `examples/demo_scasi_iii2.py`: runnable offline deterministic skill transfer demonstration with no model weights.
- `examples/evaluate_scasi_iii2.py`: real COG-FU checkpoint evaluation requiring an operator-supplied checkpoint and two JSONL datasets.

## Commands

```bash
python -m unittest tests.test_scasi_iii tests.test_scasi_iii_native_integration tests.test_scasi_iii_condensation -v
python -m examples.demo_scasi_iii2
python -m examples.evaluate_scasi_iii2 --checkpoint /path/to/existing/checkpoint --examples /path/to/train-and-validation.jsonl --heldout /path/to/private-evaluation.jsonl --output evaluation.json
```

`SkillExample` JSONL fields: `skill_id`, `input_value`, `output_value`, `partition`, `source`, `reviewed`. `HeldoutCase` JSONL: `prompt`, `expected`, `group`. For each skill, at least 3 distinct training and 2 distinct independent validation inputs are required. Disjoint source labels and input values protect the validation boundary. Test prompts use `compute <skill_id> <integer>` with heldout inputs absent from all training/validation inputs.

The condeser only merges exact normalized claims, preserves source lineage, and flags multiple different claims for one topic as needing adjudication rather than claiming automatic contradiction resolution. `externally_attested` is an unverified caller label and is *not* independently validated. A SHA-256 lineage digest is a reproducible consistency check, not a cryptographic signature.

The native model evaluator reuses the same loaded COG-FU checkpoint and matched prompts for baseline and condensation-extension inference. It reports exact match and p50/p95, plus preparation compute, and does not claim a proven EIM. The local deterministic model fixture is only a test of the API. Constrained arithmetic rule induction is not evidence of open-ended self-improvement or general world knowledge.
