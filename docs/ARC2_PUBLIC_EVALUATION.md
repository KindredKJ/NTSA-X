# NTSA-X — Real ARC-AGI-2 public evaluation

**External dataset**: arcprize/ARC-AGI-2/data/evaluation, 120 public official ARC Prize puzzles. Data is cloned directly from the organizer by GitHub Actions and its source SHA recorded.

**Kindred evaluated system**: fixed-function geometry/recolor inducer. This is an additive NTSA-X research adapter, not pretrained weights or a Kindred¤ foundation model. Its two attempts per test input are built from observed training pairs, not answers. Control: two-attempt identity/flip-h policy using the same inputs, runner and attempt count. Both must be evaluated under a single frozen source commit before inspecting actual scores.

Run: `python -m ntsa_x.arc2_public_eval --dataset external-arc-agi-2/data/evaluation --output arc2-public-results.json`

Result labels: `LOCAL_EXECUTION_OFFICIAL_PUBLIC_DATASET_NOT_OFFICIAL_LEADERBOARD`; `official_private_score=false`. This is a **local score on a genuine external official dataset**, not organizer-approved leaderboard rank or ARC-AGI-3 performance.

## Evidence structure

GitHub workflow `arc2-public-eval.yml` records:
- Kindred source commit SHA; official ARC dataset commit SHA;
- SHA-256 of local policy, unit tests, and scored artifact;
- all 120 task IDs + task-file hashes, task-level fractional pass@2 scores for both policies;
- elapsed wall time, runtime/platform, attempts, human intervention count (zero during this workflow);
- original CI run ID and timestamp provided by GitHub.

The scoring criterion requires every test input's ground-truth output to match one of two guesses exactly; task score is mean over its test pairs. **This local evaluator implements ARC's published scoring rule but does not invoke ARC Prize's separate official `arc-agi-benchmarking` scoring package.** External third-party review of scoring parity remains a follow-up.

## Qualification path

1. Inspect the complete first-run report; **do not tune against this public evaluation set**, which would contaminate any comparison.
2. Create a separate development corpus from the 1000 official training tasks. Tune model architecture only there, freeze parameters and threshold choices.
3. Arrange an independent rerun on a second runner, preserving the same frozen commit/hashes, to establish repeatability.
4. Evaluate true native Kindred¤ model once a trained, exportable checkpoint is available. Compare native vs attached-model baselines on the same held-out tasks with matched hardware, wall clock/token/tool budgets, and multiple seeds; do not rebrand the deterministic adapter as a trained foundation model.
5. For ARC-AGI-3, first obtain organizer-issued scorecard using their interactive official SDK/API. Competing for the 2026 prize additionally requires Kaggle eligibility, offline agent execution and acceptance of the official license/rules by the original entry deadline.
6. Official/private ARC Prize rankings require organizer authorization and private evaluation. No source test may be called an official leaderboard score unless ARC Prize issued it.

## Reliability, commercialization and peer review

For task autonomy, build a separate end-to-end blinded operational environment with mandatory task completion receipts, full model/tool/human intervention traces, failures, timeouts and cost. For commerce, link to Kindred Cloud isolated staging plus fulfilled real customer orders and reconciled Stripe live money; no fake paid signals. For scientific credibility, submit a reproducible methodology and external evaluator package to independent researchers. Do not claim peer review until a journal or independent reviewers have completed review.

This PR neither charges a customer nor merges/deploys existing Kindred services.
