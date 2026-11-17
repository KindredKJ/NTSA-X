# NTSA-X: Interventional Belief Memory (experimental)

Date: 2026-11-02. Research status: **unverified against external tasks**.

## Purpose

Study a measurable mechanism that records a sequence of actions and observed
outcomes, maintains uncertainty-calibrated estimates for each context/action,
and chooses future exploratory interventions by sampling from its posterior.

This is an additive, dependency-free Python module, not a competing model
runtime. It does not change the COG-FU / NativeOrbitModel checkpoint format,
the existing MemoryFabric schema, tool privileges, or live inference.

## Actual implementation

- `ntsa_x.interventional_memory.InterventionalBeliefMemory`: bounded
  Beta-Bernoulli posterior for each context/action, Thompson sampling and
  hash-linked observation records.
- `SyntheticContextualWorld`: seeded, hidden action-dependent Bernoulli
  environments with independent context schedules and consistent
  action-conditional reward streams.
- `benchmark`: compares an adaptive policy with a seeded random-action
  baseline under identical contextual schedule, horizon, warmup and task
  distributions across multiple seeds. Reports each run, SHA-256 of the
  hidden task specification, mean rewards, paired mean gain and approximate
  95% confidence interval across task seeds.
- Pure standard-library dependencies; Windows and Linux runners run the
  module's `unittest` tests in the existing NTSA-X CI.

Run:

```powershell
python -m unittest discover -s tests -p 'test_interventional_memory.py' -v
python -m ntsa_x.interventional_memory --seeds 12 --steps 800 --warmup 300 --output .\interventional-memory-report.json
```

This is **synthetic contextual bandit learning**, not structural causal
discovery, learned language/vision understanding, a trained foundation model,
a self-improving AGI system or an official benchmark result. The pseudorandom
environment is constructed for controlled ablation and reproducibility; the
controller does not receive its hidden reward probabilities.

The stored hash chain makes after-the-fact changes detectable if a root
digest was independently retained, but is not a digital signature or
external observation proof.

## What to measure next

1. Compare intervention-based selection to best static policy, greedy policy,
   no-memory policy, and policy using a shuffled context index. Match
   intervention budgets, seeds and observed context sequence.
2. Add a nonstationary-regime shift to test whether persistent beliefs
   can recover without negative transfer. Record regret and calibration.
3. Add actual transition-state prediction and held-out structural
   interventions; do not call the current policy a causal DAG model.
4. Integrate predictions into the **existing** unified cognition path
   using a typed, bounded and explicitly authorized native extension.
5. Compare paired attached-model vs native-model ablations on third-party
   evaluation tasks. Report failures and incomplete model training.
6. Obtain official ARC-AGI-3 developer scorecards using ARC's harness,
   and separate these entirely from synthetic ablations.

## Evidence labels

Every benchmark JSON includes:
`verification_status=INTERNAL_SYNTHETIC_RESEARCH_ONLY`,
`is_external_benchmark=false`, `is_agi_proof=false`, and
`is_frontier_rank_one_proof=false`. A positive delta over the random
baseline is progress on that narrow task, not evidence of worldwide
AI leadership.

Project-wide frontier ranking contract and ARC-AGI-3 pathway live in:
https://github.com/KindredKJ/kindred-brainstem/pull/14

No model weights, live customer data, benchmarks with protected answer keys,
secrets, production endpoints or cloud services are touched.
