# NTSA-X NativeOrbitModel versus causal Transformer: controlled pilot

**Date:** November 2, 2026. **Status:** exploratory original research, not an official ARC competition score, a publication, or proof of AGI.

## Research question

Does the existing Kindred native recurrent X/Y/Z cell provide any measurable predictive benefit over (a) a causal Transformer architecture and (b) the *same trained native weights* when the recurrent state is removed?

Both primary models are trained from scratch on byte-level next-token prediction of genuine public **ARC-AGI-2 training-task JSON**, not puzzle-answer prediction. This experiment does not measure ARC problem-solving ability.

## Method and matched controls

- Official source: https://github.com/arcprize/ARC-AGI-2 ; dataset commit `f3283f727488ad98fe575ea6a5ac981e4a188e49`.
- Source files used: `data/training` only (1,000 tasks). Sorted 850 task files for training and remaining 150 files held out for local validation. None of the 120 `data/evaluation` files are used.
- Byte vocabulary 262; identical sampled train and validation windows for both models; 30 AdamW optimizer steps, learning rate 0.001, batch size 2, context length 48, dropout 0; random seeds 11 and 29; single CPU thread. Each model sees **2,880 byte inputs**, 2,820 next-byte target positions, per seed.
- Native: `NativeOrbitModel`, one layer, width 32, 44,457 trainable parameters; no Transformer attention.
- Reference: from-scratch causal `TransformerEncoderLayer`, 45,406 parameters, 4 attention heads, explicit upper-triangular future-token mask.
- Trainable parameter gap: **2.135%**; identical tokens/optimizer/hardware protocol; NOT an equal-FLOP or equal-compute-time experiment.
- No-memory ablation: evaluate the *same trained native weights*, but reset X/Y/Z state before each byte. This is an evaluation-time causal intervention on state, not a separately trained competitive model.
- Quality: held-out cross-entropy reported in bits per byte (**lower is better**). Report each seed, not solely averaged performance.
- Runtime: wall-clock CPU training seconds measured for both; a single run on a hosted CPU does not establish production inference efficiency.

## Actual initial GitHub Actions result

Run: https://github.com/KindredKJ/NTSA-X/actions/runs/41520672985

| Metric | Native recurrent | Native no-memory ablation | Causal Transformer |
| --- | ---: | ---: | ---: |
| Mean held-out bits/byte (two seeds) | 4.20965 | 5.08371 | 4.41203 |
| Seed 11 held-out bits/byte | 4.32768 | 5.14236 | 5.05993 |
| Seed 29 held-out bits/byte | 4.09163 | 5.02507 | 3.76412 |
| Mean training time per seed (seconds) | ~0.829 | N/A (evaluation-only) | ~0.084 |

**Interpretation:** The native model has a lower arithmetic-mean held-out loss across these two seeds, while the Transformer **wins on seed 29**, and the native takes ~9.9x longer per equal training-token budget. Thus these two observations do **not** support a general native-model advantage or compute efficiency claim.

Resetting learned X/Y/Z state harms the native model in these evaluations; this establishes sensitivity to recurrence on this narrow workload, not that the recurrence is a superior causal world model.

## Reproduce, expand, verify

```powershell
git clone https://github.com/KindredKJ/NTSA-X.git
Set-Location NTSA-X
git checkout db682723357d886fe4f68c3a5c9531049459a6a6
python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.6,<3"
git clone https://github.com/arcprize/ARC-AGI-2.git official-arc-agi-2
git -C official-arc-agi-2 checkout f3283f727488ad98fe575ea6a5ac981e4a188e49
python -m unittest discover -s tests -p test_matched_architecture_study.py -v
python -m ntsa_x.matched_architecture_study --training-dir official-arc-agi-2/data/training --output native-transformer-study.json --steps 30
```

Source revision above includes cross-platform experimental workflow and unchanged neural algorithm. The executable produces a JSON report with training/validation canonical digests, parameter comparison, per-seed metrics and measurement limitations.

Next: pre-register a larger multi-seed run, add equal wall-clock and measured hardware FLOP/energy budget comparisons, and repeat on independent hosts. Then move to task-solving benchmarks with genuine trained checkpoints under actual ARC-AGI policies.

## Evidence / publication boundaries

No peer-reviewed acceptance has been obtained. This report is an internal exploratory experiment run against an external public training corpus; it is not ARC Prize certified, not a frontier generalization leaderboard, and not evidence that Kindred¤ is generally stronger than Transformers.

No existing production, model checkpoints, customer accounts, or revenue services are changed by this PR.
