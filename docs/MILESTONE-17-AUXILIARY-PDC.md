# Milestone 17 — Auxiliary: NTSA-X Projection-Native Diffusive Cognition

> **Status correction:** this milestone is the first software scaffold, not a
> faithful implementation of the locked Auxiliary architecture. The scaffold
> conditions PDC from the existing recurrent encoder. The locked design makes
> PDC the experience input membrane and has perception/cognition read its
> projections. See [the architecture contract](AUXILIARY-ARCHITECTURE-CONTRACT.md)
> before using these results as Auxiliary evidence.

**Auxiliary** is the proposed model name. **NTSA-X PDC** names its recurrent,
projection-native diffusive cognition architecture. This is an additive,
research-stage implementation layered on the current NTSA-X COG-FU code. It
does not replace existing checkpoints or establish frontier capability.

## What is implemented

- A trainable continuous latent denoiser resolves three learned X/Y/Z
  projections from a noise state conditioned on the current encoded experience.
- The projection refinement path is differentiable and trains jointly with the
  existing recurrent cortex and byte response denoiser.
- The generated projection is fed into the same model's response path and
  unified cognition runtime. Existing registered memory and tool extensions
  remain host-managed; the neural model does not directly execute arbitrary
  tools.
- A bounded adaptive reverse-diffusion rollout stops after its configured
  minimum when successive clean-projection estimates meet a relative-residual
  tolerance, or at the hard maximum cycle count.
- Auxiliary has a versioned checkpoint format, COG-FU weight-transfer path,
  continued training from an Auxiliary checkpoint, replay samples, held-out
  evaluation, and a task-category benchmark runner.

The training target for the projection denoiser is the recurrent cortex's
final X/Y/Z state for the input prefix. The denoiser learns to recover this
state from corrupted projections conditioned on the experience embedding.
At serving time, it starts from noise and iteratively predicts a clean
projection. This is a learned latent denoising process; the rollout is not a
proof of reasoning or truth.

## Scaling protocol

Let `T` be input bytes, `L` recurrent layers, `d` hidden width, `C` PDC
projection cycles, `A` answer-window bytes, and `D` discrete response-denoising
steps.

| Resource or work | Scaling in this implementation |
|---|---|
| Recurrent input processing | `O(T · L · d²)` compute; recurrent hidden state is `O(L · d)` per sample, independent of accumulated history length once streamed. |
| Projection-native denoising | Approximately `64 · C · d²` floating-point operations for dense PDC layers per sample, plus normalization and activation work. `C` is adaptively selected up to a fixed maximum. |
| Response denoising | Approximately `O(D · A · d²)` for the local-convolution response denoiser. `A` and `D` are separately bounded. |
| Inference weights | Fixed by checkpoint configuration: roughly `O(L · d² + d²)`; increasing `C` does not add weights. |
| PDC working activations | `O(d)` per sample at inference because current and previous projections are reused; training retains unrolled activations and scales approximately as `O(C · d)`. |
| Persistent history | Managed by the existing memory fabric and Nexus lineage, not expanded automatically by PDC. Memory growth must be measured and quota-controlled separately. |

At the defaults (`d=96`, `C<=12`), the PDC dense layers contribute about
7.1 million multiply/add FLOPs per sample at the full cycle budget, before
normalization, the recurrent cortex, and response decoding. Early convergence
reduces PDC compute. This is an estimate from layer shapes, not a hardware
benchmark. Actual latency, energy, batch efficiency, and score-versus-cost
must be measured on named devices.

**Capacity boundary:** v0.1 adapts inference effort, not parameter count. It
does not yet grow or prune neural modules, increase hidden width, or add
unbounded memory. Width and depth are selected when building/training a
checkpoint. That deliberate distinction keeps “adaptive compute” separate
from the still-unproven claim of endogenous structural self-growth.

## Training

Use an explicit-rights, document-disjoint dataset in the established JSONL
format (`{"messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}`).
No remote corpus, teacher model, or automatic promotion is used.

```bash
python -m ntsa_x.auxiliary_train \
  --train .local/train.jsonl \
  --eval .local/eval.jsonl \
  --output .local/auxiliary-v0.1 \
  --width 96 --layers 2 --max-context 2048 --max-answer 256 \
  --pdc-max-cycles 12 --pdc-min-cycles 2 --pdc-tolerance 0.015 \
  --epochs 4 --batch-size 2 --device auto
```

Transfer existing COG-FU cortex and response-denoiser weights while training
the new PDC branch:

```bash
python -m ntsa_x.auxiliary_train \
  --train .local/train.jsonl --eval .local/eval.jsonl \
  --initialize-cog-fu .local/cog-fu-v1 \
  --output .local/auxiliary-v0.1 --device auto
```

Continue only into a new output directory; existing checkpoints are never
overwritten. Held-out prompts must not occur in train or replay data.

## Evaluation

Run structural/training tests:

```bash
python -m unittest discover -s tests -p test_auxiliary.py -v
```

Benchmark a JSONL file whose rows contain exactly `category`, `messages`, and
`expected`. Categories can include `general_skill`, `reasoning`, `memory`,
`planning`, and `execution_intent`:

```bash
python -m examples.benchmark_auxiliary \
  --checkpoint .local/auxiliary-v0.1 \
  --cases .local/auxiliary-eval.jsonl \
  --output .local/auxiliary-benchmark.json \
  --device auto
```

The report gives exact-match accuracy per category, p50/p95 latency, PDC cycle
counts, projection residual, parameter count, and CUDA peak allocation when
available. The `execution_intent` category scores generated intent text only;
execution itself must be tested in resettable environments through the
existing fixed-capability host adapters. Do not interpret exact-match results
on a user-provided dataset as general skill competence.

Compare serial and overlapped direct-scan/PDC scheduling on the same fixed
cycle budget:

```bash
python -m examples.benchmark_auxiliary \
  --checkpoint .local/auxiliary-v0.1 \
  --cases .local/auxiliary-eval.jsonl \
  --output .local/auxiliary-schedule-comparison.json \
  --max-new-tokens 128 --pdc-cycles 12 --compare-schedules
```

The overlap arm scans a short prefix, refines that live X/Y/Z projection while
the same core scans the remaining experience, then merges and performs the
final PDC update at the synchronization barrier. It is an in-model schedule
ablation with fixed cycle count. It can change projections because its PDC
branch starts from a prefix snapshot; report quality with latency. The
2026-11-15 CPU smoke result is in
[the parallel PDC benchmark report](AUXILIARY-PARALLEL-PDC-RESULTS-2026-11-15.md).

Required comparisons for a research claim: adaptive versus fixed cycle
budgets; COG-FU versus Auxiliary transferred from the same checkpoint; PDC
disabled, PDC fixed-cycle, and PDC adaptive ablations; multiple random seeds;
equal training FLOPs; separate quality, retention, latency, energy, and memory
reporting. A residual threshold alone is not a task-success criterion.

## Limitations

- No Auxiliary foundation checkpoint or large-scale training run is included.
- v0.1 does not dynamically add/prune modules or adapt width/depth.
- The PDC residual is an internal numerical convergence signal, not semantic
  correctness, confidence calibration, or external evidence.
- Input experience is represented by the existing byte-stream embedding and
  recurrent cortex. Multimodal native adapters are future work.
- General skills, autonomous tool selection, open-vocabulary speech, and
  frontier-level performance require separate training and external evaluation.
- Existing safety and execution boundaries remain at the host layer.
