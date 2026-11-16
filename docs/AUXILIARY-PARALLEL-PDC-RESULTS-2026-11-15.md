# Auxiliary parallel PDC benchmark — 2026-11-15

## Question and implementation

Does overlapping projection refinement with the direct-scanning recurrent core
reduce latency compared with serial scanning followed by PDC refinement?

The overlap arm remains one `AuxiliaryModel`. The core scans a short input
prefix; PDC refines that live X/Y/Z state in a worker while the same core scans
the remaining input. At the dependency barrier, PDC state is fused with the
final scan state and receives its last refinement. Both arms execute the same
fixed number of PDC updates. This is a schedule ablation, not a separate PDC
service. It deliberately uses a prefix snapshot; the changed conditioning is a
possible source of quality loss and is measured rather than assumed away.

Implementation is exposed as `pdc_schedule="serial"|"overlap"` on
`AuxiliaryModel.draft` and `unified_draft`. The benchmark runner supports
`--compare-schedules` to score both schedules on identical cases and seeds.

## Environment

- Python 3.12.14
- PyTorch 2.15.0+cpu
- CPU only; CUDA unavailable
- PyTorch intra-op threads: 1
- Single-example inference

## Paired quality and end-to-end latency

The model was trained for six epochs on 75 generated single-digit addition
examples and evaluated on 25 disjoint operand pairs. It had width 32, one
recurrent layer, one diffusion layer, two response denoising steps, and four
fixed PDC updates. It had 115,664 trainable parameters. The synthetic task is
only a plumbing check, not a representative reasoning benchmark.

Held-out joint loss decreased from 11.781 to 2.582. Exact-match accuracy was
8% for both schedules. Latency includes direct input scanning, PDC, and
generation of up to three output bytes.

| Schedule | Exact match | p50 latency | p95 latency | Mean PDC updates | Mean final residual |
|---|---:|---:|---:|---:|---:|
| Serial | 8% | 3.32 ms | 3.88 ms | 4 | 0.0185 |
| Overlap | 8% | 4.92 ms | 6.23 ms | 4 | 0.2298 |

Observed p50 speed ratio, serial/overlap: **0.674**. A value below 1 means
overlap was slower. Exact match tied at a low rate; the residual increased
substantially in the overlap arm. These observations do not establish
non-inferior cognition.

## Projection-resolution latency sweep

This second sweep used randomly initialized models and measured only
`resolve_projections`, so it measures schedule cost and not task quality. Each
row used 20 timed runs after a warm-up, with identical seeds and cycle budgets.

| Width | Input bytes | PDC updates | Serial p50 | Overlap p50 | Serial/overlap p50 ratio | Serial p95 | Overlap p95 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 32 | 8 | 6.46 ms | 9.94 ms | 0.649 | 6.80 ms | 10.88 ms |
| 32 | 256 | 8 | 39.40 ms | 43.15 ms | 0.913 | 43.45 ms | 45.75 ms |
| 96 | 32 | 8 | 9.28 ms | 12.09 ms | 0.767 | 10.33 ms | 13.96 ms |
| 96 | 256 | 8 | 55.64 ms | 63.98 ms | 0.870 | 69.82 ms | 76.78 ms |
| 96 | 1024 | 12 | 218.94 ms | 224.26 ms | 0.976 | 228.30 ms | 254.56 ms |

Overlap was slower in all five CPU configurations. At width 96 and 1,024 input
bytes, p50 was close (about 2.4% slower) but p95 was about 11.5% slower. This
CPU implementation therefore does **not** show a latency reduction. Thread
dispatch, shared CPU resources, merge work, and prefix-conditioned refinement
can outweigh the work hidden behind scanning.

## Interpretation and next test

- Runtime behavior and bounded output were exercised; the Auxiliary unit tests
  pass with PyTorch installed (13/13); the full repository suite passes
  (160/160). Python compilation and `git diff --check` also pass.
- The synthetic task demonstrated a lower held-out training objective, but 8%
  exact match is weak performance and both schedules tied. This is not evidence
  of useful general skill.
- CPU overlap did not pay off. Do not promote it as the default schedule.
- CUDA overlap remains untested. A GPU run must capture device, stream behavior,
  synchronization waits, allocated memory, energy if available, and p50/p95
  across a much larger trained checkpoint and long-input tasks.
- The stronger comparison needs multiple training seeds, broader held-out task
  families, confidence intervals, and a fixed-versus-overlap quality margin.
  Projection residual must remain a separate diagnostic from task correctness.
