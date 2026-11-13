# K-ACT v0.1 — Adaptive Cognitive Transmission for NTSA-X

Status: **research prototype, not production serving, not a demonstrated efficiency improvement.**

## Architecture

- Existing `ntsa_x/native_model.py` remains unchanged.
- `KACTNativeOrbitModel` provides native byte-vocabulary logits with a separate typed `KACTState`.
- Each `KACTState.lanes[layer]` remains `[batch, 3, width]` for the original X/Y/Z streams; `KACTState.working[layer]` adds a distinct `[batch, width]` W memory.
- `KACTCell.orbit` is the original `OrbitCell`, and `initialize_from_native()` copies the source base model's weights into it without altering the source.
- `cross_lanes` is zero-initialized and produces trainable residual cross-lane coupling.
- Optional W memory uses gated recurrence; the memory input/output projection begins at 1% identity to permit gradients on the first training step.
- After a mandatory recurrent step, up to `max_refinements` extra recurrent steps can execute. A learned continuation head determines hard accept/skip.
- Training computes refinement branches densely and uses a **straight-through** hard gate to match the inference forward path. Training is not computationally sparse.
- At inference, refinement stops when all samples in the batch halt. Mixed batches may still incur dense refinement work; report real `dense_cell_calls` separately from per-sample logical updates.

## Native and K-ACT interfaces

```python
from ntsa_x.native_model import NativeConfig, NativeOrbitModel
from ntsa_x.kact import KACTConfig, KACTNativeOrbitModel

base = NativeOrbitModel(NativeConfig(width=96, layers=2))
kact = KACTNativeOrbitModel(KACTConfig(width=96, layers=2)).initialize_from_native(base)
# With compatible native token IDs, call kact.forward(tokens) -> (logits, KACTState).
```

K-ACT checkpoints are **not** interchangeable with native checkpoints; they include new weights and a new `working` state. Existing native checkpoint files and production serving remain unchanged. A production integration would require versioned checkpoint metadata, training and inference parity tests, and explicit service support for the K-ACT state object.

## Reproduction

Requirements: Python 3.11+ and PyTorch, CPU adequate for smoke test.

```bash
python -m unittest discover -s tests -p test_kact.py -v
python benchmark_kact.py --seed 113 --steps 40 --output kact_benchmark.json
python benchmark_kact.py --seed 213 --steps 100 --output kact_benchmark_seed213.json
```

### Benchmark definition

Two tasks share one stream format: `X` asks to recall the first of 12 bits; `Y` asks to copy the final bit. Each sample uses distinct `mode+12 bits` patterns, divided into 160 training and 64 held-out patterns. The model predicts one of two output byte IDs from the last latent position. No public dataset, language benchmark, or real service workloads are involved. Each configuration shares identical starting native weights, though its total/trainable parameter count and gradient cost differ.

Ablations: original native; forced refinement (no coupling/memory); adaptive without coupling/memory; adaptive + coupling; adaptive + coupling + W memory.

### Measured results / caveats

The first 40-step seed 113 run returned 85.94% held-out accuracy for both the native baseline and full K-ACT variant; K-ACT was slower. Initial 100-step results using *soft* training gates were poor due to a train/inference mismatch. The discrepancy was corrected by introducing straight-through hard gating. **Earlier soft-gate artifacts should not be used as final evidence.**

See `kact_benchmark.json` and `kact_benchmark_seed213_v2.json` for exact metrics and environment. One corrected seed reached 100% held-out accuracy for all three adaptive variants and for native baseline at 100 updates. This does not prove efficiency advantage: the adaptive variants had higher CPU latency and the baseline was already at 100% on that tiny test.

## Integration boundaries

- No PCK/MPO cross-process work was performed; there is no PCK execution receipt.
- SCASI fast/deep/meta recurrence is not yet integrated here. The current controller governs only per-layer extra refinements.
- No general long-term persistence across process restarts; W survives only in an in-memory `KACTState` supplied to subsequent chunks.
- No claims of 95–97% general-task accuracy, benchmark leadership, or energy savings.
- The next rigorous gate requires real tasks, diverse seeds, matched params and actual FLOPs, batch-size scaling, and a real inference latency profile.
