# NTSA-X — Krystal-inspired single-cycle counter-orbital phases

**Status: experimental feature branch. Not a benchmarked frontier model,
verified reasoning engine, or production trust boundary.**

## Purpose and preservation

One request, one recurrent decoding trajectory, with *ordered internal
phases*: **memory recall -> cognitive forward hypothesis + opposing
diffusion challenge -> fused generation -> internal convergence receipt**.
There is no automatic recursive second/third generation cycle. The phase
controller is optional; default COG-FU serving behavior, checkpoint format,
training functions, verified Nexus DAG, and user-visible APIs are unchanged.

The term **Krystal** is conceptual nomenclature for outward/challenging
paths sharing an origin. No supernatural, geometric, or physical mechanism
is asserted. This is not a new trained graph neural network.

## Actual implementation

- `KrystalSingleCycle.run` calls existing read-only `MemoryFabric.recall`
  once, includes bounded unverified source-labelled notes only when the
  model's context limit permits, and preserves the user's original request.
- The preexisting native `CogFuModel.unified_draft` encodes prompt context
  **once**, computing its causal recurrent proposal and diffusion
  denoiser's opposing proposal at each emitted byte. Both heads use the
  **same recurrent state** and fuse in log-probability space.
- The optional `phase_trace` captures actual selected and independently
  proposed byte IDs and confidence values without rerunning inference.
  The causal recurrent state then advances once for that byte.
- Typed **memory**, **cognition**, and **diffusion** graph views represent
  *observed execution provenance*, not separately-trained graph networks
  or independently verified semantic connections.
- The final report counts head disagreements. Agreement is not proof of
  factual accuracy or evidence. No unverified text can be committed into
  the verified Nexus ledger through this controller.
- No external APIs, credentials, autonomous writes, or network actions
  are introduced. Existing local trusted-model assumptions still apply.

## Usage with an existing trained COG-FU checkpoint

From the NTSA-X repository root with the existing native PyTorch environment:

```python
from ntsa_x.cog_fu_train import load_checkpoint
from ntsa_x.krystal_cycle import KrystalSingleCycle
from ntsa_x.memory_fabric import MemoryFabric

model = load_checkpoint(".local/cog-fu-v1", device="cpu")
with MemoryFabric(".local/ntsa-memory.sqlite", writable=False) as memory:
    cycle = KrystalSingleCycle(model, memory=memory)
    result = cycle.run("Explain recall-before-inference", max_new_tokens=48)
    print(result.text)
    print("Counter challenges:", result.head_challenges)
    print("Internal head agreement:", result.head_agreement_fraction)
    print("Externally verified:", result.externally_verified)
```

The paths above refer to the operator's **existing** checkpoint and
memory database; they are not created by this feature. Training data and
model competence are never fabricated. For a checkpoint smaller than
48 answer bytes, lower `max_new_tokens` to its actual `max_answer`.

## Validation

Run the *real model* checks (no synthetic service mocks):

```bash
python -m unittest discover -s tests -p "test_krystal_cycle.py" -v
python -m unittest discover -s tests -v
```

CI also tests the existing code on Windows and Linux. Regressions, failures,
and benchmark measurements determine whether this branch merits further
integration. Passing unit tests alone is not evidence of improved cognition.

## Next research gates

1. Measure baseline vs phased runtime, latency, throughput, and quality on
   identical model weights and held-out prompts.
2. Investigate sparse counter-head activation and calibration; **do not**
   claim an efficiency gain without profiling the actual denoiser.
3. Test whether learned graph-state exchange, independent evidence,
   and multi-head convergence outperform a single path.
4. Keep strict separation between *model agreement* and *external proof*.
