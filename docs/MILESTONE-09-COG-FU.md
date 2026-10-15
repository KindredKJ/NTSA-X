# Milestone 09 — COG-FU (Cognitive-Diffusion Hybrid)

**COG-FU** couples the existing trainable, non-Transformer NTSA-X recurrent
cortex with a discrete absorbing-mask diffusion response denoiser. It is ONE
jointly optimized network, not a teacher call followed by a separate generator.

This milestone implements a functioning, unpretrained research model. Neither
frontier performance nor humanlike cognition is established. It cannot be
advertised as a production-grade general assistant without extensive
pretraining, independent benchmarks and controls.

## Mechanism

1. **Recall before inference remains first.** The existing reviewed exact
   context-matched recall path in the live gateway is unchanged.
2. **Cognitive state:** each prompt passes through NativeOrbitModel; its
   recurrent X/Y/Z latent lanes encode conversation context.
3. **Diffusion:** a second trainable branch receives answer slots corrupted with
   a unique MASK token, their positions, a noise level, and *the same X/Y/Z
   cognitive state*. Local bidirectional convolutional blocks iteratively
   predict missing byte tokens, including EOS.
4. **Joint supervision:** masked-token loss is computed ONLY at hidden answer
   slots; a separate causal assistant next-byte loss trains the recurrent
   cortex. The sum backpropagates into both diffusion parameters and cognitive
   recurrence.
5. **Sampling:** initialize a bounded window to MASK; over 1-32 refinement
   iterations reveal highest-confidence predictions until all slots are
   filled. The cognitive prompt is encoded once, not once per iteration.
   End the output at the first EOS. Research decoding has no adaptive length
   modeling yet.
6. **Verification boundary:** neither loss, stochastic selection nor recurrent
   state makes generated content verified evidence. The Nexus DAG remains
   separate and deny-by-default.

### What kind of diffusion?

COG-FU uses an **absorbing-mask, discrete token corruption and reconstruction**
objective (closer to masked discrete diffusion / iterative unmasking than to
Gaussian image diffusion). It is not trained with a diffusion SDE or a
continuous-time physical process. The X/Y/Z channels are learned *latent state*
axes, not independent proof of external causality.

## Architecture

    user / conversation
             |
    approved exact recall? ---> yes: return reviewed context-matched response
             |
             no
             v
    trainable recurrent NTSA-X cortex ------> causal token loss
             |
             +--> [X, Y, Z] cognitive state (gradient-connected)
                         |
                masked byte denoiser <------- noise level + draft positions
                         |
                iterative confidence-based revealing
                         |
                final UTF-8 response (unverified)

COG-FU only loads PyTorch, repository-local NTSA-X model code and its own
weights. No Transformer package, pretrained teacher, or third-party inference
API is invoked.

## Run on Windows

Use Python 3.12+, a PyTorch 2.x installation appropriate for the machine,
and reviewed, appropriately licensed Milestone 07 examples.

    py -3.12 -m pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.6,<3"
    py -3.12 -m unittest discover -s tests -p test_cog_fu.py -v
    py -3.12 -m ntsa_x.learning --train .local/train.jsonl --eval .local/eval.jsonl

Create a hybrid model initialized from existing Milestone 08 cognitive
weights, if such a checkpoint has already been trained:

    py -3.12 -m ntsa_x.cog_fu_train --train .local/train.jsonl --eval .local/eval.jsonl --output .local/cog-fu-v1 --initialize-cortex .local/native-v1 --width 96 --layers 2 --max-context 1024 --max-answer 256 --epochs 4 --batch-size 2 --device cpu

The width, layers, context window and dropout MUST match the earlier native
checkpoint exactly. Omit --initialize-cortex to train both branches from
scratch. The existing teacher-to-student LoRA system is not required.

Continue learning from a prior COG-FU checkpoint:

    py -3.12 -m ntsa_x.cog_fu_train --train .local/new-train.jsonl --eval .local/new-eval.jsonl --replay .local/rehearsal.jsonl --initialize-from .local/cog-fu-v1 --output .local/cog-fu-v2 --width 96 --layers 2 --max-context 1024 --max-answer 256 --epochs 4 --batch-size 2 --device cpu

A new checkpoint is staged atomically, with previous-weights SHA-256 lineage
in its report. If held-out combined loss never improves, best selection keeps
the initialization rather than exporting degraded weights. This is NOT an
independent guarantee of reasoning quality or freedom from forgetting.

Serve COG-FU locally:

    py -3.12 -m ntsa_x.cog_fu_server --checkpoint .local/cog-fu-v1 --model-id ntsa-cog-fu-v1 --port 8081 --max-new-tokens 128

In a separate terminal, connect to the existing recall-first web UI without
any teacher inference engine:

    py -3.12 -m ntsa_x.live_gateway --student-model ntsa-cog-fu-v1 --student-max-tokens 128 --student-endpoint http://127.0.0.1:8081/v1/chat/completions --use-student

Open http://127.0.0.1:8765. Only reviewed exact recall or COG-FU native
inference is used in this mode; no fallback to teacher models.

## Validation

- Same-training-pass nonzero gradients into both the native recurrent routing
  matrices and the denoiser/contex-conditioning modules.
- Full-mask reconstruction has no leaked target text in input.
- Fixed-seed iterative refinement is deterministic and emits only bytes;
  generation stops at EOS if predicted.
- Tiny repeated-sample smoke test shows reduction of training loss.
- Native cortex warm-start, new COG-FU checkpoint, and continued COG-FU
  training with rehearsal all load/save correctly.
- Train and replay user prompts must not appear in held-out evaluation.
- Actual local SSE chat and health routes function with no teacher loaded.
- Existing no-dependency NTSA-X tests and Windows/Linux PyTorch CI remain
  healthy.

## Limitations and next benchmarks

Diffusion operates over a fixed maximum answer window: it cannot yet adapt
the window efficiently to semantic length. Current serving only sends response
bytes **after the denoising steps complete**; this is not fully incremental
low-latency duplex speech. Discrete unmasking quality, confidence calibration,
reasoning effectiveness, resource consumption and scalability require
independent empirical evaluation.

The native recurrent cortex does not implement a verified reasoning oracle,
and neither diffusion scores nor the checkpoint report should be treated as
external proof. A small approved corpus is unsuitable for general language
competence. Next research gates are dynamic-length decoding, adaptive-step
allocation, bidirectional reasoning verification, memory-conditioned
denoising, contamination-resistant evals, and CUDA-optimized recurrence.
