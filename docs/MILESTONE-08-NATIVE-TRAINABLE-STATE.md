# Milestone 08 — Native trainable NTSA-X recurrent state

## Scope

This is a native, randomly initialized, trainable non-Transformer NTSA-X research model. It is not the earlier fine-tuned Transformer student, a pretrained conversational model, or a proven frontier model. It needs substantial training data and compute before generating useful text.

The implementation contains **no self-attention, Transformer blocks, external teacher API, or pretrained weights**.

- Byte-level UTF-8 vocabulary: 256 bytes plus six control tokens for padding, BOS, EOS and role boundaries.
- Three learnable X/Y/Z latent state lanes per layer, each with input-dependent gates and routing; these channels are **not** the established state/evidence axes of the verified orbit scheduler.
- Cross-axis recurrent mixing, trainable signed candidate state, convex bounded update and per-token causal logits.
- Masked next-byte supervision: only the assistant continuation and stop token contribute to loss; prompt bytes remain context.
- Direct gradient backpropagation through native recurrent state-update parameters.
- Fixed-size recurrent state during generation and linear number of token steps, but this Python reference is **not optimized for throughput**.
- Deterministic seed, AdamW, gradient clipping, held-out loss/perplexity, checkpoint saving and safe weights-only loading.
- Local teacher-free SSE model server compatible with the existing live gateway and recall-first routing.

The existing verified Nexus/recall kernel, worker factory and Transformer-LoRA student code are unchanged.

## Recurrence equations

For each token x, current three-lane state h, and model layer:

    route = 3 * softmax(W_r [normalize(x); mean(h)])
    gate = sigmoid(W_g x + U_g normalize(flatten(h)))
    candidate = tanh(W_c x + U_c normalize(flatten(h)))
    alpha = clamp(route * gate, max=1)
    h_next = h + alpha * (candidate - h)

All matrices are learned end-to-end with gradient descent. Initial update biases create three different time scales but remain trainable. A token only updates states from tokens it has already consumed.

## Windows PowerShell execution

From the repository root, install a CPU-capable modern PyTorch version and run tests:

    py -3.12 -m pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.6,<3"
    py -3.12 -m unittest discover -s tests -p test_native_model.py -v

For NVIDIA GPU training, use official PyTorch CUDA installation instructions matching your local driver instead of the CPU install command.

Export consent-approved, appropriately licensed examples using the existing learning corpus:

    py -3.12 -m ntsa_x.learning --train .local/train.jsonl --eval .local/eval.jsonl

Train native recurrence parameters (no model weights downloaded and no teacher inference requests):

    py -3.12 -m ntsa_x.native_train --train .local/train.jsonl --eval .local/eval.jsonl --output .local/native-v1 --width 96 --layers 2 --max-context 1024 --epochs 4 --batch-size 4 --lr 0.001 --device cpu

Output: config.json, weights.pt, report.json in the new model folder. The trainer will not overwrite a previous run.

In a separate terminal, serve the resulting native recurrent model on localhost:

    py -3.12 -m ntsa_x.native_server --checkpoint .local/native-v1 --model-id ntsa-native-v1 --port 8081

In another terminal, connect the existing browser gateway in **student-only** mode:

    py -3.12 -m ntsa_x.live_gateway --student-model ntsa-native-v1 --student-endpoint http://127.0.0.1:8081/v1/chat/completions --use-student

Then visit http://127.0.0.1:8765. Exact approved recall happens before native inference. **No teacher engine needs to be installed for student-only serving.**

## Continue native learning across training waves

For each new batch, export a reviewed train/eval dataset and initialize training
from a prior checkpoint rather than resetting the native recurrent weights.
Optional rehearsal examples should also be approved and licensed, and the
combined train/replay prompts must remain disjoint from the evaluation set.

    py -3.12 -m ntsa_x.native_train --train .local/new-train.jsonl --eval .local/new-eval.jsonl --replay .local/rehearsal.jsonl --initialize-from .local/native-v1 --output .local/native-v2 --width 96 --layers 2 --max-context 1024 --epochs 4 --batch-size 4 --lr 0.001 --device cpu

The new checkpoint retains a SHA-256 lineage pointer to prior native weights
in its report. A baseline held-out evaluation is performed before updating:
if all training epochs perform worse, the new checkpoint saves the prior weights
rather than the degraded candidate. This is a limited held-out loss gate, **not**
a guarantee against catastrophic forgetting or reduced task-specific quality.
Review independent benchmarks before promoting the new checkpoint to serving.

## Validation gates

1. State-update router, gate, candidate and cross-lane parameters receive nonzero gradients.
2. Causality check: a complete pass and a chunked continuation produce numerically equivalent logits and final state.
3. Teacher-free verification: the native trainer and model-serving paths never construct an external model client.
4. Dataset holdout: prompt-disjoint train/eval; separately review semantic contamination and training data licensing before claiming generalization.
5. Training: record held-out metrics, optimizer steps, hardware, parameter count and checkpoint contents.
6. Serving: profile time-to-first-byte, throughput, memory, longer-context behavior and robustness under load.
7. Promotion: retain rollback snapshots and do not automatically promote unreviewed model updates to live service.

## Limitations

This is **real native weight learning**, not cache recall, but it is also a small-from-scratch research implementation. Neither useful conversational fluency nor superiority to a Transformer has been demonstrated. A few dozen approved examples are insufficient to produce a competent general-purpose assistant. The Python token loop is not GPU-optimized. The verified Nexus DAG does not accept native generated text as independent evidence; outputs remain unverified.

Future work: recall-conditioned latent state, native associative memory, streaming training with truncated BPTT, parallel/fused recurrent kernels, auxiliary teacher-supervised reasoning traces when licensing permits, independent held-out reasoning benchmarks and safe checkpoint promotion. The local model server is only for a trusted machine, not unauthenticated public deployment.
