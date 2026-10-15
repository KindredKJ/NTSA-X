# Milestone 07 — Recall-first teacher-to-student learning

## Objective

Gradually remove dependence on the **teacher inference model**, not on a
compute runtime. NTSA-X remains a research state/verification kernel, and its
current Python modules do **not** have trainable neural weights. This milestone
connects approved recall and a separately trainable local student to the live
interface. The student is initially an independently fine-tuned causal model,
**not yet a native trainable non-Transformer NTSA-X foundation model**.

## Architecture

1. **Recall gate:** exact SHA-256 prompt + conversation-history match against
   previously reviewed, non-expired answers. Reuse only within the same session
   or across sessions if the sample is explicitly approved for sharing and was
   generated without prior conversational context.
2. **Teacher mode:** no approved match -> tokens stream from a local OpenAI-compatible
   inference endpoint. The answer remains *unverified*.
3. **Capture gate:** a browser checkbox, off by default, allows the current
   teacher response to be saved as a candidate. No candidate is treated as
   training data until the user reviews it and confirms the right to use it.
4. **Approval gate:** approval applies to a single exact output and records
   provenance, local session scope, a 30-day expiry and explicit permission.
   Human review is not independent factual verification.
5. **Offline learning:** export approved samples into non-overlapping
   train/evaluation JSONL; the student trainer uses real backpropagation with
   PyTorch + PEFT LoRA. Training never contacts the teacher.
6. **Student-only mode:** run the gateway against a separately hosted student
   model and verify that **zero requests go to the teacher**. It still needs an
   inference *runtime* to compute tokens.
7. **Evaluation:** assess independently held-out tasks, refusal behavior,
   factuality, coding, latency, token/s, and contamination. Do not promote a
   model because it merely reproduced cached sentences or lowered training
   loss.

## Execute

From NTSA-X repository root, run the teacher inference server locally on
127.0.0.1:8080. For example, configure your existing GGUF model server to
serve model ID `Qwen2.5-7B-Instruct` through its OpenAI-compatible endpoint.
Use its actual exposed ID if different.

```powershell
py -3.12 -m unittest discover -s tests -v
py -3.12 -m ntsa_x.live_gateway --model Qwen2.5-7B-Instruct
```

Open http://127.0.0.1:8765. Enable **Offer teacher answer for learning review**.
Ask several questions. Review the responses, select **Approve last answer**,
and attest you hold the permissions required by your teacher model license.
Declining the checkbox creates no learning sample. Clearing the session deletes
its transcript and corresponding training samples, including shared samples.

Export the approved examples:

```powershell
py -3.12 -m ntsa_x.learning --train .local/train.jsonl --eval .local/eval.jsonl
```

If there are insufficient different approved prompts to populate both
partitions, export intentionally fails. Keep `.local/` private and out of Git.
Review training samples for personal data, prompt-injection content, incorrect
answers, and teacher output licensing *before* training.

**Actual parameter learning** requires optional training packages and a local
**Hugging Face format trainable student model**, not just a GGUF inference file.
Install PyTorch for your hardware using the appropriate official instructions,
then install compatible recent versions of `transformers` and `peft`.

```powershell
py -3.12 -m pip install transformers peft
py -3.12 -m ntsa_x.train_student --base-model C:\KindredLabs\Models\student-hf --train .local/train.jsonl --eval .local/eval.jsonl --output .local/student-v1 --epochs 1 --merge-model
```

The folder `C:\KindredLabs\Models\student-hf` must contain licensed local
trainable student weights and a compatible tokenizer/chat template. This
command performs gradient descent and writes an adapter and optionally
standalone merged weights, but is not executed automatically when you
approve examples. A GPU is recommended for nontrivial models; large models
can exceed desktop memory. Review `training_report.json` for held-out loss.
Lower held-out loss is **not sufficient** to establish correctness.

Serve your merged student using a compatible local inference engine at
127.0.0.1:8081 (convert to GGUF separately if your chosen server requires it).
Select the student explicitly:

```powershell
py -3.12 -m ntsa_x.live_gateway --student-model kindred-student-v1 --use-student
```

**No teacher endpoint is initialized or called** in this mode. You may omit
`--model` entirely. The student server must actually expose the specified
student ID; naming it does not create or load the model.

## Non-negotiable release gates

- Exact approved recall works and **never** claims to authenticate facts.
- No transcript or teacher data silently enters the verified Nexus DAG.
- Learning never occurs without explicit review and model-output rights.
- The training/evaluation partitions are prompt-disjoint and both nonempty.
- Student-only runs contact no teacher; check the server logs and benchmark on
  unseen prompts before disabling the teacher in a real product.

## Remaining work toward native NTSA-X

The above implements a valid knowledge-transfer pathway, but does not yet
train the NTSA-X recurrent non-Transformer architecture. That requires a
trainable native state-update function, tokenizer or byte stream interface,
differentiable objective, replay buffer, optimizer, checkpoint format, loss
masks, benchmark harness and safe autonomous update policy. Future online
learning should train an isolated candidate model, then promote only after
validation and rollback readiness. No unvalidated update should overwrite a
serving model.

## Privacy and limitations

The local learning corpus stores full captured prompts, conversation history,
and answers. Keep it access-controlled. There is no production-grade identity
system, content moderation, automated fact checker, cryptographic teacher
attestation, or automatic data deletion daemon at this stage; expired samples
are excluded from recall/export, not automatically vacuumed. The browser's
speech recognition can use external vendor services.

Some model licenses and service terms restrict output-based training or
distillation. Explicit approval alone does not confer rights if your source
model license prohibits student training. The program cannot verify legal
permissions and does not certify external facts.
