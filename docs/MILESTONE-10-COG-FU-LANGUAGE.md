# Milestone 10 — COG-FU native language acquisition (research)

**Status:** Executable reference trainer and tests. No substantial pretraining has been run here. Coherent speech, general reasoning, and superiority over Transformers or recurrence alone have NOT been demonstrated.

The original local checkpoint .local/cog-fu-smoke-v1 is preserved. New training creates new directories and refuses overwrites.

## What's new

- Self-supervised next-byte causal learning and masked-byte diffusion reconstruction are trained **jointly**, with gradients into the recurrent X/Y/Z cognitive state.
- Explicit rights-attested local UTF-8 documents; no automatic downloads, directory crawling or teacher model.
- Exact-duplicate exclusion, document-disjoint train/holdout split, deterministic heldout noise, provenance and model weight lineage.
- Optional matched recurrent-only baseline with the same initial cognitive weights, source windows and training steps; compare **causal held-out losses**, not combined hybrid loss against baseline-only loss.
- UTF-8-safe response output: invalid predicted sequences become question-mark placeholders instead of Unicode replacement glyphs. This is wire repair, not linguistic intelligence.
- Operator-authored local identity context, explicitly not equated to internal learned knowledge.

**Restrictions:** Documents are 32 bytes to 1 MB each; the reference runner accepts 2–1000 documents and 20 MB total. Only manifest-relative .txt files are ingested. The license and rights attestations are supplied by the operator, not independently audited. Near-duplicate text is NOT detected. Do not include secrets, private health records, confidential customer data, credentials or other unlicensed text.

## Step 1 — Prepare a real rights-cleared local corpus

In PowerShell:

    Set-Location -LiteralPath (Join-Path $HOME "KindredLabs\NTSA-X-GitHub")
    New-Item -ItemType Directory -Force ".local\language\texts" | Out-Null

Place **at least three genuinely different UTF-8 text files** that you own or are licensed to train on in that folder: first.txt, second.txt, third.txt.

Create .local/language/manifest.json containing:

    {
      "version": 1,
      "documents": [
        {"path":"texts/first.txt","license":"user-owned-original","rights_confirmed":true},
        {"path":"texts/second.txt","license":"user-owned-original","rights_confirmed":true},
        {"path":"texts/third.txt","license":"user-owned-original","rights_confirmed":true}
      ]
    }

Replace the license descriptions with the actual rights for each source. Each document contributes *all* its windows to train OR holdout, never both. The deterministic holdout comprises ~20% of whole document hashes (minimum one).

## Step 2 — CPU-scale pilot with a matched recurrence baseline

    py -3.12 -m ntsa_x.cog_fu_pretrain --manifest .local/language/manifest.json --output .local/cog-fu-language-v1 --baseline-output .local/recurrent-language-v1 --width 32 --layers 1 --max-context 128 --max-answer 48 --diffusion-layers 1 --denoise-steps 3 --epochs 2 --batch-size 2 --max-train-windows 100 --max-eval-windows 30 --device cpu

Outputs:

- Hybrid: .local/cog-fu-language-v1/weights.pt, config.json, report.json
- Recurrent-only: .local/recurrent-language-v1/weights.pt, config.json, report.json
- The hybrid report records data provenance and document hashes, parameter count, optimization steps, initial/best held-out joint/cognitive/diffusion losses, recurrent baseline held-out causal loss and unresolved near-duplicate limitation.

Caps deliberately constrain runtime for the first trial. Set max-train-windows 0 and max-eval-windows 0 to use all windows. More training, GPU optimization and substantially more diverse legitimate text are needed for useful language acquisition.

## Step 3 — Continue learning without overwriting

    py -3.12 -m ntsa_x.cog_fu_pretrain --manifest .local/language/manifest.json --initialize-from .local/cog-fu-language-v1 --output .local/cog-fu-language-v2 --width 32 --layers 1 --max-context 128 --max-answer 48 --diffusion-layers 1 --denoise-steps 3 --epochs 2 --batch-size 2 --max-train-windows 100 --max-eval-windows 30 --device cpu

Continuation preserves original weights and exports to a new folder. The entire model config must match. Held-out checkpoint selection protects measured joint loss on **that holdout**, not downstream reasoning or forgetting.

## Step 4 — Optional factual identity card

Create a UTF-8 local file:

    @'
    Identity: NTSA-X COG-FU.
    Creator: Kindred Labs.
    Description: Experimental native recurrent cognitive-diffusion hybrid.
    Limitations: Coherent language and reasoning are not yet independently proven.
    '@ | Set-Content -LiteralPath ".local\cog-fu-identity.txt" -Encoding utf8

Operator-authored identity text is included in the gateway's system message using --identity-file and limited to 1200 UTF-8 bytes. This is unverified **operator context**, not trained knowledge, and the model may fail to follow it.

## Step 5 — Run the new model locally

First stop existing ports 8081 and 8765. Terminal 1:

    py -3.12 -m ntsa_x.cog_fu_server --checkpoint .local/cog-fu-language-v1 --model-id ntsa-cog-fu-language-v1 --port 8081 --max-new-tokens 48 --context-limit 1024

Terminal 2:

    py -3.12 -m ntsa_x.live_gateway --student-model ntsa-cog-fu-language-v1 --student-max-tokens 48 --student-endpoint http://127.0.0.1:8081/v1/chat/completions --identity-file .local/cog-fu-identity.txt --use-student

Open http://127.0.0.1:8765. Training on raw texts is not sufficient to produce an instruction-following chat assistant; the next stage needs explicit reviewed dialogue supervision and independent instruction-following evaluation.

## Scientific and operational limits

The reference PyTorch implementation iterates the recurrent cortex serially in Python and is CPU-expensive. The held-out corpus excludes exact duplicate documents only, not partial copied passages. Evaluation measures statistical language modeling objectives, not factuality or cognition. We still need large-scale licensed datasets, contamination audits, equal-compute comparisons, long-context tests, safety testing, instruction tuning and GPU-friendly kernel engineering.
