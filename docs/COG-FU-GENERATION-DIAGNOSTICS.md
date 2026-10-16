# COG-FU language-v1 repeated-character generation: diagnosis

The native local model can lower held-out next-byte and masked-token losses
without producing fluent English. Three observed chat prompts yielded mostly
the letter "e". This is characteristic of an undertrained repetitive decoder,
not evidence of an inference connection problem.

## Two specific training / serving mismatches

- Self-supervised pretraining uses byte continuations from original documents,
  with a BOS prefix. At live inference the prompt contains system, user and
  assistant role markers; these were not learned from that corpus.
- The COG-FU training objective optimizes BOTH a recurrent next-byte predictor
  and a masked diffusion denoiser. Before this change, the live server used
  only iterative absorbing-mask diffusion for output, making no direct use of
  the trained causal output head at generation time.

## Run four controlled tests on an existing checkpoint (no retraining)

Stop neither server: this diagnostic uses a fresh separate process.

PowerShell:

    Set-Location -LiteralPath (Join-Path $HOME "KindredLabs\NTSA-X-GitHub")
    git pull --ff-only origin main
    py -3.12 -m unittest discover -s tests -p test_cog_fu_diagnose.py -v
    py -3.12 -m ntsa_x.cog_fu_diagnose --checkpoint .local/cog-fu-language-v1 --prompt "Hello. What are you?" --decoder both --framing both --max-new-tokens 48 --context-limit 1024

The four rows are diffusion/text, causal-cortex/text,
diffusion/chat and causal-cortex/chat.

Fields:

- response: literal generated text, capped at 48 bytes
- dominant_fraction: fraction of characters representing the most common one
- longest_same_character_run: longest uninterrupted run of the same character
- repetition_collapse_warning: heuristic flag; true for empty output or a
  longer response dominated by a single character / long run

This command uses only native weights and no external model, doesn't edit
checkpoint files and does not require web access.

**Interpretation:** If both decoding modes collapse on plain text, additional
quality data, larger model capacity and training are necessary. If the
causal decoder handles plain text better but fails in chat, instruction/chat
supervision and framing alignment are missing. If diffusion collapses while
causal generation provides more varied text, diffusion inference and
reconstruction alignment require attention. Variation alone never proves
meaning, fluency or factual accuracy.

## Optional runtime ablation

Restart only the COG-FU server after stopping its earlier process:

    py -3.12 -m ntsa_x.cog_fu_server --checkpoint .local/cog-fu-language-v1 --model-id ntsa-cog-fu-language-v1 --port 8081 --max-new-tokens 48 --context-limit 1024 --decoder cortex

Alternatively use --decoder diffusion, the legacy default.
The gateway arguments stay unchanged. The console announces which decoder
is actually active.

## How to truly improve English

1. Obtain and audit a substantially larger, varied, rights-cleared corpus.
2. Establish full-document and near-duplicate-separated heldouts.
3. Train and test more than 80 short windows; increase capacity in measured,
   hardware-appropriate steps, comparing fixed training budgets.
4. Add reviewed chat/instruction pairs with the *same ByteCodec role framing*
   the live gateway uses; evaluate responses on unseen conversation tasks.
5. Track heldout cognitive loss, diffusion loss, valid UTF-8, repeated-byte
   rate, instruction-following, factual calibration and agent tool selection.
6. Keep the verified computation kernel and execution approvals separate
   from unverified model output.

This release supplies diagnostics and decoder selection only. It does not
train new weights or establish fluency.
