# Milestone 14 — Context-compatible Academy training pilot

**Why this exists:** The observed local checkpoint has a 128-byte training context, but every original Academy teaching sequence exceeded it (longest reported: 377 bytes). All target answers fit within its 48-byte answer limit. Increasing the live server context does not solve this training problem. Truncating explanations would defeat the user's detailed-curriculum goal.

## Engineering mechanism

- Preserve the original original COG-FU weights and all its configuration other than a versioned increase of the maximum recurrent training context. The recurrent context cap has no learnable position table; the state dictionary can be loaded strictly into a new instance with an enlarged context.
- Context expansion requires **explicit `--allow-context-expansion`** and **`--initialize-from`**. Changing width, depth, answer length, denoise steps, or any other model property is rejected for checkpoint continuation.
- Training always writes a NEW checkpoint folder and refuses overwrites.
- Prepare **12 training** and **12 development** supervised dialogues from the existing 24 original exercises. Every lesson contributes one of each. The 12 reserved exam items are not accessed by the training bridge or exported into either training set.
- Existing training behavior, diffusion loss, and native recurrent architecture are otherwise unchanged.

**Limits:** The development prompts use the *same* underlying explanations as training, so this is a software/optimization smoke test, **not a concept-disjoint proof of generalization**. The 12 reserved exams are tiny and should not be used repeatedly to select a checkpoint. A tiny dataset cannot produce fluent English without other substantial language data and adequate compute/model capacity.

## 1. Prepare the length-compatible data

From a **non-admin** PowerShell session:

    Set-Location -LiteralPath (Join-Path $HOME "KindredLabs\NTSA-X-GitHub")
    git pull --ff-only origin main
    py -3.12 -m unittest discover -s tests -p test_academy_bridge.py -v
    py -3.12 -m ntsa_x.academy_bridge
    py -3.12 -m ntsa_x.academy_bridge --prepare .local/academy/m14-v1

The report prints the old context, actual required sequence, suggested rounded-up training context, model configuration, SHA-256 source provenance, and counts. The command will refuse to overwrite the prepared directory.

## 2. One-epoch supervised training pilot (manual, never run automatically)

Check `.local/cog-fu-language-v1/config.json` first. **If its actual settings are** width=32, layers=1, max_context=128, max_answer=48, diffusion_layers=1 and denoise_steps=3, and the Academy bridge reported suggested training context=384, use:

    py -3.12 -m ntsa_x.cog_fu_train --train .local/academy/m14-v1/train.jsonl --eval .local/academy/m14-v1/dev.jsonl --initialize-from .local/cog-fu-language-v1 --allow-context-expansion --output .local/cog-fu-academy-m14-v1 --width 32 --layers 1 --max-context 384 --max-answer 48 --diffusion-layers 1 --denoise-steps 3 --epochs 1 --batch-size 1 --device cpu

If any checkpoint setting differs, **do not use these hardcoded numbers**; translate the report's checkpoint_model_config into the training command while changing only max_context to the reported suggested_training_context.

The trainer makes a new versioned checkpoint, evaluates initial and final development loss, records previous weights SHA and the original context. This is a pilot, not sufficient production training. It can be slow on CPU; run it only when ready to wait. Do not point your live server to the new checkpoint until comparing its outputs and untouched exams.

## 3. Measure before promoting

Run `python -m ntsa_x.cog_fu_diagnose` for both source and new checkpoint using the same prompts and native/recurrent decoders. Compare repetitiveness, readability and *independent* task success. Do not promote based solely on a loss reduction. Add fresh exam problems held out across concepts, avoid memorized test questions, and gather more appropriately licensed explanatory material before scaling training.

No teacher calls, no network downloads, no Transformer, no AI-produced answer labels, and no implicit authorization to train on proprietary textbooks.
