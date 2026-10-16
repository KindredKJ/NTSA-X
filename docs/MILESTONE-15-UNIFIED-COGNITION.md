# Milestone 15 — Unified cognitive mechanism and direct native extensions

**Status:** Experimental, checkpoint-compatible inference architecture + local typed memory + trusted extension slots. NOT a fluent model, secure untrusted-code sandbox, verified knowledge engine, or automatic online neural training.

## Design

One cognitive trajectory contains native X/Y/Z recurrence, one masked-byte denoising pass for the current position, an explicit probability fusion and an immediate recurrent state update after each emitted byte. The same cognitive state is reused by both heads, with no new trainable weights; original COG-FU checkpoints remain byte-for-byte unchanged. This is an **experimental shared-state inference strategy**. It does not remove the existing recurrent and denoiser layers/parameter heads in stored checkpoints, nor does it guarantee coherence or improved learning.

A separate **runtime wrapper** provides shared context to the same unified generator:

1. Decode the user's already validated chat prompt (no separate agent planning or tool-call transcript).
2. Retrieve bounded, unverified working / semantic / episodic / procedural / relational/associative memories.
3. Run explicitly attached *trusted native extensions* concurrently (bounded to two in-request workers, returning results in registration order) as local computation lanes (currently fixed scalar worker and read-only workbench); no model-directed arbitrary tool calls or plugin manifest negotiation.
4. Fold bounded extension computations and memory into the model's input before the unified recurrent/denoising cognitive cycle. Native plugins do useful actual computation or read protected files rather than only declaring capabilities.
5. Optionally write a time-limited unverified episodic observation **only if** operator explicitly enables both local memory storage and writes. Additional semantic/procedural facts require explicit `UnifiedCognitiveMechanism.learn(...)` calls.

All extension text is **untrusted as instructions** and cannot provide verification. Trusted native callbacks execute in the Python process: they **are not sandboxed**. Only vetted allowlisted native extensions should ever be attached. Arbitrary plugin packages and unattended external-effect capabilities are deliberately *not* supported by this prototype. Existing Workbench path checks still apply to its read-only extension; no write/network/browser/execution extension is enabled.

Memory uses local SQLite, scoped working memory, optional TTL, source tags, and one-hop associative edges. Retrieval is lexical overlap plus adjacency, **not** learned semantic embeddings or automated logical inference. It is not the Nexus verified evidence ledger, is not encrypted by this module, and is not suitable for secret storage.

## Local Windows launch

From a non-admin PowerShell terminal in NTSA-X:

    git pull --ff-only origin main
    py -3.12 -m unittest discover -s tests -p test_unified_cognition.py -v
    py -3.12 -m unittest discover -s tests -p test_unified_native_fusion.py -v

Keep the old server if you want its original model. To switch the test endpoint, first stop the old service on port 8081, then launch:

    py -3.12 -m ntsa_x.cog_fu_server `
        --checkpoint .local/cog-fu-academy-m14-v1 `
        --model-id ntsa-unified-m15 `
        --port 8081 `
        --max-new-tokens 48 `
        --context-limit 1024 `
        --decoder unified `
        --memory-db .local/unified-memory-m15.sqlite `
        --enable-scalar-extension `
        --readonly-workspace .

The new local SQLite memory is created. **Memory writes remain disabled by default**. For an explicitly approved local episodic logging experiment, also pass BOTH `--allow-memory-write` and `--persist-episodes`, acknowledging dialogue content will be stored in the new local DB. No raw ledger, secrets, credentials or sensitive data should be placed there.

Second terminal (existing compatible gateway):

    py -3.12 -m ntsa_x.live_gateway `
        --student-model ntsa-unified-m15 `
        --student-max-tokens 48 `
        --student-endpoint http://127.0.0.1:8081/v1/chat/completions `
        --use-student

Background processing levels are **bounded, inside a single incoming request**:
- L0: immediate state-coupled decoding and denoising
- L1: session-scoped working plus other typed memory recall
- L2: up to two native extension workers in parallel, observations ordered
- L3: episodic memory consolidation if and only if writes were explicitly enabled
- L4: offline model adaptation intentionally disabled; there is no autonomous optimizer

**This is not a persistent always-on background agent**; it does not run while the service is idle. Active background scheduling, GPU acceleration and learned plugin routing require a separate validated design.

The runtime keeps the existing SSE and model IDs intact. Use the prompts `scalar increment 7 by 2` or `search workspace for cortex` to route trusted compute extensions *inside the same cognitive runtime*. Plugin observations contribute as context to both native heads. Because model weights remain severely undertrained, outputs might still be repeated letters even if the extension result is correct.

## Verification and limits

Compare model output on unseen held-out tasks against the old recurrent-only and diffusion-only baselines. Report repetition collapse, exact verified arithmetic computation in extension observations, external-tool safety, memory contamination, response quality and wall-clock compute. Further research must measure whether learned shared-state feedback actually improves coherence; objective/loss reductions alone are not proof. No training or live deployment is performed by merging this branch.
