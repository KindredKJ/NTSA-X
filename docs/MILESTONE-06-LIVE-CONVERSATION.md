# NTSA-X Milestone 06 — Local Live Conversation Bridge

**Status:** Local-first interactive research adapter. This is not a trained NTSA-X model, a general intelligence benchmark, a production cloud service, or proof of the NTSA-X/UMR cognitive design.

This milestone provides a real interaction surface while preserving the integrity of the research kernel:

- Streams text tokens from a separately running OpenAI-compatible inference backend through local HTTP Server-Sent Events (SSE).
- Saves coherent user/assistant turn pairs atomically to a distinct local SQLite transcript store.
- Maintains isolated session histories and prevents overlapping generations in the same session.
- Gives a standalone browser UI with optional microphone dictation and browser speech synthesis. Browser microphone support varies and speech recognition may use the browser vendor's remote service.
- Leaves the existing verification-gated Nexus DAG untouched. **Dialogue is not verified evidence.**
- Binds the gateway to 127.0.0.1 and rejects unexpected Host and Origin values. The upstream inference URL must be an HTTP loopback /v1/chat/completions endpoint.
- Adds regression tests for unverified transcript handling, stream decoding, session isolation, rollback, origin rejection, malformed inputs and resource bounds.

## Local setup (Windows PowerShell)

Run a local OpenAI-compatible inference server *first*. For example, use your existing llama.cpp deployment and load your licensed model GGUF; confirm your model server exposes:

    http://127.0.0.1:8080/v1/chat/completions

From the repository root in a separate PowerShell terminal:

    py -3.12 -m unittest discover -s tests -v
    py -3.12 -m ntsa_x.live_gateway --model YOUR_ACTUAL_LOCAL_MODEL_ID

Then open:

    http://127.0.0.1:8765

Use --endpoint to point to a different **loopback-only** OpenAI-compatible inference server; --port to change the local browser port; --db to place the SQLite file elsewhere. The default SQLite location is ~/.kindred/ntsa-live.sqlite3.

The model must already exist and be running; the gateway does not download, train, or install neural networks.

## Trust boundaries and deployment

The dialogue table's 'unverified' field is deliberately distinct from the Nexus verified lineage chain. Local transcripts are user input and probabilistic model output, not cryptographic attestations, convergence decisions, or approved actions. Failed/incomplete responses do not commit a conversation turn.

This is only suitable for a trusted local machine. Public exposure requires a separate authenticated TLS reverse proxy, rate limiting, audit/privacy controls, strict request limits, server lifecycle supervision and threat review. The browser's optional speech recognition may involve external browser-vendor services; disable microphone input if strictly offline processing is required. Text streaming is implemented; low-latency duplex speech, native ASR/TTS, UMR delegation, bounded-evidence retrieval, and voice interruption remain future integrations.

## Next release gates

1. Connect an actual UMR Kindred runtime through a typed, deny-by-default adapter after its API is verified.
2. Benchmark first-token latency, completed-turn latency, memory quality and cost on a fixed reproducible suite; compare with the underlying local model alone.
3. Add an on-device native speech stack and interruption handling, with user-consented storage/retention controls.
4. Introduce authenticated production deployment only after threat modeling and validation.
