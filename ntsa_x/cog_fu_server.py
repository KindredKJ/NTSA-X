"""Run COG-FU directly behind NTSA-X's existing local SSE chat contract.

The model is local and self-contained: no external inference engine or teacher.
Diffusion refines a complete bounded answer window before streaming its bytes.
"""
from argparse import ArgumentParser
from dataclasses import replace
from http.server import ThreadingHTTPServer
import torch

from .cog_fu_train import load_checkpoint
from .native_server import handler_factory


def configure_serving_context(model, requested: int) -> int:
    """Raise a recurrent checkpoint's inference context cap without retraining.

    NativeOrbitModel has no context-position embeddings; max_context is only
    a runtime length guard. The byte denoiser's learned positions depend on
    max_answer, which remains unchanged. This makes longer serving contexts
    structurally valid, but DOES NOT establish out-of-distribution quality.
    """
    if (type(requested) is not int or
            not model.config.max_context <= requested <= 32768):
        raise ValueError("Serving context must be >= trained context and <= 32768")
    model.config = replace(model.config, max_context=requested)
    model.cortex.config = replace(model.cortex.config, max_context=requested)
    return requested


def main():
    parser = ArgumentParser(description="Serve native cognitive-diffusion NTSA-X")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-id", default="ntsa-cog-fu-v1")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--decoder", choices=("diffusion", "cortex", "unified"),
                        default="diffusion",
                        help="Serving-only ablation: full-window diffusion "
                             "or causal recurrent cortex. Neither establishes "
                             "English fluency; no checkpoint is changed.")
    parser.add_argument("--memory-db", help="Local SQLite memory path (unified mode only)")
    parser.add_argument("--allow-memory-write", action="store_true",
                        help="Enable local memory commits, separately from generation")
    parser.add_argument("--persist-episodes", action="store_true",
                        help="Record unverified dialogue in local memory; requires write opt-in")
    parser.add_argument("--enable-scalar-extension", action="store_true",
                        help="Attach trusted fixed-function scalar worker inside cognition")
    parser.add_argument("--readonly-workspace",
                        help="Attach only read-only workbench extension")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--context-limit", type=int,
                        help="Serving-only recurrent input cap, default >=1024. "
                             "Weights remain unchanged; larger-than-trained "
                             "contexts have unverified quality.")
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("Invalid local server port")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable")
    model = load_checkpoint(args.checkpoint, device=args.device)
    model.inference_decoder = args.decoder
    # A short smoke checkpoint can have a 128-byte training budget, but the
    # recall-first gateway's honest system banner alone uses 354 UTF-8 bytes.
    # Recurrent weights are context-length agnostic; expand the serving limit
    # without editing the original checkpoint or weakening the gateway prompt.
    requested = (args.context_limit if args.context_limit is not None
                 else max(1024, model.config.max_context))
    try:
        serving_limit = configure_serving_context(model, requested)
    except ValueError as exc:
        parser.error(str(exc))
    if not 1 <= args.max_new_tokens <= model.config.max_answer:
        parser.error("--max-new-tokens exceeds checkpoint's max_answer")
    if args.decoder != "unified" and any(
        (args.memory_db, args.allow_memory_write, args.persist_episodes,
         args.enable_scalar_extension, args.readonly_workspace)):
        parser.error("Native extensions and memory require --decoder unified")
    if args.persist_episodes and (not args.allow_memory_write or not args.memory_db):
        parser.error("Episode persistence requires local memory DB and explicit write opt-in")
    if args.allow_memory_write and not args.memory_db:
        parser.error("Memory writes require explicit --memory-db")
    memory = None
    if args.decoder == "unified":
        from .unified_cognition import (
            UnifiedCognitiveMechanism, scalar_extension, workspace_extension)
        from .memory_fabric import MemoryFabric
        from .agent_workbench import Workbench
        memory = (MemoryFabric(args.memory_db, writable=args.allow_memory_write)
                  if args.memory_db else None)
        extensions = []
        if args.enable_scalar_extension:
            extensions.append(scalar_extension())
        if args.readonly_workspace:
            extensions.append(workspace_extension(Workbench(args.readonly_workspace)))
        model = UnifiedCognitiveMechanism(
            model, memory=memory, extensions=tuple(extensions),
            persist_episodes=args.persist_episodes)
    server = ThreadingHTTPServer(
        ("127.0.0.1", args.port),
        handler_factory(model, args.model_id, args.port, args.max_new_tokens))
    server.daemon_threads = True
    print(f"COG-FU local SSE endpoint http://127.0.0.1:{args.port}/v1/chat/completions",
          flush=True)
    print(f"COG-FU serving decoder: {args.decoder}", flush=True)
    print(f"COG-FU input context cap: {serving_limit} bytes "
          "(may exceed training limit; response quality is unverified).",
          flush=True)
    print("Research system: full-window denoising precedes streamed bytes; "
          "no teacher model and no verified truth claims.", flush=True)
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        if memory is not None:
            memory.close()


if __name__ == "__main__":
    main()
