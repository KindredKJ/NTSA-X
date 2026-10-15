"""Run COG-FU directly behind NTSA-X's existing local SSE chat contract.

The model is local and self-contained: no external inference engine or teacher.
Diffusion refines a complete bounded answer window before streaming its bytes.
"""
from argparse import ArgumentParser
from http.server import ThreadingHTTPServer
import torch

from .cog_fu_train import load_checkpoint
from .native_server import handler_factory


def main():
    parser = ArgumentParser(description="Serve native cognitive-diffusion NTSA-X")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-id", default="ntsa-cog-fu-v1")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("Invalid local server port")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable")
    model = load_checkpoint(args.checkpoint, device=args.device)
    if not 1 <= args.max_new_tokens <= model.config.max_answer:
        parser.error("--max-new-tokens exceeds checkpoint's max_answer")
    server = ThreadingHTTPServer(
        ("127.0.0.1", args.port),
        handler_factory(model, args.model_id, args.port, args.max_new_tokens))
    server.daemon_threads = True
    print(f"COG-FU local SSE endpoint http://127.0.0.1:{args.port}/v1/chat/completions",
          flush=True)
    print("Research system: full-window denoising precedes streamed bytes; "
          "no teacher model and no verified truth claims.", flush=True)
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
