"""Teacher-free diagnostic for COG-FU repetitive-byte generation.

Runs the SAME checkpoint with the native recurrent and diffusion decoders
under both plain-text and live-chat input framing. This distinguishes model
collapse from context-framing mismatch; it does not claim an improvement.
No training, network requests, checkpoint writes, or outside model weights.
"""
from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from pathlib import Path
import json
import math

from .live_gateway import SYSTEM_MESSAGE


def generation_metrics(text: str) -> dict:
    """Descriptive warnings only; no unsupported intelligence score."""
    if not isinstance(text, str):
        raise ValueError("Expected generated text")
    if not text:
        return {
            "characters": 0, "distinct_characters": 0,
            "dominant_character": None, "dominant_fraction": 0.0,
            "longest_same_character_run": 0,
            "repetition_collapse_warning": True,
            "warning_reason": "empty_response",
        }
    frequency = Counter(text)
    dominant_char, dominant_count = frequency.most_common(1)[0]
    longest, current, previous = 0, 0, None
    for char in text:
        if char == previous:
            current += 1
        else:
            previous, current = char, 1
        longest = max(longest, current)
    fraction = dominant_count / len(text)
    collapsed = len(text) >= 12 and (
        fraction >= 0.65 or longest >= 12)
    return {
        "characters": len(text),
        "distinct_characters": len(frequency),
        "dominant_character": dominant_char,
        "dominant_fraction": round(fraction, 4),
        "longest_same_character_run": longest,
        "repetition_collapse_warning": collapsed,
        "warning_reason": ("dominant_character_or_long_run"
                           if collapsed else None),
    }


def diagnose(*, checkpoint: str | Path, prompt: str,
             decoder: str = "both", framing: str = "both",
             max_new_tokens: int = 48,
             context_limit: int = 1024,
             device: str = "cpu") -> dict:
    if (not isinstance(prompt, str) or not prompt.strip()
            or len(prompt.encode("utf-8")) > 2000):
        raise ValueError("Prompt must be 1-2000 UTF-8 bytes")
    if decoder not in {"both", "cortex", "diffusion"}:
        raise ValueError("Unknown decoder")
    if framing not in {"both", "text", "chat"}:
        raise ValueError("Unknown prompt framing")
    if device not in {"cpu", "cuda"}:
        raise ValueError("Unsupported compute device")
    # Keep simple metrics importable when optional PyTorch is not installed.
    from .cog_fu_server import configure_serving_context
    from .cog_fu_train import load_checkpoint
    from .native_model import ByteCodec

    model = load_checkpoint(checkpoint, device=device)
    configure_serving_context(model, context_limit)
    if not 1 <= max_new_tokens <= model.config.max_answer:
        raise ValueError("Token count exceeds checkpoint max_answer")
    prompts: dict[str, list[int]] = {}
    if framing in {"both", "text"}:
        prompts["text"] = [ByteCodec.BOS, *prompt.encode("utf-8")]
    if framing in {"both", "chat"}:
        prompts["chat"], _ = model.codec.conversation([
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": prompt}],
            generation=True, max_tokens=model.config.max_context)
    output = []
    for format_name, prefix in prompts.items():
        if len(prefix) > model.config.max_context:
            raise ValueError("Prompt exceeds diagnostic context limit")
        for mode in ("diffusion", "cortex"):
            if decoder != "both" and decoder != mode:
                continue
            model.inference_decoder = mode
            text = bytes(model.generate(
                prefix, max_new_tokens=max_new_tokens, temperature=0.0)
            ).decode("utf-8", errors="strict")
            entry = {
                "framing": format_name,
                "decoder": mode,
                "prompt_tokens": len(prefix),
                "response": text,
                "metrics": generation_metrics(text),
            }
            output.append(entry)
    return {
        "architecture": model.architecture,
        "checkpoint": str(Path(checkpoint)),
        "training_max_context": load_training_context(checkpoint),
        "serving_max_context": context_limit,
        "checkpoint_modified": False,
        "teacher_calls": 0,
        "results": output,
        "limitations": (
            "Repetition is a descriptive diagnostic, not a quality grade. "
            "Raw-text pretraining differs from system/user/assistant chat "
            "framing. Neither decoder is guaranteed to be fluent. "
            "This command cannot evaluate learned reasoning.")
    }


def load_training_context(checkpoint: str | Path) -> int:
    info = json.loads(
        (Path(checkpoint) / "config.json").read_text(encoding="utf-8"))
    return int(info["config"]["max_context"])


def main():
    parser = ArgumentParser(
        description="Compare COG-FU diffusion and recurrent generation with "
                    "no retraining or external teacher")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", default="Hello. What are you?")
    parser.add_argument("--decoder",
                        choices=("both", "diffusion", "cortex"), default="both")
    parser.add_argument("--framing",
                        choices=("both", "text", "chat"), default="both")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--context-limit", type=int, default=1024)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    try:
        outcome = diagnose(
            checkpoint=args.checkpoint,
            prompt=args.prompt, decoder=args.decoder,
            framing=args.framing, max_new_tokens=args.max_new_tokens,
            context_limit=args.context_limit, device=args.device)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(outcome, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
