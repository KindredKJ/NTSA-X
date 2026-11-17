"""Nine measured Krystal cognition probes: three rounds, three distinct tasks.

Not an official frontier benchmark. Without a supplied training checkpoint
the script uses a real, seeded but UNTRAINED native COG-FU model. It never
presents random weights as a trained model or fabricates successful answers.
"""
from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
from time import perf_counter

import torch

from ntsa_x.cog_fu import CogFuConfig, CogFuModel
from ntsa_x.cog_fu_train import load_checkpoint
from ntsa_x.krystal_cycle import KrystalSingleCycle


# Nine independently gradable requests, three tests in each of three rounds.
# These are diagnostic probes, NOT samples drawn from official benchmark sets.
TASKS = (
    ("Arithmetic", (
        ("What is 2 plus 3? Respond with the number only.", "5"),
        ("What is 17 minus 8? Respond with the number only.", "9"),
        ("What is 12 times 7? Respond with the number only.", "84"),
    )),
    ("General knowledge", (
        ("What is the capital of France? Respond with one word.", "Paris"),
        ("Which planet is called the Red Planet? Respond with one word.", "Mars"),
        ("What is the chemical symbol for gold? Respond with the symbol only.", "Au"),
    )),
    ("Python fundamentals", (
        ("Which Python keyword defines a function? One word only.", "def"),
        ("What does Python len('cat') return? Respond with a number.", "3"),
        ("Which Python keyword starts an exception handler? One word only.", "except"),
    )),
)


def run(*, checkpoint: str | None, output: str, seed: int,
        max_new_tokens: int) -> dict:
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Invalid seed")
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    if checkpoint:
        path = Path(checkpoint).expanduser().resolve()
        required = [path / item for item in ("weights.pt", "config.json")]
        if not all(item.is_file() for item in required):
            raise FileNotFoundError("Trained checkpoint unavailable: " + str(path))
        model = load_checkpoint(path, device="cpu")
        checkpoint_digest = sha256((path / "weights.pt").read_bytes()).hexdigest()
        weight_source = "provided_checkpoint"
    else:
        # Real native model, but initialized randomly. These results CANNOT
        # be compared with trained frontier-model published scores.
        model = CogFuModel(CogFuConfig(
            width=16, layers=1, max_context=512, max_answer=32,
            diffusion_layers=1, denoise_steps=1))
        checkpoint_digest = None
        weight_source = "untrained_random_weights"
    model.eval()
    if max_new_tokens > model.config.max_answer:
        raise ValueError("Generation exceeds checkpoint maximum answer length")
    engine = KrystalSingleCycle(model, memory=None)
    results = []
    for round_index in range(3):
        for category, cases in TASKS:
            question, expected = cases[round_index]
            start = perf_counter()
            result = engine.run(question, max_new_tokens=max_new_tokens)
            elapsed = perf_counter() - start
            actual = result.text.strip()
            score = int(actual.casefold() == expected.casefold())
            record = {
                "round": round_index + 1,
                "test": category,
                "question": question,
                "expected": expected,
                "actual": result.text,
                "exact_match": score,
                "seconds": round(elapsed, 6),
                "output_utf8_bytes": len(result.text.encode("utf-8")),
                "model_emitted_bytes": result.selected_bytes,
                "head_challenges": result.head_challenges,
                "head_agreement": result.head_agreement_fraction,
                "recalled_items": len(result.recalled_ids),
                "externally_verified": result.externally_verified,
            }
            results.append(record)
            print("BENCH_RESULT " + json.dumps(record, ensure_ascii=True), flush=True)
    payload = {
        "protocol": "krystal-3-rounds-x-3-types-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "kind": "DIAGNOSTIC_ONLY_NOT_OFFICIAL_FRONTIER_BENCHMARK",
        "weight_source": weight_source,
        "weights_sha256": checkpoint_digest,
        "seed": seed,
        "torch": torch.__version__,
        "device": "cpu",
        "git_sha": os.environ.get("GITHUB_SHA"),
        "model_configuration": asdict(model.config),
        "generation_limit_bytes": max_new_tokens,
        "total_tests": len(results),
        "exact_matches": sum(item["exact_match"] for item in results),
        "mean_seconds": sum(item["seconds"] for item in results) / len(results),
        "results": results,
    }
    dest = Path(output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
                    encoding="utf-8")
    print("BENCH_SUMMARY " + json.dumps({
        "total": payload["total_tests"],
        "exact_matches": payload["exact_matches"],
        "weight_source": weight_source,
        "path": str(dest),
    }), flush=True)
    return payload


if __name__ == "__main__":
    parser = ArgumentParser(description="Actual 3x3 Krystal research probes")
    parser.add_argument("--checkpoint",
                        help="Existing real checkpoint dir; if omitted UNTRAINED")
    parser.add_argument("--output", default="krystal-9-real-probes.json")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-new-tokens", type=int, default=20)
    args = parser.parse_args()
    run(checkpoint=args.checkpoint, output=args.output,
        seed=args.seed, max_new_tokens=args.max_new_tokens)
