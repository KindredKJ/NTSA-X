"""Create reproducible NTSA-X baseline and genuinely trained COG-FU checkpoints.

Train on text already present in this repository. This is REAL source text and
REAL gradient-based learning; no teacher, fabricated examples, or benchmark
answer labels. A separate document is held out. Model is tiny and not expected
to be capable at frontier tasks.
"""
from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
import json
import math
import random

import torch

from ntsa_x.cog_fu import COG_FU_ARCH, CogFuConfig, CogFuModel
from ntsa_x.cog_fu_train import load_checkpoint

TRAIN_SOURCES = (
    "README.md",
    "docs/ARCHITECTURE.md",
    "docs/MILESTONE-15-UNIFIED-COGNITION.md",
)
HELD_OUT_SOURCE = "docs/VALIDATION.md"


def windows(raw: bytes, *, prefix_bytes: int = 48,
            answer_bytes: int = 24, stride: int = 96):
    width = prefix_bytes + answer_bytes
    if len(raw) < width:
        raise ValueError("Insufficient real source content")
    return [
        (list(raw[i:i + prefix_bytes]),
         list(raw[i + prefix_bytes:i + width]))
        for i in range(0, len(raw) - width + 1, stride)
    ]


def save_checkpoint(model: CogFuModel, directory: Path, report: dict) -> str:
    if directory.exists():
        raise FileExistsError("Refusing to overwrite a checkpoint: " + str(directory))
    directory.mkdir(parents=True)
    torch.save(model.state_dict(), directory / "weights.pt")
    (directory / "config.json").write_text(json.dumps({
        "architecture": COG_FU_ARCH, "config": model.config.to_dict()
    }, indent=2), encoding="utf-8")
    digest = sha256((directory / "weights.pt").read_bytes()).hexdigest()
    report["weights_sha256"] = digest
    (directory / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return digest


def heldout_loss(model: CogFuModel, items: list[tuple[list[int], list[int]]]):
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            losses = [float(model.language_loss(
                prefix, continuation, noise_fraction=0.5,
                seed=12345 + i)["loss"].item())
                for i, (prefix, continuation) in enumerate(items)]
        if not all(math.isfinite(x) for x in losses):
            raise ArithmeticError("Non-finite held-out losses")
        return sum(losses) / len(losses)
    finally:
        model.train(was_training)


def main(*, output_dir: str, steps: int, seed: int) -> dict:
    if not 1 <= steps <= 400:
        raise ValueError("steps outside allowed bound")
    if not 0 <= seed < 2**32:
        raise ValueError("invalid seed")
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng = random.Random(seed)
    root = Path(__file__).resolve().parents[1]
    out = Path(output_dir)
    if out.exists():
        raise FileExistsError("Refusing to overwrite existing output directory")
    sources = {
        p: (root / p).read_bytes()
        for p in (*TRAIN_SOURCES, HELD_OUT_SOURCE)
    }
    assert HELD_OUT_SOURCE not in TRAIN_SOURCES
    samples = []
    for name in TRAIN_SOURCES:
        samples.extend(windows(sources[name]))
    heldout = windows(sources[HELD_OUT_SOURCE], stride=160)
    if not samples or not heldout:
        raise RuntimeError("Missing real training or independent holdout content")
    config = CogFuConfig(width=16, layers=1, max_context=512,
                         max_answer=32, diffusion_layers=1, denoise_steps=1)
    model = CogFuModel(config)
    initial = heldout_loss(model, heldout)
    common = {
        "architecture": COG_FU_ARCH,
        "torch": torch.__version__,
        "seed": seed,
        "config": asdict(config),
        "train_source_sha256": {name: sha256(sources[name]).hexdigest()
                                for name in TRAIN_SOURCES},
        "holdout_source": HELD_OUT_SOURCE,
        "holdout_sha256": sha256(sources[HELD_OUT_SOURCE]).hexdigest(),
        "training_sample_windows": len(samples),
        "holdout_sample_windows": len(heldout),
        "original_holdout_loss": initial,
        "official_frontier_score": None,
    }
    baseline_digest = save_checkpoint(model, out / "baseline", {
        **common, "training_steps": 0, "kind": "UNTRAINED_BASELINE",
    })

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002,
                                  weight_decay=0.01)
    history = []
    model.train()
    for step in range(steps):
        prefix, continuation = samples[rng.randrange(len(samples))]
        optimizer.zero_grad(set_to_none=True)
        objective = model.language_loss(
            prefix, continuation, noise_fraction=0.5,
            seed=seed + step + 1)["loss"]
        if not bool(torch.isfinite(objective).item()):
            raise ArithmeticError("Nonfinite training objective")
        objective.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 8 == 0 or step == steps - 1:
            history.append({"step": step + 1,
                            "training_joint_loss": float(objective.item())})
    final = heldout_loss(model, heldout)
    trained_digest = save_checkpoint(model, out / "trained", {
        **common, "kind": "TRAINED_BYTE_NATIVE_COG_FU",
        "training_steps": steps,
        "learning_rate": 0.002,
        "training_loss_observations": history,
        "trained_holdout_loss": final,
        "heldout_loss_improvement": initial - final,
        "training_process": "joint native next-byte and masked-byte losses",
        "excludes_probe_questions": True,
        "trained_on_only_project_documentation": True,
        "frontier_claim": False,
    })
    # Confirm actual strict checkpoint-load compatibility.
    loaded = load_checkpoint(out / "trained", device="cpu")
    if any(not torch.equal(loaded.state_dict()[k], model.state_dict()[k])
           for k in model.state_dict()):
        raise AssertionError("Checkpoint reload changed trained weights")
    output = {"baseline_sha256": baseline_digest,
              "trained_sha256": trained_digest, "steps": steps,
              "initial_holdout_loss": initial, "trained_holdout_loss": final,
              "train_windows": len(samples), "holdout_windows": len(heldout)}
    print("CHECKPOINT_TRAINING " + json.dumps(output), flush=True)
    return output


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--output-dir", default="krystal-checkpoints")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=7)
    a = parser.parse_args()
    main(output_dir=a.output_dir, steps=a.steps, seed=a.seed)
