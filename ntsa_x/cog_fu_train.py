"""COG-FU joint cognitive/diffusion training and versioned checkpoint loading.

Uses approved Milestone 07 JSONL; no teacher calls or Transformer dependency.
Nothing is automatically promoted into the live serving checkpoint.
"""
from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from tempfile import TemporaryDirectory
from dataclasses import asdict
from hashlib import sha256
import json
import math
import os
import random

import torch
from torch.nn.utils import clip_grad_norm_

from .cog_fu import COG_FU_ARCH, CogFuConfig, CogFuModel
from .native_model import ByteCodec
from .native_train import load_checkpoint as load_native_checkpoint


def read_conversations(path: str | Path, config: CogFuConfig) -> list[list[dict]]:
    codec = ByteCodec()
    rows = []
    for index, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            if not isinstance(record, dict) or set(record) != {"messages"}:
                raise ValueError("Expected an object containing only 'messages'")
            messages = record["messages"]
            codec.conversation(messages, max_tokens=config.max_context)
            if (len(messages) < 2 or messages[-2]["role"] != "user"
                    or len(messages[-1]["content"].encode("utf-8")) + 1
                       > config.max_answer):
                raise ValueError("Expected user/assistant pair within answer bound")
            rows.append(messages)
        except (KeyError, ValueError, TypeError) as exc:
            raise ValueError(f"Invalid training sample at line {index}: {exc}") from exc
    if not rows:
        raise ValueError("No approved samples in dataset")
    return rows


def _sample_seed(messages: list[dict]) -> int:
    material = json.dumps(messages, ensure_ascii=False, sort_keys=True)
    return int(sha256(material.encode("utf-8")).hexdigest()[:8], 16)


@torch.no_grad()
def evaluate(model: CogFuModel, records: list[list[dict]]) -> dict[str, float]:
    was_training = model.training
    model.eval()
    total = {"loss": 0.0, "cognitive_loss": 0.0, "diffusion_loss": 0.0}
    try:
        for record in records:
            # Stable across every epoch and every checkpoint. No random eval
            # mask variability may masquerade as model improvement.
            losses = model.joint_loss(
                record, noise_fraction=0.5, seed=_sample_seed(record))
            for key in total:
                value = float(losses[key].item())
                if not math.isfinite(value):
                    raise RuntimeError("Nonfinite held-out COG-FU loss")
                total[key] += value / len(records)
    finally:
        model.train(was_training)
    return total


def load_checkpoint(folder: str | Path, *, device: str = "cpu") -> CogFuModel:
    path = Path(folder)
    info = json.loads((path / "config.json").read_text(encoding="utf-8"))
    if info.get("architecture") != COG_FU_ARCH:
        raise ValueError("Not a matching COG-FU checkpoint")
    config = CogFuConfig(**info["config"])
    model = CogFuModel(config)
    weights = torch.load(path / "weights.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(weights, strict=True)
    return model.to(device).eval()


def train(*, train_path: str | Path, eval_path: str | Path,
          output: str | Path, config: CogFuConfig, epochs: int = 4,
          learning_rate: float = 0.001, batch_size: int = 2,
          clip_norm: float = 1.0, device: str = "auto",
          seed: int = 42, initialize_cortex: str | Path | None = None,
          initialize_from: str | Path | None = None,
          replay_path: str | Path | None = None) -> dict:
    if not (1 <= epochs <= 100 and 1 <= batch_size <= 128
            and 1e-7 <= learning_rate <= 0.05
            and 0.01 <= clip_norm <= 100):
        raise ValueError("Invalid training hyperparameters")
    if initialize_from is not None and initialize_cortex is not None:
        raise ValueError("Select either full hybrid checkpoint or native cortex")
    if device not in ("auto", "cpu", "cuda"):
        raise ValueError("Unsupported device")
    target_device = (
        "cuda" if torch.cuda.is_available() else "cpu"
    ) if device == "auto" else device
    if target_device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable")

    target = Path(output).expanduser().resolve()
    if target.exists():
        raise FileExistsError("Refusing to overwrite a previous COG-FU checkpoint")
    target.parent.mkdir(parents=True, exist_ok=True)
    training = read_conversations(train_path, config)
    rehearsal = read_conversations(replay_path, config) if replay_path else []
    training += rehearsal
    evaluation = read_conversations(eval_path, config)
    train_prompts = {m[-2]["content"] for m in training}
    eval_prompts = {m[-2]["content"] for m in evaluation}
    if train_prompts & eval_prompts:
        raise ValueError("Train/rehearsal and evaluation user prompts overlap")

    torch.manual_seed(seed)
    random.seed(seed)
    parent_hash = None
    source = "scratch"
    if initialize_from is not None:
        previous_path = Path(initialize_from)
        model = load_checkpoint(previous_path, device=target_device)
        if model.config != config:
            raise ValueError("COG-FU continuation must match model config")
        parent_hash = sha256((previous_path / "weights.pt").read_bytes()).hexdigest()
        source = "cog-fu-checkpoint"
    else:
        model = CogFuModel(config).to(target_device)
        if initialize_cortex is not None:
            native_path = Path(initialize_cortex)
            native = load_native_checkpoint(native_path, device="cpu")
            expected = model.cortex.config
            if native.config != expected:
                raise ValueError("Existing native cortex configuration mismatches")
            model.cortex.load_state_dict(native.state_dict(), strict=True)
            parent_hash = sha256((native_path / "weights.pt").read_bytes()).hexdigest()
            source = "native-cortex"

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                  weight_decay=0.01)
    initial = evaluate(model, evaluation)
    best = initial["loss"]
    best_weights = {k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()}
    history = []
    steps = 0
    for epoch in range(epochs):
        random.shuffle(training)
        model.train()
        aggregate = 0.0
        for offset in range(0, len(training), batch_size):
            batch = training[offset:offset + batch_size]
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for example in batch:
                probability = random.uniform(0.20, 1.0)
                losses.append(model.joint_loss(
                    example, noise_fraction=probability)["loss"])
            joint = torch.stack(losses).mean()
            if not bool(torch.isfinite(joint).item()):
                raise RuntimeError("Nonfinite COG-FU training loss")
            joint.backward()
            clip_grad_norm_(model.parameters(), clip_norm, error_if_nonfinite=True)
            optimizer.step()
            aggregate += float(joint.detach().item())
            steps += 1
        observed = evaluate(model, evaluation)
        history.append({"epoch": epoch + 1,
                        "mean_training_batch_loss": aggregate /
                            math.ceil(len(training) / batch_size),
                        "heldout": observed})
        if observed["loss"] < best:
            best = observed["loss"]
            best_weights = {k: v.detach().cpu().clone()
                            for k, v in model.state_dict().items()}
    report = {
        "architecture": COG_FU_ARCH,
        "from": source,
        "previous_weights_sha256": parent_hash,
        "transformer_layers": 0,
        "teacher_calls": 0,
        "trained_cortex_and_denoiser_jointly": True,
        "total_trainable_parameters": sum(
            p.numel() for p in model.parameters() if p.requires_grad),
        "training_samples": len(training),
        "rehearsal_samples": len(rehearsal),
        "holdout_samples": len(evaluation),
        "optimizer_steps": steps,
        "initial_holdout": initial,
        "best_holdout_joint_loss": best,
        "heldout_improved": best < initial["loss"],
        "epochs": history,
        "seed": seed, "device": target_device,
        "independent_reasoning_quality_proven": False,
    }
    with TemporaryDirectory(prefix="ntsa-cog-fu-staging-", dir=target.parent) as folder:
        stage = Path(folder)
        torch.save(best_weights, stage / "weights.pt")
        (stage / "config.json").write_text(
            json.dumps({"architecture": COG_FU_ARCH,
                        "config": config.to_dict()}, indent=2),
            encoding="utf-8")
        (stage / "report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8")
        os.replace(stage, target)
    return report


def main() -> None:
    parser = ArgumentParser(description="Train native NTSA-X COG-FU hybrid")
    parser.add_argument("--train", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--max-context", type=int, default=2048)
    parser.add_argument("--max-answer", type=int, default=256)
    parser.add_argument("--diffusion-layers", type=int, default=3)
    parser.add_argument("--denoise-steps", type=int, default=8)
    parser.add_argument("--diffusion-weight", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"],
                        default="auto")
    parser.add_argument("--initialize-cortex")
    parser.add_argument("--initialize-from")
    parser.add_argument("--replay")
    args = parser.parse_args()
    config = CogFuConfig(
        width=args.width, layers=args.layers, max_context=args.max_context,
        max_answer=args.max_answer, diffusion_layers=args.diffusion_layers,
        denoise_steps=args.denoise_steps,
        diffusion_weight=args.diffusion_weight)
    result = train(train_path=args.train, eval_path=args.eval,
                   output=args.output, config=config, epochs=args.epochs,
                   batch_size=args.batch_size, learning_rate=args.learning_rate,
                   device=args.device, initialize_cortex=args.initialize_cortex,
                   initialize_from=args.initialize_from, replay_path=args.replay)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
