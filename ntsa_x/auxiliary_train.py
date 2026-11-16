"""Train and checkpoint Auxiliary PDC from approved local JSONL examples."""
from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from tempfile import TemporaryDirectory
from hashlib import sha256
import json
import math
import os
import random

import torch
from torch.nn.utils import clip_grad_norm_

from .auxiliary import AUXILIARY_ARCH, AuxiliaryConfig, AuxiliaryModel
from .cog_fu import CogFuModel
from .cog_fu_train import read_conversations


def _seed_for(record: list[dict]) -> int:
    packed = json.dumps(record, ensure_ascii=False, sort_keys=True)
    return int(sha256(packed.encode("utf-8")).hexdigest()[:8], 16)


@torch.no_grad()
def evaluate(model: AuxiliaryModel, records: list[list[dict]]) -> dict[str, float]:
    previous = model.training
    model.eval()
    sums = {"loss": 0.0, "cognitive_loss": 0.0,
            "diffusion_loss": 0.0, "pdc_loss": 0.0}
    try:
        for record in records:
            result = model.joint_loss(record, noise_fraction=0.5,
                                      seed=_seed_for(record))
            for key in sums:
                value = float(result[key].item())
                if not math.isfinite(value):
                    raise RuntimeError(f"Nonfinite Auxiliary evaluation metric: {key}")
                sums[key] += value / len(records)
    finally:
        model.train(previous)
    return sums


def _transfer_cog_fu(model: AuxiliaryModel, source: str | Path,
                     device: str) -> str:
    from .cog_fu_train import load_checkpoint as load_cog_fu
    parent = load_cog_fu(source, device=device)
    expected = model.config
    if (parent.config.width != expected.width
            or parent.config.layers != expected.layers
            or parent.config.max_context != expected.max_context
            or parent.config.max_answer != expected.max_answer
            or parent.config.diffusion_layers != expected.diffusion_layers):
        raise ValueError("COG-FU checkpoint shape does not match Auxiliary")
    target = model.state_dict()
    source_state = parent.state_dict()
    for name, value in source_state.items():
        if name not in target or target[name].shape != value.shape:
            raise ValueError(f"Incompatible COG-FU parameter: {name}")
        target[name] = value.detach().clone()
    model.load_state_dict(target, strict=True)
    return sha256((Path(source) / "weights.pt").read_bytes()).hexdigest()


def train(*, train_path: str | Path, eval_path: str | Path,
          output: str | Path, config: AuxiliaryConfig,
          epochs: int = 4, batch_size: int = 2,
          learning_rate: float = 0.001, clip_norm: float = 1.0,
          device: str = "auto", seed: int = 42,
          initialize_from: str | Path | None = None,
          initialize_cog_fu: str | Path | None = None,
          replay_path: str | Path | None = None) -> dict:
    if (not 1 <= epochs <= 100 or not 1 <= batch_size <= 128
            or not 1e-7 <= learning_rate <= 0.05
            or not 0.01 <= clip_norm <= 100):
        raise ValueError("Invalid Auxiliary training hyperparameters")
    if initialize_from and initialize_cog_fu:
        raise ValueError("Choose one initialization checkpoint")
    if device not in ("auto", "cpu", "cuda"):
        raise ValueError("Unsupported device")
    target_device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    if target_device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable")
    target = Path(output).expanduser().resolve()
    if target.exists():
        raise FileExistsError("Refusing to overwrite an existing checkpoint")
    target.parent.mkdir(parents=True, exist_ok=True)
    training = read_conversations(train_path, config)
    replay = read_conversations(replay_path, config) if replay_path else []
    training.extend(replay)
    heldout = read_conversations(eval_path, config)
    train_prompts = {row[-2]["content"] for row in training}
    eval_prompts = {row[-2]["content"] for row in heldout}
    if train_prompts & eval_prompts:
        raise ValueError("Training/replay and holdout prompts overlap")

    torch.manual_seed(seed)
    random.seed(seed)
    model = AuxiliaryModel(config).to(target_device)
    parent_hash = None
    source = "scratch"
    if initialize_from:
        from .auxiliary import load_checkpoint
        previous = load_checkpoint(initialize_from, device=target_device)
        if previous.config != config:
            raise ValueError("Auxiliary continuation requires exact config match")
        model.load_state_dict(previous.state_dict(), strict=True)
        parent_hash = sha256((Path(initialize_from) / "weights.pt").read_bytes()).hexdigest()
        source = "auxiliary-checkpoint"
    elif initialize_cog_fu:
        parent_hash = _transfer_cog_fu(model, initialize_cog_fu, target_device)
        source = "cog-fu-transfer-pdc-new"

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                  weight_decay=0.01)
    initial = evaluate(model, heldout)
    best = initial["loss"]
    best_state = {key: value.detach().cpu().clone()
                  for key, value in model.state_dict().items()}
    epochs_report = []
    steps = 0
    for epoch in range(epochs):
        random.shuffle(training)
        model.train()
        total = 0.0
        for start in range(0, len(training), batch_size):
            batch = training[start:start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            losses = [model.joint_loss(
                sample, noise_fraction=random.uniform(0.2, 1.0),
                seed=random.randrange(0, 2**31 - 1))["loss"]
                for sample in batch]
            loss = torch.stack(losses).mean()
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError("Nonfinite Auxiliary training loss")
            loss.backward()
            clip_grad_norm_(model.parameters(), clip_norm,
                            error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach().item())
            steps += 1
        measured = evaluate(model, heldout)
        epochs_report.append({
            "epoch": epoch + 1,
            "mean_training_loss": total / math.ceil(len(training) / batch_size),
            "heldout": measured,
        })
        if measured["loss"] < best:
            best = measured["loss"]
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}

    report = {
        "architecture": AUXILIARY_ARCH,
        "source": source,
        "parent_weights_sha256": parent_hash,
        "training_samples_including_replay": len(training),
        "replay_samples": len(replay),
        "heldout_samples": len(heldout),
        "optimizer_steps": steps,
        "total_trainable_parameters": sum(p.numel() for p in model.parameters()
                                           if p.requires_grad),
        "initial_holdout": initial,
        "best_holdout_loss": best,
        "heldout_improved": best < initial["loss"],
        "epochs": epochs_report,
        "seed": seed,
        "device": target_device,
        "pdc_max_cycles": config.pdc_max_cycles,
        "pdc_adaptive_stop": True,
        "general_reasoning_or_skill_transfer_proven": False,
    }
    with TemporaryDirectory(prefix="auxiliary-staging-", dir=target.parent) as temp:
        stage = Path(temp)
        torch.save(best_state, stage / "weights.pt")
        (stage / "config.json").write_text(json.dumps({
            "architecture": AUXILIARY_ARCH, "config": config.to_dict()}, indent=2),
            encoding="utf-8")
        (stage / "report.json").write_text(json.dumps(report, indent=2),
                                           encoding="utf-8")
        os.replace(stage, target)
    return report


def main() -> None:
    parser = ArgumentParser(description="Train Auxiliary NTSA-X PDC")
    parser.add_argument("--train", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--max-context", type=int, default=2048)
    parser.add_argument("--max-answer", type=int, default=256)
    parser.add_argument("--diffusion-layers", type=int, default=3)
    parser.add_argument("--denoise-steps", type=int, default=8)
    parser.add_argument("--pdc-max-cycles", type=int, default=12)
    parser.add_argument("--pdc-min-cycles", type=int, default=2)
    parser.add_argument("--pdc-tolerance", type=float, default=0.015)
    parser.add_argument("--pdc-weight", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--initialize-from")
    parser.add_argument("--initialize-cog-fu")
    parser.add_argument("--replay")
    args = parser.parse_args()
    config = AuxiliaryConfig(
        width=args.width, layers=args.layers, max_context=args.max_context,
        max_answer=args.max_answer, diffusion_layers=args.diffusion_layers,
        denoise_steps=args.denoise_steps, pdc_max_cycles=args.pdc_max_cycles,
        pdc_min_cycles=args.pdc_min_cycles, pdc_tolerance=args.pdc_tolerance,
        pdc_weight=args.pdc_weight)
    result = train(
        train_path=args.train, eval_path=args.eval, output=args.output,
        config=config, epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.learning_rate, device=args.device,
        initialize_from=args.initialize_from,
        initialize_cog_fu=args.initialize_cog_fu, replay_path=args.replay)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
