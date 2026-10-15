"""Train COG-FU from rights-attested local text with held-out source documents.

Reference CPU/GPU learner, not a distributed pretraining system. Optional
equal-step, same-initialization standalone recurrent-only baseline enables
an honest architecture ablation on the very same held-out windows.
"""
from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import math
import os
import random

import torch
from torch import nn
from torch.nn import functional as F

from .cog_fu import COG_FU_ARCH, CogFuConfig, CogFuModel
from .cog_fu_train import load_checkpoint
from .cog_fu_language_data import LanguageWindow, load_corpus
from .native_model import NativeOrbitModel


def cortex_window_loss(cortex: NativeOrbitModel, window: LanguageWindow):
    device = cortex.embedding.weight.device
    tokens = torch.tensor([window.prefix + window.continuation],
                          dtype=torch.long, device=device)
    logits, _ = cortex(tokens)
    begin = len(window.prefix) - 1
    selected = logits[:, begin:-1, :]
    targets = tokens[:, len(window.prefix):]
    return F.cross_entropy(
        selected.reshape(-1, cortex.config.vocab_size),
        targets.reshape(-1))


def _window_seed(window: LanguageWindow) -> int:
    digest = sha256(bytes(window.continuation) +
                    bytes(str(window.prefix), "ascii")).digest()
    return int.from_bytes(digest[:4], "big")


@torch.no_grad()
def evaluate_hybrid(model: CogFuModel, windows: list[LanguageWindow]):
    model.eval()
    scores = {"joint_loss": 0.0, "cognitive_loss": 0.0,
              "diffusion_loss": 0.0}
    for window in windows:
        losses = model.language_loss(
            list(window.prefix), list(window.continuation),
            noise_fraction=0.5, seed=_window_seed(window))
        for output, loss_key in (
                ("joint_loss", "loss"), ("cognitive_loss", "cognitive_loss"),
                ("diffusion_loss", "diffusion_loss")):
            value = float(losses[loss_key].item())
            if not math.isfinite(value):
                raise RuntimeError("Nonfinite held-out COG-FU loss")
            scores[output] += value / len(windows)
    return scores


@torch.no_grad()
def evaluate_cortex(model: NativeOrbitModel, windows: list[LanguageWindow]):
    model.eval()
    result = sum(float(cortex_window_loss(model, w).item())
                 for w in windows) / len(windows)
    if not math.isfinite(result):
        raise RuntimeError("Nonfinite held-out recurrent-only loss")
    return result


def _atomic_checkpoint(destination: Path, model: nn.Module,
                       config: dict, report: dict):
    if destination.exists():
        raise FileExistsError(f"Cannot overwrite checkpoint: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="cog-fu-language-stage-",
                            dir=destination.parent) as stage:
        directory = Path(stage)
        torch.save({k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()},
                   directory / "weights.pt")
        (directory / "config.json").write_text(
            json.dumps(config, indent=2), encoding="utf-8")
        (directory / "report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8")
        os.replace(directory, destination)


def pretrain(*, manifest: str | Path, output: str | Path,
             config: CogFuConfig, epochs: int = 2,
             learning_rate: float = 0.001, batch_size: int = 2,
             seed: int = 42, device: str = "auto",
             initialize_from: str | Path | None = None,
             baseline_output: str | Path | None = None,
             max_train_windows: int = 0, max_eval_windows: int = 0):
    if (not 1 <= epochs <= 100 or not 1e-7 <= learning_rate <= 0.05
            or not 1 <= batch_size <= 128 or
            not 0 <= max_train_windows <= 1000000 or
            not 0 <= max_eval_windows <= 1000000 or
            device not in {"auto", "cpu", "cuda"}):
        raise ValueError("Invalid training configuration")
    location = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    if location == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable")
    target = Path(output).expanduser().resolve()
    comparison = Path(baseline_output).expanduser().resolve() if baseline_output else None
    if target.exists() or (comparison and comparison.exists()):
        raise FileExistsError("Refusing to overwrite any checkpoint")
    if comparison == target:
        raise ValueError("Hybrid and baseline checkpoints must be distinct")
    train_windows, eval_windows, corpus = load_corpus(manifest, config)
    if max_train_windows:
        train_windows = train_windows[:max_train_windows]
    if max_eval_windows:
        eval_windows = eval_windows[:max_eval_windows]
    torch.manual_seed(seed)
    random.seed(seed)
    previous = None
    if initialize_from:
        original = Path(initialize_from).expanduser().resolve()
        model = load_checkpoint(original, device=location)
        if model.config != config:
            raise ValueError("Continuation must match existing checkpoint configuration")
        previous = sha256((original / "weights.pt").read_bytes()).hexdigest()
    else:
        model = CogFuModel(config).to(location)
    # Same cognitive weights, same document windows, same optimization steps.
    baseline = None
    if comparison:
        baseline = NativeOrbitModel(model.cortex.config).to(location)
        baseline.load_state_dict(model.cortex.state_dict(), strict=True)
        opt_baseline = torch.optim.AdamW(
            baseline.parameters(), lr=learning_rate, weight_decay=0.01)
    opt = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    best_hybrid = evaluate_hybrid(model, eval_windows)
    initial_hybrid = dict(best_hybrid)
    best_hybrid_weights = {k: v.detach().cpu().clone()
                           for k, v in model.state_dict().items()}
    baseline_initial = evaluate_cortex(baseline, eval_windows) if baseline else None
    best_baseline = baseline_initial
    best_baseline_weights = (
        {k: v.detach().cpu().clone() for k, v in baseline.state_dict().items()}
        if baseline else None)
    history, steps = [], 0
    for epoch in range(epochs):
        order = list(train_windows)
        random.shuffle(order)
        model.train()
        if baseline:
            baseline.train()
        running, running_base, batches = 0.0, 0.0, 0
        for start in range(0, len(order), batch_size):
            group = order[start:start + batch_size]
            opt.zero_grad(set_to_none=True)
            losses = [
                model.language_loss(
                    list(row.prefix), list(row.continuation),
                    noise_fraction=random.uniform(0.2, 1.0))["loss"]
                for row in group]
            loss = torch.stack(losses).mean()
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError("Nonfinite language training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0,
                                     error_if_nonfinite=True)
            opt.step()
            running += float(loss.detach().item())
            if baseline:
                opt_baseline.zero_grad(set_to_none=True)
                baseline_loss = torch.stack([
                    cortex_window_loss(baseline, w) for w in group]).mean()
                if not bool(torch.isfinite(baseline_loss).item()):
                    raise RuntimeError("Nonfinite recurrent baseline loss")
                baseline_loss.backward()
                nn.utils.clip_grad_norm_(baseline.parameters(), 1.0,
                                         error_if_nonfinite=True)
                opt_baseline.step()
                running_base += float(baseline_loss.detach().item())
            steps += 1
            batches += 1
        outcome = evaluate_hybrid(model, eval_windows)
        baseline_loss = evaluate_cortex(baseline, eval_windows) if baseline else None
        if outcome["joint_loss"] < best_hybrid["joint_loss"]:
            best_hybrid = dict(outcome)
            best_hybrid_weights = {k: v.detach().cpu().clone()
                                   for k, v in model.state_dict().items()}
        if baseline and baseline_loss < best_baseline:
            best_baseline = baseline_loss
            best_baseline_weights = {
                k: v.detach().cpu().clone()
                for k, v in baseline.state_dict().items()}
        history.append({
            "epoch": epoch + 1,
            "mean_training_batch_joint_loss": running / batches,
            "holdout": outcome,
            "baseline_mean_training_batch_causal_loss":
                running_base / batches if baseline else None,
            "baseline_holdout_causal_loss": baseline_loss})
    report = {
        "architecture": COG_FU_ARCH,
        "training_objective": "native-causal-plus-masked-discrete-diffusion",
        "pretraining_from": "hybrid-checkpoint" if initialize_from else "scratch",
        "previous_weights_sha256": previous,
        "teacher_calls": 0, "transformer_layers": 0,
        "trainable_parameters": sum(p.numel() for p in model.parameters()),
        "steps": steps, "epochs": history,
        "window_counts": {"training": len(train_windows),
                          "holdout": len(eval_windows)},
        "holdout": {"initial": initial_hybrid, "best": best_hybrid,
                    "improved": best_hybrid["joint_loss"]
                                < initial_hybrid["joint_loss"]},
        "baseline": {
            "trained": baseline is not None,
            "same_initial_cognitive_weights": bool(baseline),
            "same_training_windows_and_step_budget": bool(baseline),
            "initial_causal_loss": baseline_initial,
            "best_causal_loss": best_baseline},
        "corpus": corpus, "seed": seed, "device": location,
        "model_capability_proven": False,
        "near_duplicate_contamination_ruled_out": False,
    }
    # Restore BEST weights before checkpointing, without modifying a live model.
    model.load_state_dict(best_hybrid_weights)
    _atomic_checkpoint(target, model, {
        "architecture": COG_FU_ARCH, "config": asdict(config)}, report)
    if comparison:
        baseline.load_state_dict(best_baseline_weights)
        baseline_report = {
            "architecture": "ntsa-x-language-recurrent-baseline-v1",
            "same_input_windows_and_steps_as_hybrid": True,
            "holdout_initial_causal_loss": baseline_initial,
            "holdout_best_causal_loss": best_baseline,
            "training_steps": steps, "corpus_document_sha256": {
                "training": corpus["training_document_sha256"],
                "holdout": corpus["holdout_document_sha256"]},
            "seed": seed, "transformer_layers": 0}
        _atomic_checkpoint(
            comparison, baseline, asdict(baseline.config), baseline_report)
    return report


def main():
    p = ArgumentParser(description="COG-FU rights-attested native language acquisition")
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--initialize-from",
                   help="Continue training from a compatible COG-FU checkpoint")
    p.add_argument("--baseline-output",
                   help="Train a separate same-step recurrent-only baseline")
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--max-context", type=int, default=256)
    p.add_argument("--max-answer", type=int, default=64)
    p.add_argument("--diffusion-layers", type=int, default=2)
    p.add_argument("--denoise-steps", type=int, default=4)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-train-windows", type=int, default=0)
    p.add_argument("--max-eval-windows", type=int, default=0)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = p.parse_args()
    configuration = CogFuConfig(
        width=args.width, layers=args.layers,
        max_context=args.max_context, max_answer=args.max_answer,
        diffusion_layers=args.diffusion_layers, denoise_steps=args.denoise_steps)
    report = pretrain(
        manifest=args.manifest, output=args.output, config=configuration,
        epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.lr, seed=args.seed, device=args.device,
        initialize_from=args.initialize_from,
        baseline_output=args.baseline_output,
        max_train_windows=args.max_train_windows,
        max_eval_windows=args.max_eval_windows)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
