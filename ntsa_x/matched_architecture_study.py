"""Controlled native recurrent vs causal Transformer small-model study.

Both models train from scratch on the official ARC-AGI-2 PUBLIC TRAINING SPLIT
as *byte-level language modeling*, NOT ARC puzzle solving. No benchmark
evaluation puzzles, attached foundation models, or external paid APIs are used.
Matching parameter counts and token exposure does NOT mean matching FLOPs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from ntsa_x.native_model import NativeConfig, NativeOrbitModel


class CausalByteTransformer(nn.Module):
    def __init__(self, width: int, sequence_length: int):
        super().__init__()
        if width % 4 or width < 16:
            raise ValueError("head-aligned transformer width required")
        self.embedding = nn.Embedding(262, width)
        self.positions = nn.Embedding(sequence_length, width)
        self.encoder = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=4,
            dim_feedforward=3*width,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, 262)

    def forward(self, tokens):
        batch, length = tokens.shape
        positions = torch.arange(length, device=tokens.device)
        x = self.embedding(tokens) + self.positions(positions).unsqueeze(0)
        # Explicit causal attention; no token can inspect future answers.
        mask = torch.ones(length, length, dtype=torch.bool, device=tokens.device).triu(1)
        x = self.encoder(x, src_mask=mask)
        return self.head(self.norm(x))


def native_without_recurrent_memory(model: NativeOrbitModel, tokens):
    """Ablation of trained native weights: zero X/Y/Z state BEFORE EACH token."""
    batch, length = tokens.shape
    logits = []
    embedding = model.embedding(tokens)
    for step in range(length):
        x = embedding[:, step, :]
        for layer in model.cells:
            zero = torch.zeros(batch, 3, model.config.width,
                               dtype=x.dtype, device=x.device)
            x, _ = layer(x, zero)
        logits.append(model.head(model.norm(x)))
    return torch.stack(logits, dim=1)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _data(training_directory: Path):
    files = sorted(training_directory.glob("*.json"))
    if len(files) != 1000:
        raise ValueError("expected exactly 1000 official ARC-AGI-2 training tasks")
    # The split is deterministic before any model/validation result is inspected.
    train_files, held_files = files[:850], files[850:]
    def pack(selected):
        chunks = []
        dig = hashlib.sha256()
        for path in selected:
            obj = json.loads(path.read_text(encoding="utf-8"))
            if not (isinstance(obj, dict) and "train" in obj and "test" in obj):
                raise ValueError("invalid ARC training task")
            raw = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
            dig.update(path.name.encode() + b"\x00" + raw + b"\x00")
            chunks.append(raw + b"\n")
        return b"".join(chunks), dig.hexdigest()
    training, training_hash = pack(train_files)
    heldout, heldout_hash = pack(held_files)
    return training, heldout, training_hash, heldout_hash


def _windows(data: bytes, seed: int, steps: int, batch_size: int,
             sequence_length: int):
    rng = random.Random(seed)
    if len(data) < sequence_length + 2:
        raise ValueError("not enough data for sequence windows")
    for _ in range(steps):
        starts = [rng.randrange(len(data) - sequence_length) for _ in range(batch_size)]
        yield torch.tensor(
            [list(data[i:i+sequence_length]) for i in starts],
            dtype=torch.long,
        )


def token_loss(logits, tokens):
    return F.cross_entropy(logits[:, :-1, :].reshape(-1, 262),
                           tokens[:, 1:].reshape(-1))


def _eval(model, batches, *, no_memory=False):
    model.eval()
    with torch.inference_mode():
        return statistics.mean(
            float(token_loss(
                native_without_recurrent_memory(model, tokens)
                if no_memory else (model(tokens)[0] if isinstance(model, NativeOrbitModel)
                                   else model(tokens)), tokens).item())
            for tokens in batches
        ) / math.log(2)


def _train(model, inputs, *, learning_rate: float):
    opt = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    model.train()
    start = time.perf_counter()
    for tokens in inputs:
        opt.zero_grad(set_to_none=True)
        output = model(tokens)
        logits = output[0] if isinstance(model, NativeOrbitModel) else output
        loss = token_loss(logits, tokens)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return time.perf_counter() - start


def run(training_dir: Path, *, seeds=(11, 29), steps=30,
        sequence_length=48, batch_size=2, heldout_batches=12):
    if steps < 1 or steps > 200 or not 16 <= sequence_length <= 128 or not 1 <= batch_size <= 16:
        raise ValueError("invalid fixed research budget")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("nonempty unique seed list required")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    data, val, train_digest, val_digest = _data(training_dir)
    template = NativeOrbitModel(NativeConfig(width=32, layers=1,
                                               dropout=0.0, max_context=sequence_length))
    native_params = count_parameters(template)
    candidate_widths = range(16, 129, 4)
    transformer_width = min(
        candidate_widths,
        key=lambda w: abs(count_parameters(CausalByteTransformer(w, sequence_length)) - native_params),
    )
    matched_transformer_params = count_parameters(CausalByteTransformer(transformer_width, sequence_length))
    delta_percent = abs(native_params-matched_transformer_params) / native_params * 100
    if delta_percent > 10:
        raise RuntimeError("parameter budgets not comparable within 10 percent")
    records = []
    for seed in seeds:
        train_batches = tuple(_windows(data, seed+1000, steps, batch_size, sequence_length))
        eval_batches = tuple(_windows(val, seed+2000, heldout_batches, batch_size, sequence_length))
        torch.manual_seed(seed)
        native = NativeOrbitModel(NativeConfig(width=32, layers=1, dropout=0.0,
                                               max_context=sequence_length))
        native_seconds = _train(native, train_batches, learning_rate=0.001)
        native_bpb = _eval(native, eval_batches)
        native_memory_ablation_bpb = _eval(native, eval_batches, no_memory=True)
        torch.manual_seed(seed)
        transformer = CausalByteTransformer(transformer_width, sequence_length)
        transformer_seconds = _train(transformer, train_batches, learning_rate=0.001)
        transformer_bpb = _eval(transformer, eval_batches)
        records.append({
            "seed": seed,
            "native_train_cpu_wall_seconds": native_seconds,
            "transformer_train_cpu_wall_seconds": transformer_seconds,
            "native_validation_bits_per_byte": native_bpb,
            "native_without_recurrent_memory_bits_per_byte": native_memory_ablation_bpb,
            "transformer_validation_bits_per_byte": transformer_bpb,
            "parameter_count_native": native_params,
            "parameter_count_transformer": matched_transformer_params,
        })
    return {
        "experiment": "NTSA_X_NATIVE_ORBIT_VS_TINY_CAUSAL_TRANSFORMER_V1",
        "status": "SELF_EXECUTED_EXTERNAL_TRAINING_DATA_NOT_OFFICIAL_ARC_EVAL",
        "task": "next-byte prediction of official public ARC training task JSON",
        "models": ["NTSA-X NativeOrbitModel one layer, width 32", "from-scratch causal Transformer"],
        "official_arc_puzzle_accuracy": None,
        "trained_agi_claim": False,
        "external_peer_review_completed": False,
        "model_comparison_is_parameter_matched": True,
        "model_comparison_is_exact_flop_matched": False,
        "token_exposure_matched": True,
        "same_optimizer_and_hardware": True,
        "parameter_delta_percent": delta_percent,
        "protocol": {"training_files":850,"heldout_files":150,"training_sha256":train_digest,
                     "heldout_sha256":val_digest,"seeds":list(seeds),"steps":steps,
                     "sequence_length":sequence_length,"batch_size":batch_size,
                     "heldout_batches":heldout_batches,"optimizer":"AdamW lr0.001 wd0.01",
                     "cpu_threads":1,"native_width":32,
                     "transformer_width":transformer_width},
        "hardware":platform.platform(),
        "pytorch_version":torch.__version__,
        "utc":datetime.now(timezone.utc).isoformat(),
        "mean_native_bpb":statistics.mean(r["native_validation_bits_per_byte"] for r in records),
        "mean_native_no_memory_bpb":statistics.mean(r["native_without_recurrent_memory_bits_per_byte"] for r in records),
        "mean_transformer_bpb":statistics.mean(r["transformer_validation_bits_per_byte"] for r in records),
        "per_seed":records,
        "limitations":[
            "This measures next-byte loss on ARC training JSON, not abstract reasoning or puzzle solutions.",
            "Native recurrent model and causal Transformer are parameter/token matched within 10%, NOT exact-FLOP matched.",
            "Wall time is measured, not controlled; architecture efficiency claims require energy/FLOP profiling and independent runs.",
            "The validation subset was chosen from public training data, not an independently hidden dataset.",
            "No model weights or checkpoint from this experiment are deployed for public use.",
        ],
    }


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument("--training-dir",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--steps",type=int,default=30)
    args=parser.parse_args(argv)
    report=run(args.training_dir,steps=args.steps)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_bytes((json.dumps(report,sort_keys=True,indent=2,default=str)+"\n").encode())
    print(json.dumps({k:v for k,v in report.items() if k!="per_seed"},indent=2,default=str))


if __name__=="__main__":
    main()
