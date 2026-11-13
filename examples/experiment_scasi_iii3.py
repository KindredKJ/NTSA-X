"""SCASI III.3: prepare or train/evaluate matched COG-FU weight ablation.

No checkpoint is created without --train. Never overwrites an existing
checkpoint, training directory or output report.
"""
from __future__ import annotations
from argparse import ArgumentParser
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
import json
import time

from ntsa_x.scasi_iii_assimilation import (
    evaluate_checkpoint_arms, materialize_datasets, prepare_assimilation,
)
from ntsa_x.scasi_iii_condensation import HeldoutCase, SkillExample


def read_jsonl(path: Path, klass):
    if not path.is_file() or not 0 < path.stat().st_size <= 2_000_000:
        raise ValueError(f"Missing, empty, or oversized JSONL: {path}")
    records = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            records.append(klass(**json.loads(raw)))
        except (TypeError, ValueError, KeyError) as exc:
            raise ValueError(f"Invalid row at {path.name}:{line_no}: {exc}") from exc
    if not 1 <= len(records) <= 256:
        raise ValueError("Must supply 1–256 records")
    return tuple(records)


def run(args) -> dict:
    source = args.examples.expanduser().resolve()
    reserved = args.heldout.expanduser().resolve()
    work = args.workdir.expanduser().resolve()
    if source == reserved:
        raise ValueError("Examples and sealed heldout must differ")
    if work.exists():
        raise FileExistsError(f"Output experiment directory exists: {work}")
    examples = read_jsonl(source, SkillExample)
    cases = read_jsonl(reserved, HeldoutCase)
    dataset = prepare_assimilation(examples, cases)
    if args.train:
        if args.checkpoint is None:
            raise ValueError("--train requires --checkpoint pointing at an existing COG-FU model")
        checkpoint = args.checkpoint.expanduser().resolve()
        if not (checkpoint / "weights.pt").is_file() or not (checkpoint / "config.json").is_file():
            raise ValueError("A real parent COG-FU checkpoint with weights.pt/config.json is required")
        if checkpoint == work or work in checkpoint.parents or checkpoint in work.parents:
            raise ValueError("Training output must not overlap the original checkpoint")
    paths = materialize_datasets(dataset, work)
    result = {
        "schema": "scasi-iii3-experiment-v1",
        "mode": "trained-checkpoint-ablation" if args.train else "preparation-only",
        "examples_sha256": sha256(source.read_bytes()).hexdigest(),
        "heldout_sha256": sha256(reserved.read_bytes()).hexdigest(),
        "evidence_digest": dataset.evidence_digest,
        "skills": [asdict(skill) for skill in dataset.skills],
        "heldout_samples": len(cases),
        "no_heldout_fitting": True,
        "training_byte_difference": dataset.training_byte_difference,
        "model_weight_update_executed": False,
        "effective_intelligence_multiplier": None,
        "datasets": paths,
    }
    if not args.train:
        (work / "preparation.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return result

    from ntsa_x.cog_fu_train import load_checkpoint, read_conversations, train
    import torch
    parent_hash = sha256((checkpoint / "weights.pt").read_bytes()).hexdigest()
    original = load_checkpoint(checkpoint, device=args.device)
    config = original.config
    if not 1 <= args.max_new_tokens <= config.max_answer:
        raise ValueError("Requested response limit exceeds checkpoint capacity")
    # Run native dataset parser before either training arm creates weights.
    for key in ("control_train", "condensed_train", "validation"):
        read_conversations(paths[key], config)
    if args.epochs < 1 or args.epochs > 10 or not 1 <= args.batch_size <= 32:
        raise ValueError("III.3 experiment epochs/batch size exceed budget")
    result["parent_sha256"] = parent_hash
    result["architecture"] = str(original.architecture)
    result["parameter_count"] = sum(p.numel() for p in original.parameters())
    training_reports = {}
    for arm, key in (("control", "control_train"), ("assimilated", "condensed_train")):
        started = time.perf_counter_ns()
        train_report = train(
            train_path=paths[key], eval_path=paths["validation"],
            output=work / f"{arm}_checkpoint", config=config,
            initialize_from=checkpoint, epochs=args.epochs,
            learning_rate=args.learning_rate, batch_size=args.batch_size,
            seed=args.seed, device=args.device)
        training_reports[arm] = {
            "train_report": train_report,
            "duration_seconds": (time.perf_counter_ns() - started) / 1e9,
            "checkpoint_sha256": sha256((work / f"{arm}_checkpoint" / "weights.pt").read_bytes()).hexdigest(),
        }
    if parent_hash != sha256((checkpoint / "weights.pt").read_bytes()).hexdigest():
        raise RuntimeError("Parent checkpoint changed during training")
    models = {
        "original": original,
        "control": load_checkpoint(work / "control_checkpoint", device=args.device),
        "assimilated": load_checkpoint(work / "assimilated_checkpoint", device=args.device),
    }
    with torch.inference_mode():
        comparison = evaluate_checkpoint_arms(
            models, cases, repeats=args.repeats, max_new_tokens=args.max_new_tokens)
    result["model_weight_update_executed"] = True
    result["training"] = training_reports
    result["heldout_evaluation"] = comparison
    result["training_minutes"] = sum(x["duration_seconds"] for x in training_reports.values()) / 60
    result["conclusion"] = (
        "Paired evidence collected. Any neural generalization claim requires external repetition, "
        "confidence intervals, and heldout suites beyond this arithmetic task.")
    destination = work / "evaluation.json"
    with destination.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main():
    parser = ArgumentParser(description="SCASI III.3 native neural assimilation experiment")
    parser.add_argument("--examples", required=True, type=Path)
    parser.add_argument("--heldout", required=True, type=Path)
    parser.add_argument("--workdir", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--train", action="store_true", help="Train separate new COG-FU checkpoints")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
