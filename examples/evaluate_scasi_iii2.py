"""SCASI III.2: runnable held-out evaluator for a real native NTSA-X checkpoint.

Example:
python -m examples.evaluate_scasi_iii2 --checkpoint /absolute/checkpoint \
    --examples /absolute/reviewed-lessons.jsonl --heldout /absolute/private-exam.jsonl
"""
from __future__ import annotations
from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path
import json
import time

from ntsa_x.scasi_iii_condensation import (
    HeldoutCase, SkillBank, SkillExample, benchmark_native_model,
    induce_affine_skill)


def read_jsonl(path: Path, klass):
    if not path.is_file() or path.stat().st_size > 2_000_000:
        raise ValueError("Missing or excessive dataset size")
    records = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if raw.strip():
            records.append(klass(**json.loads(raw)))
    if not records or len(records) > 256:
        raise ValueError("Dataset must contain 1–256 records")
    return records


def main() -> None:
    parser = ArgumentParser(description="Evaluate native NTSA-X/SCASI III.2 held-out response accuracy")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--examples", required=True, type=Path)
    parser.add_argument("--heldout", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.examples.resolve() == args.heldout.resolve():
        parser.error("Holdout and training datasets must be different files")
    if not (args.checkpoint / "weights.pt").is_file():
        parser.error("Valid existing COG-FU weights.pt checkpoint required")
    train = read_jsonl(args.examples, SkillExample)
    holdout = read_jsonl(args.heldout, HeldoutCase)
    prompts = {r.prompt for r in holdout}
    numeric_holdout = set()
    for item in holdout:
        parts = item.prompt.split(" ")
        if len(parts) == 3 and parts[0] == "compute":
            numeric_holdout.add((parts[1], int(parts[2])))
    if {(r.skill_id, r.input_value) for r in train} & numeric_holdout:
        parser.error("Train/validation and heldout input overlap")
    started = time.perf_counter_ns()
    groups = {}
    for row in train:
        groups.setdefault(row.skill_id, []).append(row)
    with SkillBank(writable=True) as bank:
        for group in groups.values():
            bank.promote(induce_affine_skill(group))
        preparation_ns = time.perf_counter_ns() - started
        from ntsa_x.cog_fu_train import load_checkpoint
        model = load_checkpoint(args.checkpoint, device=args.device)
        result = benchmark_native_model(model, holdout, bank=bank,
                                        repeats=args.repeats,
                                        max_new_tokens=args.max_new_tokens,
                                        preparation_ns=preparation_ns)
    result["checkpoint_verified"] = True  # checkpoint format and weights successfully loaded; not AI quality
    result["checkpoint_sha256"] = sha256((args.checkpoint / "weights.pt").read_bytes()).hexdigest()
    result["holdout_sha256"] = sha256(args.heldout.read_bytes()).hexdigest()
    result["example_sha256"] = sha256(args.examples.read_bytes()).hexdigest()
    result["training_heldout_overlap_checked"] = True
    result["checkpoint_verified_meaning"] = "File hash and native load only, not factual/model quality"
    result["evaluation_kind"] = "real-loaded-native-model"
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
