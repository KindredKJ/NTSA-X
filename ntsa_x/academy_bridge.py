"""M14: offline, context-aware Academy -> COG-FU supervised-data bridge.

This is a pilot data preparation step. It never loads a model, writes weights,
reads the reserved exam file, or trains a checkpoint.
"""
from __future__ import annotations

from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path
import json

from .academy import LESSONS, audit_lessons
from .native_model import ByteCodec


def _expected_records() -> list[dict]:
    result = []
    for lesson in LESSONS:
        for question, answer in lesson.exercises:
            prompt = (
                f"Subject: {lesson.domain}. Concept: {lesson.concept}. "
                f"Explanation: {lesson.explanation} "
                f"Worked example: {lesson.worked[0]} "
                f"Exercise: {question}"
            )
            result.append({"messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": answer},
            ]})
    return result


def plan(source_dir: str | Path, checkpoint_dir: str | Path) -> dict:
    audit_lessons()
    source = Path(source_dir).expanduser().resolve()
    source_file = source / "train-chat.jsonl"
    check = Path(checkpoint_dir).expanduser().resolve()
    if not source_file.is_file() or not (check / "config.json").is_file():
        raise ValueError("Expected Academy training export and native checkpoint")
    if source_file.stat().st_size > 500_000:
        raise ValueError("Academy pilot corpus exceeds expected size")
    try:
        rows = [json.loads(line) for line in
                source_file.read_text(encoding="utf-8").splitlines()
                if line.strip()]
        info = json.loads((check / "config.json").read_text(encoding="utf-8"))
        config = info["config"]
        if info["architecture"] != "ntsa-x-cog-fu-discrete-v1":
            raise ValueError("Not a COG-FU checkpoint")
        old = int(config["max_context"])
        answer_limit = int(config["max_answer"])
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid Academy export or checkpoint configuration") from exc
    if rows != _expected_records():
        raise ValueError("Academy export differs from reviewed original lessons")
    codec = ByteCodec()
    counts, answer_lengths = [], []
    for row in rows:
        seq, _ = codec.conversation(row["messages"], max_tokens=32768)
        counts.append(len(seq))
        answer_lengths.append(
            len(row["messages"][-1]["content"].encode("utf-8")) + 1)
    needed = max(counts)
    updated = max(old, ((needed + 63) // 64) * 64)
    if updated > 32768 or max(answer_lengths) > answer_limit:
        raise ValueError("Sample exceeds maximum supported architecture bounds")
    if len(rows) != len(LESSONS) * 2:
        raise ValueError("Expected two practice samples per lesson")
    return {
        "curriculum": "kindred-academy-m13-v1",
        "source": str(source_file),
        "source_sha256": sha256(source_file.read_bytes()).hexdigest(),
        "checkpoint": str(check),
        "checkpoint_context": old,
        "required_full_sequence_bytes": needed,
        "suggested_training_context": updated,
        "answer_limit": answer_limit,
        "longest_answer_bytes_including_eos": max(answer_lengths),
        "training_samples": len(LESSONS),
        "development_samples": len(LESSONS),
        "reserved_exam_examples_used": 0,
        "warning": "One demonstration and one development item per concept, "
                   "not an independent test of generalization or language fluency.",
    }


def prepare(source_dir: str | Path, checkpoint_dir: str | Path,
            output_dir: str | Path) -> dict:
    metadata = plan(source_dir, checkpoint_dir)
    dest = Path(output_dir).expanduser().resolve()
    if dest.exists():
        raise FileExistsError("Refusing to overwrite existing M14 prepared data")
    if not dest.parent.is_dir():
        raise ValueError("Create the parent folder before preparing the dataset")
    rows = [json.loads(x) for x in
            Path(metadata["source"]).read_text(encoding="utf-8").splitlines()
            if x.strip()]
    # Alternate within each lesson: fixed held-out development exercise.
    train_rows = rows[::2]
    development_rows = rows[1::2]
    train_prompts = {x["messages"][-2]["content"] for x in train_rows}
    dev_prompts = {x["messages"][-2]["content"] for x in development_rows}
    if train_prompts & dev_prompts:
        raise ValueError("Train and development prompts overlap")
    dest.mkdir()
    try:
        for name, part in [("train.jsonl", train_rows),
                           ("dev.jsonl", development_rows)]:
            (dest / name).write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n"
                        for row in part), encoding="utf-8")
        (dest / "plan.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    except BaseException:
        # Files are created only in our new destination. Leave no misleading
        # partial prepared dataset if write fails.
        for child in dest.iterdir():
            if child.is_file():
                child.unlink()
        dest.rmdir()
        raise
    return {
        **metadata,
        "prepared_dir": str(dest),
        "prepared_sha256": {p.name: sha256(p.read_bytes()).hexdigest()
                            for p in dest.iterdir() if p.is_file()},
    }


def main():
    p = ArgumentParser(description="Prepare small COG-FU Academy pilot; never trains")
    p.add_argument("--academy", default=".local/academy/m13-v1")
    p.add_argument("--checkpoint", default=".local/cog-fu-language-v1")
    p.add_argument("--prepare", metavar="NEW_OUTPUT_DIR")
    args = p.parse_args()
    try:
        outcome = (prepare(args.academy, args.checkpoint, args.prepare)
                   if args.prepare else plan(args.academy, args.checkpoint))
    except (OSError, ValueError) as exc:
        p.error(str(exc))
    print(json.dumps(outcome, indent=2))


if __name__ == "__main__":
    main()
