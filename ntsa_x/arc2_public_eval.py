"""Evaluate a frozen, deterministic NTSA-X ARC rule induction policy on real ARC-AGI-2 public tasks.

The official ARC Prize public evaluation set is downloaded by the workflow. This
is NOT an ARC Prize private leaderboard score, trained foundation model result,
or evidence of AGI. Ground truth outputs are withheld from the prediction call.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import time
from pathlib import Path


def grid_ok(value):
    return (isinstance(value, list) and 1 <= len(value) <= 30
            and all(isinstance(r, list) and 1 <= len(r) <= 30 and len(r) == len(value[0])
                    and all(type(v) is int and 0 <= v <= 9 for v in r) for r in value))


def transform(grid, name):
    if name == "identity":
        return [r[:] for r in grid]
    if name == "rot180":
        return [list(reversed(row)) for row in reversed(grid)]
    if name == "flip_h":
        return [list(reversed(row)) for row in grid]
    if name == "flip_v":
        return [row[:] for row in reversed(grid)]
    if name == "transpose":
        return [list(row) for row in zip(*grid)]
    if name == "rotate90":
        return [list(row) for row in zip(*reversed(grid))]
    if name == "rotate270":
        return [list(row) for row in reversed(list(zip(*grid)))]
    raise ValueError("unknown transform")


TRANSFORMS = ("identity", "flip_h", "flip_v", "rot180", "transpose", "rotate90", "rotate270")


def learn_mapping(training, geometry):
    """Infer a consistent function from each observed source color to target color."""
    mapping = {}
    for pair in training:
        inp = transform(pair["input"], geometry)
        out = pair["output"]
        if len(inp) != len(out) or len(inp[0]) != len(out[0]):
            return None
        for row_in, row_out in zip(inp, out):
            for source, target in zip(row_in, row_out):
                if source in mapping and mapping[source] != target:
                    return None
                mapping[source] = target
    # No evidence for unseen colors; preserve them, never infer unseen outputs.
    return mapping


def apply_mapping(grid, mapping):
    return [[mapping.get(v, v) for v in row] for row in grid]


def candidates(training):
    if not training or any(not grid_ok(pair.get("input")) or not grid_ok(pair.get("output"))
                           for pair in training):
        return [("identity", {})]
    proposals = []
    for geometry in TRANSFORMS:
        cmap = learn_mapping(training, geometry)
        if cmap is None:
            continue
        if all(apply_mapping(transform(p["input"], geometry), cmap) == p["output"]
               for p in training):
            # Prefer fewer transformed colors and a simple geometry.
            changes = sum(source != target for source, target in cmap.items())
            proposals.append((changes, TRANSFORMS.index(geometry), geometry, cmap))
    proposals.sort(key=lambda item: (item[0], item[1]))
    return [(geo, mapping) for _, _, geo, mapping in proposals] or [("identity", {})]


def predict(train, input_grid, baseline=False):
    if not grid_ok(input_grid):
        raise ValueError("malformed test input")
    if not isinstance(train, list) or len(train) > 100:
        raise ValueError("malformed training examples")
    if baseline:
        selected = [("identity", {}), ("flip_h", {})]
    else:
        selected = candidates(train)
    guesses = []
    for geometry, cmap in selected:
        proposal = apply_mapping(transform(input_grid, geometry), cmap)
        if proposal not in guesses:
            guesses.append(proposal)
        if len(guesses) == 2:
            break
    for extra in ("identity", "flip_h", "flip_v", "rot180"):
        if len(guesses) == 2:
            break
        proposal = transform(input_grid, extra)
        if proposal not in guesses:
            guesses.append(proposal)
    if len(guesses) == 1:
        guesses.append([row[:] for row in guesses[0]])
    return guesses[:2]


def evaluate(path: Path, *, max_tasks: int = 120):
    files = sorted(path.glob("*.json"))
    if not files:
        raise ValueError("no real ARC tasks found")
    if len(files) != 120:
        raise ValueError("expected official 120-task public ARC-AGI-2 evaluation split")
    if max_tasks != 120:
        raise ValueError("do not selectively sample a public-eval split")
    records = []
    started = time.perf_counter()
    for file in files:
        raw = file.read_bytes()
        task = json.loads(raw)
        train = task.get("train")
        tests = task.get("test")
        if not isinstance(train, list) or not isinstance(tests, list) or not tests:
            raise ValueError("malformed ARC task")
        if any(not grid_ok(x.get("input")) or not grid_ok(x.get("output"))
               for x in train + tests):
            raise ValueError("invalid ARC grid")
        pair_results = []
        for test in tests:
            # IMPORTANT: test output is never supplied to either prediction call.
            inp = test["input"]
            agent = predict(train, inp)
            ablation = predict([], inp, baseline=True)
            target = test["output"]
            pair_results.append({
                "agent_solved": any(g == target for g in agent),
                "ablation_solved": any(g == target for g in ablation),
                "two_attempts_each": True,
            })
        records.append({
            "task_id": file.stem,
            "task_sha256": hashlib.sha256(
                json.dumps(task, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "agent_score": sum(p["agent_solved"] for p in pair_results)/len(pair_results),
            "ablation_score": sum(p["ablation_solved"] for p in pair_results)/len(pair_results),
            "test_pair_count": len(pair_results),
        })
    elapsed = time.perf_counter() - started
    primary = statistics.mean(r["agent_score"] for r in records)
    ablation = statistics.mean(r["ablation_score"] for r in records)
    return {
        "benchmark": "ARC-AGI-2",
        "dataset": "arcprize/ARC-AGI-2/data/evaluation",
        "split": "PUBLIC_EVALUATION",
        "task_count": len(records),
        "attempts_per_pair": 2,
        "primary_system": "NTSA-X fixed-function rule-induction research adapter (not trained model)",
        "ablation_system": "fixed identity/flip-h two-attempt reference",
        "status": "LOCAL_EXECUTION_OFFICIAL_PUBLIC_DATASET_NOT_OFFICIAL_LEADERBOARD",
        "official_private_score": False,
        "no_hidden_answer_use": True,
        "fully_autonomous_run": True,
        "interactive_agent": False,
        "model_training_during_evaluation": False,
        "model_provider": None,
        "primary_mean_fractional_task_score": primary,
        "ablation_mean_fractional_task_score": ablation,
        "difference_percentage_points": (primary - ablation) * 100,
        "wall_seconds_all_modes": elapsed,
        "hardware": platform.platform(),
        "task_sha256_aggregate": hashlib.sha256(
            json.dumps([(r["task_id"], r["task_sha256"]) for r in records],
                       sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "per_task": records,
        "limitations": [
            "Public evaluation tasks are not the competition's semi-private or private split.",
            "Native ARC solver is deterministic fixed-function program induction, not Kindred¤ neural foundation-model inference.",
            "This is not ARC-AGI-3 and cannot establish interactive planning capability.",
            "Identical task budgets and two guesses do not imply comparable neural compute costs.",
            "Scores are local, not verified or published by ARC Prize; inspect source and logs.",
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = evaluate(args.dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes((json.dumps(report, indent=2, sort_keys=True) + "\n").encode())
    print(json.dumps({k: v for k, v in report.items() if k != "per_task"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
