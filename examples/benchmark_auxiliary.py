"""Measure Auxiliary skill-task accuracy and bounded inference scaling.

Input JSONL records have exactly: ``category``, ``messages``, ``expected``.
Each category can name a distinct evaluation family such as general_skill,
reasoning, memory, planning, or execution_intent. Execution here scores the
generated intent string only; it does not execute live tools.
"""
from __future__ import annotations

from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from statistics import mean
from time import perf_counter
import json
import math

import torch

from ntsa_x.auxiliary import load_checkpoint


def read_cases(path: str | Path) -> list[dict]:
    cases = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL row {number}") from exc
        if (not isinstance(row, dict)
                or set(row) != {"category", "messages", "expected"}
                or not isinstance(row["category"], str)
                or not row["category"].strip()
                or not isinstance(row["messages"], list)
                or not isinstance(row["expected"], str)):
            raise ValueError(f"Invalid benchmark case at row {number}")
        cases.append(row)
    if not cases:
        raise ValueError("Benchmark input contains no cases")
    return cases


def _percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    point = (len(ordered) - 1) * p
    low, high = math.floor(point), math.ceil(point)
    return ordered[low] + (ordered[high] - ordered[low]) * (point - low)


def benchmark(model, cases: list[dict], *, max_new_tokens: int = 128,
              pdc_cycles: int | None = None, adaptive: bool = True,
              seed: int = 0, schedule: str = "serial") -> dict:
    if schedule not in ("serial", "overlap"):
        raise ValueError("schedule must be serial or overlap")
    if schedule == "overlap" and adaptive:
        raise ValueError("overlap benchmark currently uses fixed cycle budgets")
    if not 1 <= max_new_tokens <= model.config.max_answer:
        raise ValueError("max_new_tokens exceeds checkpoint capacity")
    grouped = defaultdict(lambda: {"total": 0, "exact": 0, "latencies": [],
                                   "cycles": [], "residuals": []})
    rows = []
    for index, case in enumerate(cases):
        prefix, _ = model.codec.conversation(
            case["messages"], generation=True,
            max_tokens=model.config.max_context)
        device = model.cortex.embedding.weight.device
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = perf_counter()
        output = bytes(model.draft(
            prefix, max_new_tokens=max_new_tokens, temperature=0.0,
            pdc_cycles=pdc_cycles, adaptive_pdc=adaptive,
            seed=seed + index, pdc_schedule=schedule
        )).decode("utf-8", errors="replace")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = perf_counter() - start
        receipt = model.last_projection_receipt
        exact = output.strip() == case["expected"].strip()
        result = grouped[case["category"]]
        result["total"] += 1
        result["exact"] += int(exact)
        result["latencies"].append(elapsed)
        result["cycles"].append(receipt.cycles if receipt else 0)
        result["residuals"].append(
            receipt.final_relative_residual if receipt else 0.0)
        rows.append({
            "case": index,
            "category": case["category"],
            "exact_match": exact,
            "latency_seconds": elapsed,
            "pdc": receipt.to_dict() if receipt else None,
            "expected": case["expected"],
            "generated": output,
        })
    summary = {}
    for category, values in sorted(grouped.items()):
        latency = values["latencies"]
        summary[category] = {
            "cases": values["total"],
            "exact_match_rate": values["exact"] / values["total"],
            "latency_p50_seconds": _percentile(latency, 0.50),
            "latency_p95_seconds": _percentile(latency, 0.95),
            "mean_pdc_cycles": mean(values["cycles"]),
            "mean_final_projection_residual": mean(values["residuals"]),
        }
    return {
        "architecture": model.architecture,
        "schedule": schedule,
        "cases": len(cases),
        "pdc_mode": "adaptive" if adaptive else "fixed",
        "pdc_cycle_budget": pdc_cycles or model.config.pdc_max_cycles,
        "trainable_parameters": sum(p.numel() for p in model.parameters()),
        "cuda_peak_allocated_bytes": (
            torch.cuda.max_memory_allocated(model.cortex.embedding.weight.device)
            if model.cortex.embedding.weight.is_cuda else None),
        "categories": summary,
        "results": rows,
        "execution_note": "Execution category scores generated intent only; no live tools are invoked.",
    }


def compare_schedules(model, cases: list[dict], *, max_new_tokens: int = 128,
                      pdc_cycles: int | None = None, seed: int = 0) -> dict:
    """Paired serial/overlap comparison at equal fixed PDC cycle count."""
    serial = benchmark(
        model, cases, max_new_tokens=max_new_tokens, pdc_cycles=pdc_cycles,
        adaptive=False, seed=seed, schedule="serial")
    overlap = benchmark(
        model, cases, max_new_tokens=max_new_tokens, pdc_cycles=pdc_cycles,
        adaptive=False, seed=seed, schedule="overlap")
    deltas = {}
    for category in sorted(set(serial["categories"]) & set(overlap["categories"])):
        before = serial["categories"][category]
        after = overlap["categories"][category]
        deltas[category] = {
            "p50_latency_speedup": (
                before["latency_p50_seconds"] / after["latency_p50_seconds"]
                if after["latency_p50_seconds"] > 0 else None),
            "p95_latency_speedup": (
                before["latency_p95_seconds"] / after["latency_p95_seconds"]
                if after["latency_p95_seconds"] > 0 else None),
            "exact_match_delta": (after["exact_match_rate"]
                                  - before["exact_match_rate"]),
            "serial_exact_match_rate": before["exact_match_rate"],
            "overlap_exact_match_rate": after["exact_match_rate"],
        }
    return {
        "architecture": model.architecture,
        "cases": len(cases),
        "comparison": "serial versus overlapped direct-scan/PDC schedule",
        "fixed_pdc_cycles": pdc_cycles or model.config.pdc_max_cycles,
        "serial": serial,
        "overlap": overlap,
        "deltas": deltas,
        "interpretation": (
            "Overlap is an inference schedule ablation. Report latency only "
            "alongside exact-match quality; the PDC branch starts from the "
            "first-symbol state and merges at the scan-completion barrier."),
    }


def main() -> None:
    parser = ArgumentParser(description="Benchmark Auxiliary PDC task transfer")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--pdc-cycles", type=int)
    parser.add_argument("--fixed-cycles", action="store_true")
    parser.add_argument("--schedule", choices=["serial", "overlap"], default="serial")
    parser.add_argument("--compare-schedules", action="store_true",
                        help="Run paired serial and overlap schedules with fixed cycles")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    model = load_checkpoint(args.checkpoint, device=args.device)
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    cases = read_cases(args.cases)
    if args.compare_schedules:
        result = compare_schedules(
            model, cases, max_new_tokens=args.max_new_tokens,
            pdc_cycles=args.pdc_cycles, seed=args.seed)
    else:
        result = benchmark(
            model, cases, max_new_tokens=args.max_new_tokens,
            pdc_cycles=args.pdc_cycles, adaptive=not args.fixed_cycles,
            seed=args.seed, schedule=args.schedule)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    if args.compare_schedules:
        printed = {key: value for key, value in result.items()
                   if key not in {"serial", "overlap"}}
    else:
        printed = {key: value for key, value in result.items()
                   if key != "results"}
    print(json.dumps(printed, indent=2, default=str))


if __name__ == "__main__":
    main()
