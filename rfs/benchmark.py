"""Reproducible RFS CPU-work reuse benchmark; no model/AI performance claims.

Execute: python -m rfs.benchmark --unique 12 --repeats 4 --iterations 50000
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import statistics
import tempfile
import time
from pathlib import Path

from .engine import RecallFirstEngine, WorkSpec


def heavy_work(args):
    result = hashlib.pbkdf2_hmac(
        "sha256", args["payload"].encode(), args["salt"].encode(),
        args["iterations"])
    return {"digest": result.hex()}


def quantile(values, percent):
    ordered = sorted(values)
    return round(ordered[max(0, math.ceil(len(ordered) * percent) - 1)], 3)


def run(unique=12, repeats=4, iterations=50000):
    if unique < 1 or repeats < 2 or iterations < 1:
        raise ValueError("Need unique>=1, repeats>=2, iterations>=1")
    jobs = [{"payload": f"recall-work-{i}", "salt": f"isolated-{i}",
             "iterations": iterations} for i in range(unique)]
    t0 = time.perf_counter()
    expected = {args["payload"]: heavy_work(args) for args in jobs}
    setup_ms = round((time.perf_counter() - t0) * 1000, 3)

    def verify(args, answer, _ctx):
        return hmac.compare_digest(
            str(answer.get("digest", "")),
            expected[args["payload"]]["digest"])

    baseline_ms = []
    t0 = time.perf_counter()
    for _ in range(repeats):
        for args in jobs:
            t = time.perf_counter_ns()
            result = heavy_work(args)
            assert verify(args, result, {})
            baseline_ms.append((time.perf_counter_ns() - t) / 1e6)
    baseline_total = (time.perf_counter() - t0) * 1000

    with tempfile.TemporaryDirectory(prefix="rfs-bench-") as directory:
        specs = [
            WorkSpec(
                namespace="rfs-benchmark", tenant="local-benchmark",
                operation="pbkdf2-hmac", arguments=args,
                context={"fixture_revision": "v1"},
                implementation_revision="pbkdf2-sha256-v1",
                verifier_revision="fixture-v1", policy_revision="benchmark-v1",
                ttl_seconds=3600,
            )
            for args in jobs
        ]
        recalled_ms = []
        with RecallFirstEngine(Path(directory) / "bench.rfs.sqlite3") as engine:
            t0 = time.perf_counter()
            for _ in range(repeats):
                for req in specs:
                    t = time.perf_counter_ns()
                    response = engine.execute(req, heavy_work, verify)
                    assert verify(req.arguments, response.value, req.context)
                    recalled_ms.append((time.perf_counter_ns() - t) / 1e6)
            recall_total = (time.perf_counter() - t0) * 1000
            metrics = engine.metrics()

    return {
        "category": "KINDRED-RFS-001",
        "workload": "deterministic PBKDF2 CPU work with precomputed fixture verifier",
        "limitations": "Synthetic exact-repeat benchmark; NOT neural inference or training; external verification and fixture setup are not free.",
        "unique": unique, "repeats": repeats, "iterations": iterations,
        "request_count": len(baseline_ms),
        "fixture_setup_ms": setup_ms,
        "baseline_total_ms": round(baseline_total, 3),
        "rfs_total_ms": round(recall_total, 3),
        "relative_speedup_for_workload": round(baseline_total / recall_total, 3) if recall_total else None,
        "baseline_p50_ms": round(statistics.median(baseline_ms), 3),
        "baseline_p95_ms": quantile(baseline_ms, .95),
        "baseline_p99_ms": quantile(baseline_ms, .99),
        "rfs_p50_ms": round(statistics.median(recalled_ms), 3),
        "rfs_p95_ms": quantile(recalled_ms, .95),
        "rfs_p99_ms": quantile(recalled_ms, .99),
        "cache_metrics": metrics,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unique", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=50000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(args.unique, args.repeats, args.iterations)
    content = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding="utf-8")
    print(content, end="")


if __name__ == "__main__":
    main()
