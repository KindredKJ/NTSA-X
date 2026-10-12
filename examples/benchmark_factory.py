"""Measured throughput harness for a batch of deterministic factory tasks.

This measures fixed-function local worker jobs on the current machine.
It does not measure AI-model inference tokens/sec, intelligence or energy.
"""
import argparse
import json
from pathlib import Path
from platform import platform, python_version
from tempfile import TemporaryDirectory

from ntsa_x.factory import FactoryTask, ProductionFactory
from ntsa_x.nexus import NexusStore


def measure(tasks, workers, batch_id):
    # A fresh store prevents cache/reuse from contaminating comparisons.
    with NexusStore(":memory:") as store:
        result = ProductionFactory(store, workers=workers).run(tasks, batch_id=batch_id)
        if result.completed != len(tasks) or result.failed or result.blocked:
            raise RuntimeError("Benchmark verification failed")
        if not all(store.verify(receipt.digest) for receipt in result.receipts):
            raise RuntimeError("Benchmark lineage verification failed")
        return result.to_dict()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=int, default=100)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--compare", action="store_true",
                        help="Also measure sequential worker=1 baseline")
    parser.add_argument("--output", type=Path, default=None,
                        help="Save a JSON report to an explicit path")
    args = parser.parse_args()
    if not 1 <= args.tasks <= 1000:
        parser.error("--tasks must be 1 to 1000")
    if not 1 <= args.workers <= 100:
        parser.error("--workers must be 1 to 100")
    tasks = [
        FactoryTask(task_id=f"bench-{index:04d}",
                    operation="increment", amount=2, seed=index)
        for index in range(args.tasks)
    ]
    baseline = measure(tasks, 1, "baseline") if args.compare else None
    parallel = measure(tasks, args.workers, "parallel")
    ratio = (
        round(baseline["elapsed_seconds"] / parallel["elapsed_seconds"], 3)
        if baseline is not None and parallel["elapsed_seconds"] else None
    )
    report = {
        "schema_version": "ntsa-factory-throughput/v1",
        "environment": {
            "os": platform(),
            "python": python_version(),
        },
        "workload": "local arithmetic child-process invocation; independent local oracle; Nexus SHA256 commit",
        "target": "100-task production capacity; measured speedup is environment-dependent",
        "baseline": baseline,
        "parallel": parallel,
        "observed_speedup_vs_sequential": ratio,
        "note": "Not a trained AI model performance benchmark; not a claim of 100x speedup.",
    }
    output = json.dumps(report, indent=2, sort_keys=True)
    print(output)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
