"""Reproducible Persistent Scalar vs per-task process throughput measurement.

Synthetic arithmetic only. Every output is checked by a separate local oracle
and linked to Nexus. Does not measure AI-model inference or energy usage.
"""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
from platform import platform, python_version
from statistics import mean, median, pstdev
from time import perf_counter

from ntsa_x.factory import FactoryTask, ProductionFactory
from ntsa_x.nexus import NexusStore
from ntsa_x.plugins import PluginManifest, PluginRegistry
from ntsa_x.persistent_workers import PersistentPluginWorker


def one_run(tasks, concurrency, persistent):
    with ExitStack() as stack:
        store = stack.enter_context(NexusStore(":memory:"))
        worker = None
        startup_time = 0.0
        if persistent:
            registry = PluginRegistry()
            registry.register(PluginManifest(
                "ntsa.factory.scalar", "1.0.0",
                ("increment", "decrement", "identity", "negate"),
            ))
            begin = perf_counter()
            worker = stack.enter_context(PersistentPluginWorker(
                registry, workers=concurrency, timeout=10,
                max_tasks_per_worker=max(len(tasks) + 1, 1000),
            ))
            startup_time = perf_counter() - begin
        factory = ProductionFactory(store, workers=concurrency, plugin_worker=worker)
        run = factory.run(tasks, batch_id="benchmark")
        if run.completed != len(tasks) or run.failed or run.blocked:
            raise RuntimeError("Benchmark failed independent verification")
        if not all(store.verify(receipt.digest) for receipt in run.receipts):
            raise RuntimeError("Nexus digest verification failed")
        return {
            "startup_seconds": round(startup_time, 6),
            "execution_seconds": round(run.elapsed_seconds, 6),
            "end_to_end_seconds": round(startup_time + run.elapsed_seconds, 6),
            "verified_tasks": run.completed,
            "tasks_per_second": round(run.tasks_per_second, 3),
            "peak_in_flight": run.peak_in_flight,
            "worker_process_starts": worker.stats["starts"] if worker else len(tasks),
            "worker_process_restarts": worker.stats["restarts"] if worker else 0,
        }


def stats(trials):
    throughputs = [t["tasks_per_second"] for t in trials]
    elapsed = [t["execution_seconds"] for t in trials]
    return {
        "trials": trials,
        "median_tasks_per_second": round(median(throughputs), 3),
        "mean_tasks_per_second": round(mean(throughputs), 3),
        "stddev_tasks_per_second": round(pstdev(throughputs), 3),
        "median_execution_seconds": round(median(elapsed), 6),
        "median_end_to_end_seconds": round(median(t["end_to_end_seconds"] for t in trials), 6),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=int, default=100)
    parser.add_argument("--workers", default="1,4,8,16")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    try:
        workers = sorted({int(x.strip()) for x in args.workers.split(",")})
    except ValueError:
        parser.error("workers must be comma-separated integers")
    if not 1 <= args.tasks <= 1000 or not 1 <= args.repeats <= 10:
        parser.error("tasks must be 1..1000 and repeats must be 1..10")
    if not workers or any(not 1 <= worker <= 100 for worker in workers):
        parser.error("worker counts must be 1..100")
    tasks = [
        FactoryTask(f"task-{i:04d}", "increment", 2, i)
        for i in range(args.tasks)
    ]
    # Warmup runs aren't incorporated into measurements.
    one_run(tasks[:min(5, len(tasks))], 1, False)
    one_run(tasks[:min(5, len(tasks))], 1, True)

    baseline_trials = [one_run(tasks, 1, False) for _ in range(args.repeats)]
    baseline = stats(baseline_trials)
    persistent = {}
    for count in workers:
        group = stats([one_run(tasks, count, True) for _ in range(args.repeats)])
        baseline_rate = baseline["median_tasks_per_second"]
        group["median_speedup_vs_oneshot_single"] = (
            round(group["median_tasks_per_second"] / baseline_rate, 3)
            if baseline_rate else None
        )
        persistent[str(count)] = group
    recommended_warm = max(
        workers, key=lambda n: persistent[str(n)]["median_tasks_per_second"]
    )
    recommended_cold = min(
        workers, key=lambda n: persistent[str(n)]["median_end_to_end_seconds"]
    )
    report = {
        "schema_version": "ntsa-persistent-scalar-benchmark/v1",
        "environment": {"platform": platform(), "python": python_version()},
        "workload": "synthetic scalar subprocess tasks, independent local check, Nexus verification",
        "tasks": args.tasks, "repeats": args.repeats,
        "baseline_oneshot_single": baseline,
        "persistent_scalar": persistent,
        "recommended_workers_warm_throughput": recommended_warm,
        "recommended_workers_cold_start_100_tasks": recommended_cold,
        "recommendation_note": "Specific to this measured synthetic workload, not general automatic hardware tuning.",
        "warning": "Not a trained AI-model benchmark, 100x claim, or measured joules.",
    }
    rendered = json.dumps(report, sort_keys=True, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
