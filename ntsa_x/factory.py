"""NTSA-X Factory: bounded DAG scheduling for up to 100 parallel fixed workers.

This is an experimental *task throughput* pipeline, not 100x model speed.
Only the repository's trusted fixed-function arithmetic worker is allowed.
Every committed output is independently recomputed by a local arithmetic oracle.
No external effects, arbitrary plugin source, remote calls, or trained model.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from math import isfinite
from time import perf_counter
from typing import Sequence

from .convergence import Check, ConvergenceEngine, REQUIRED_CHECKS
from .events import EventStream
from .nexus import NexusStore
from .plugins import PluginManifest, PluginRegistry, PluginWorker

OPERATIONS = ("increment", "decrement", "identity", "negate")
MAX_SCALAR = 1_000_000_000


@dataclass(frozen=True)
class FactoryTask:
    task_id: str
    operation: str = "increment"
    amount: int = 1
    seed: int = 0
    dependencies: tuple[str, ...] = ()
    use_dependency_sum: bool = False

    def __post_init__(self):
        if not isinstance(self.task_id, str) or not 0 < len(self.task_id) <= 128:
            raise ValueError("task_id must be a short nonempty string")
        if self.operation not in OPERATIONS:
            raise ValueError("Only fixed-function scalar operations are permitted")
        if type(self.amount) is not int or not 0 <= self.amount <= 1_000_000:
            raise ValueError("amount must be an integer in [0, 1000000]")
        if type(self.seed) is not int or abs(self.seed) > MAX_SCALAR:
            raise ValueError("seed is outside allowed range")
        if not isinstance(self.dependencies, tuple) or any(
            not isinstance(d, str) or not d for d in self.dependencies
        ):
            raise ValueError("dependencies must be a tuple of task IDs")
        if len(set(self.dependencies)) != len(self.dependencies):
            raise ValueError("Repeated dependency")
        if type(self.use_dependency_sum) is not bool:
            raise ValueError("use_dependency_sum must be boolean")
        if self.use_dependency_sum and not self.dependencies:
            raise ValueError("Dependency sum requires at least one parent")


@dataclass(frozen=True)
class FactoryReceipt:
    task_id: str
    status: str
    value: int | None
    digest: str | None
    reason: str


@dataclass(frozen=True)
class FactoryRun:
    batch_id: str
    receipts: tuple[FactoryReceipt, ...]
    elapsed_seconds: float
    workers: int
    peak_in_flight: int

    @property
    def completed(self):
        return sum(r.status == "committed" for r in self.receipts)

    @property
    def failed(self):
        return sum(r.status == "failed" for r in self.receipts)

    @property
    def blocked(self):
        return sum(r.status == "blocked" for r in self.receipts)

    @property
    def tasks_per_second(self):
        return self.completed / self.elapsed_seconds if self.elapsed_seconds else 0.0

    def to_dict(self):
        return {
            "batch_id": self.batch_id, "total": len(self.receipts),
            "committed": self.completed, "failed": self.failed,
            "blocked": self.blocked, "workers": self.workers,
            "peak_in_flight": self.peak_in_flight,
            "elapsed_seconds": round(self.elapsed_seconds, 6),
            "committed_tasks_per_second": round(self.tasks_per_second, 3),
        }


def _validate_dag(tasks):
    ids = {t.task_id for t in tasks}
    if len(ids) != len(tasks):
        raise ValueError("Duplicate task IDs")
    for task in tasks:
        if task.task_id in task.dependencies or any(dep not in ids for dep in task.dependencies):
            raise ValueError("Self-referential or unknown dependency")
    visited, visiting = set(), set()
    index = {t.task_id: t for t in tasks}
    def visit(key):
        if key in visiting:
            raise ValueError("Cyclic task dependency")
        if key in visited:
            return
        visiting.add(key)
        for dep in index[key].dependencies:
            visit(dep)
        visiting.remove(key)
        visited.add(key)
    for task in tasks:
        visit(task.task_id)


def _expected(operation, value, amount):
    if operation == "increment":
        return value + amount
    if operation == "decrement":
        return value - amount
    if operation == "negate":
        return -value
    if operation == "identity":
        return value
    raise ValueError("Unsupported operation")


class ProductionFactory:
    """DAG coordinator with explicit capacity and fail-closed verified commits.

    Does not provide hostile-code sandboxing or guarantee speedups. CPU-heavy
    tasks should use actual process pools, native operators, or remote workers.
    """

    def __init__(
        self, store: NexusStore, *, workers: int = 8,
        max_batch: int = 1000, worker_timeout: float = 5,
        stream: EventStream | None = None, plugin_worker: PluginWorker | None = None,
    ):
        if not isinstance(store, NexusStore):
            raise TypeError("NexusStore is required")
        if type(workers) is not int or not 1 <= workers <= 100:
            raise ValueError("workers must be in [1,100]")
        if type(max_batch) is not int or not 1 <= max_batch <= 10000:
            raise ValueError("max_batch must be in [1,10000]")
        self.store = store
        self.workers = workers
        self.max_batch = max_batch
        self.stream = stream if stream is not None else EventStream()
        if plugin_worker is None:
            registry = PluginRegistry()
            registry.register(PluginManifest("ntsa.factory.scalar", "1.0.0", OPERATIONS))
            plugin_worker = PluginWorker(registry, timeout=worker_timeout)
        if not isinstance(plugin_worker, PluginWorker):
            raise TypeError("plugin_worker must be a registered PluginWorker")
        self.worker = plugin_worker
        self.verifier = ConvergenceEngine()

    def _compute(self, task, input_value):
        result = self.worker.invoke("ntsa.factory.scalar", "1.0.0", task.operation,
                                    {"value": input_value, "amount": task.amount})
        if type(result.get("value")) is not int:
            raise ValueError("Invalid worker result")
        output = result["value"]
        expected = _expected(task.operation, input_value, task.amount)
        checks = [
            Check(category, output == expected, "local-independent-arithmetic-oracle",
                  external=(category == "evidence"))
            for category in REQUIRED_CHECKS
        ]
        report = self.verifier.evaluate(checks, forward={"value": output},
                                        reverse={"value": expected})
        if not report.converged:
            raise ValueError("Independent local arithmetic verification failed")
        return output, report

    def run(self, tasks: Sequence[FactoryTask], *, batch_id: str) -> FactoryRun:
        if not isinstance(batch_id, str) or not 0 < len(batch_id) <= 128:
            raise ValueError("Invalid batch ID")
        tasks = tuple(tasks)
        if not 0 < len(tasks) <= self.max_batch:
            raise ValueError("Batch size exceeds configured capacity")
        if not all(isinstance(task, FactoryTask) for task in tasks):
            raise TypeError("Only FactoryTask instances allowed")
        _validate_dag(tasks)
        start = perf_counter()
        receipts: dict[str, FactoryReceipt] = {}
        in_flight = {}
        peak = 0
        self.stream.emit("factory.start", batch_size=len(tasks), max_parallel=self.workers)
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            while len(receipts) < len(tasks):
                for task in tasks:
                    if task.task_id in receipts or any(
                        active.task_id == task.task_id for active in in_flight.values()
                    ):
                        continue
                    parents = [receipts.get(key) for key in task.dependencies]
                    if any(parent is not None and parent.status != "committed"
                           for parent in parents):
                        receipts[task.task_id] = FactoryReceipt(
                            task.task_id, "blocked", None, None, "dependency_failed"
                        )
                        self.stream.emit("factory.blocked", task_id=task.task_id,
                                         reason="dependency_failed")
                        continue
                    if any(parent is None for parent in parents):
                        continue
                    if len(in_flight) >= self.workers:
                        continue
                    value = (sum(parent.value for parent in parents)
                             if task.use_dependency_sum else task.seed)
                    if abs(value) > MAX_SCALAR:
                        receipts[task.task_id] = FactoryReceipt(
                            task.task_id, "failed", None, None, "input_range_exceeded"
                        )
                        continue
                    future = executor.submit(self._compute, task, value)
                    in_flight[future] = task
                    peak = max(peak, len(in_flight))
                    self.stream.emit("factory.dispatch", task_id=task.task_id,
                                     running=len(in_flight), max_parallel=self.workers)
                if not in_flight:
                    if len(receipts) == len(tasks):
                        break
                    raise RuntimeError("No runnable tasks: invalid dependency state")
                future = next(as_completed(in_flight))
                task = in_flight.pop(future)
                try:
                    output, report = future.result()
                    input_value = (
                        sum(receipts[key].value for key in task.dependencies)
                        if task.use_dependency_sum else task.seed
                    )
                    node = self.store.append(
                        {
                            "kind": "factory-task/v1", "batch_id": batch_id,
                            "task_id": task.task_id, "operation": task.operation,
                            "input": input_value, "amount": task.amount,
                            "output": output, "verification": "local-arithmetic-oracle/v1",
                        },
                        report=report,
                        parents=tuple(receipts[key].digest for key in task.dependencies),
                    )
                    receipts[task.task_id] = FactoryReceipt(
                        task.task_id, "committed", output, node.digest, "verified"
                    )
                    self.stream.emit("factory.commit", task_id=task.task_id,
                                     digest=node.digest, verified=True)
                except Exception:
                    # Don't persist exception contents, credentials or unverified work.
                    receipts[task.task_id] = FactoryReceipt(
                        task.task_id, "failed", None, None, "execution_or_verification_failed"
                    )
                    self.stream.emit("factory.fail", task_id=task.task_id,
                                     reason="execution_or_verification_failed")
        elapsed = perf_counter() - start
        ordered = tuple(receipts[task.task_id] for task in tasks)
        summary = FactoryRun(batch_id, ordered, elapsed, self.workers, peak)
        self.stream.emit("factory.close", batch_size=len(tasks),
                         completed=summary.completed, failed=summary.failed,
                         blocked=summary.blocked, throughput=summary.tasks_per_second)
        return summary
