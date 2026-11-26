"""Measured, horizon-aware selection for a trusted pure binary task.

Plans are in-process snapshots: recreate after changing task revision or
invalidating the ledger. Calibration performs original computation; replay
only avoids it if an independently registered exact rule is selected.
"""
from dataclasses import dataclass, field
from statistics import median
from time import perf_counter_ns
from typing import Callable
from array import array
from sys import getsizeof

from .capability_ciu import BinaryTask, CapabilityRuntime
from .xor_ciu import _bits


@dataclass(frozen=True)
class Cost:
    method: str
    setup_ns: int
    per_call_ns: float
    extra_table_bytes: int = 0

    def total(self, calls: int) -> float:
        return self.setup_ns + calls * self.per_call_ns


@dataclass(frozen=True)
class ExecutionPlan:
    task: BinaryTask
    method: str
    costs: tuple[Cost, ...]
    expected_calls: int
    calibration_oracle_calls: int
    planning_ns: int
    receipt: str | None
    _apply: Callable[[int], int] = field(repr=False, compare=False)

    def apply(self, state: int) -> int:
        return self._apply(_bits(state, self.task.width))


def select(runtime: CapabilityRuntime, task: BinaryTask, *, expected_calls: int,
           table_budget_bytes: int = 0, samples: int = 32,
           repeats: int = 5) -> ExecutionPlan:
    """Profile existing choices; never learn or promote in the planning path.

    Table budget counts the extra array allocation, not the rule/plan objects.
    Timings are estimates, not guaranteed savings; planning cost is reported
    separately. The original must obey the same purity/revision contract.
    """
    if not isinstance(runtime, CapabilityRuntime) or not isinstance(task, BinaryTask):
        raise TypeError('runtime and task required')
    for value, lo, hi in [(expected_calls, 1, 10**12), (table_budget_bytes, 0, 1_048_576),
                          (samples, 1, 256), (repeats, 1, 21)]:
        if type(value) is not int or not lo <= value <= hi:
            raise ValueError('invalid planning budget')
    start = perf_counter_ns()
    size = 1 << task.width
    states = tuple(i * size // min(size, samples) for i in range(min(size, samples)))
    calls = 0

    def original(s):
        return _bits(task.oracle(s), task.width)

    def profile(fn, expected=None):
        times = []
        for _ in range(repeats):
            tick = perf_counter_ns()
            results = tuple(fn(s) for s in states)
            times.append((perf_counter_ns() - tick) / len(states))
            if expected is not None and results != expected:
                return None
        return median(times)

    expected = tuple(original(s) for s in states)
    calls += len(states)
    direct = profile(original, expected)
    calls += len(states) * repeats
    if direct is None:
        raise ValueError('original task violates deterministic contract')
    choices = [(Cost('original', 0, direct), original)]
    lookup = perf_counter_ns()
    with runtime.store._lock:
        row = runtime.store._db.execute(
            'SELECT digest FROM ciu_capability_index WHERE name=? AND revision=? AND width=?',
            (task.name, task.revision, task.width)).fetchone()
    rule = None
    if row:
        try:
            rule = runtime.library.recall(row[0], context_id=task.name, revision=task.revision)
        except (ValueError, TypeError, KeyError):
            pass
    setup = perf_counter_ns() - lookup
    if rule is not None and rule.width == task.width:
        cost = profile(rule.apply, expected)
        if cost is not None:
            choices.append((Cost('rule', setup, cost), rule.apply))
            empty = array('H')
            if getsizeof(empty) + size * empty.itemsize <= table_budget_bytes:
                tick = perf_counter_ns()
                table = array('H', [0]) * size
                for s in range(size):
                    table[s] = rule.apply(s)
                build = perf_counter_ns() - tick
                actual_bytes = getsizeof(table)
                if actual_bytes <= table_budget_bytes:
                    cost = profile(table.__getitem__, expected)
                    if cost is not None:
                        choices.append((Cost('table', setup + build, cost, actual_bytes), table.__getitem__))
    best, fn = min(choices, key=lambda c: c[0].total(expected_calls))
    return ExecutionPlan(task, best.method, tuple(c for c, _ in choices), expected_calls,
                         calls, perf_counter_ns() - start,
                         row[0] if row and best.method != 'original' else None, fn)
