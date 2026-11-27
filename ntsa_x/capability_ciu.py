"""Revision-scoped capability dispatch on the existing Nexus ledger.

Only deterministic, side-effect-free, operator-registered binary tasks qualify.
Revision must change whenever implementation or relevant dependencies change.
"""
from dataclasses import dataclass
from time import perf_counter_ns
from typing import Callable
import re

from .adaptive_ciu import NexusRuleLibrary, learn
from .convergence import ConvergenceReport
from .nexus import NexusStore
from .xor_ciu import _bits, _label


@dataclass(frozen=True)
class BinaryTask:
    name: str
    revision: str
    width: int
    oracle: Callable[[int], int]
    report: ConvergenceReport | None = None

    def __post_init__(self):
        if not isinstance(self.name, str) or not re.fullmatch(r'[a-z][a-z0-9_.-]{1,40}', self.name):
            raise ValueError('invalid task name')
        _label(self.revision)
        if type(self.width) is not int or not 1 <= self.width <= 12 or not callable(self.oracle):
            raise ValueError('expected pure binary task with width 1..12')


@dataclass(frozen=True)
class Execution:
    value: int
    source: str
    oracle_calls: int
    elapsed_ns: int
    receipt: str | None


class CapabilityRuntime:
    def __init__(self, store: NexusStore):
        self.library = NexusRuleLibrary(store)
        self.store = store
        with store._lock:
            store._db.execute('''CREATE TABLE IF NOT EXISTS ciu_capability_index (
                name TEXT NOT NULL, revision TEXT NOT NULL, width INTEGER NOT NULL,
                digest TEXT NOT NULL REFERENCES nodes(digest),
                PRIMARY KEY(name, revision, width))''')

    def execute(self, task: BinaryTask, state: int, *, acquire: bool = False) -> Execution:
        if not isinstance(task, BinaryTask):
            raise TypeError('BinaryTask required')
        _bits(state, task.width)
        start = perf_counter_ns()
        calls = 0
        key = (task.name, task.revision, task.width)

        def oracle(s):
            nonlocal calls
            calls += 1
            return _bits(task.oracle(s), task.width)

        def outcome(value, source, receipt=None):
            return Execution(value, source, calls, perf_counter_ns() - start, receipt)

        with self.store._lock:
            row = self.store._db.execute('SELECT digest FROM ciu_capability_index WHERE name=? AND revision=? AND width=?', key).fetchone()
        if row:
            try:
                rule = self.library.recall(row[0], context_id=task.name, revision=task.revision)
            except (ValueError, TypeError, KeyError):
                rule = None
            if rule is not None and rule.width == task.width:
                return outcome(rule.apply(state), 'recalled', row[0])
            # Do not silently overwrite a damaged index or hide its provenance.
            return outcome(oracle(state), 'invalid_receipt_fallback')

        report = task.report
        if acquire and isinstance(report, ConvergenceReport) and report.converged and report.external_evidence:
            size = 1 << task.width
            candidate = learn(task.width, oracle, range(size),
                              max_degree=min(6, task.width), max_observations=size)
            if candidate.rule is not None and candidate.exhaustive:
                try:
                    receipt = self.library.register(candidate.rule, context_id=task.name,
                        revision=task.revision, oracle=oracle, report=report)
                except ValueError:
                    return outcome(oracle(state), 'verification_fallback')
                with self.store._lock:
                    self.store._db.execute('INSERT OR IGNORE INTO ciu_capability_index VALUES (?,?,?,?)', (*key, receipt))
                    selected = self.store._db.execute('SELECT digest FROM ciu_capability_index WHERE name=? AND revision=? AND width=?', key).fetchone()[0]
                if selected != receipt:
                    return outcome(oracle(state), 'conflict_fallback')
                return outcome(candidate.rule.apply(state), 'acquired', receipt)
        return outcome(oracle(state), 'original')


def ciu_extension(runtime: CapabilityRuntime, tasks: tuple[BinaryTask, ...], *, acquire: bool = False):
    """Attach to UnifiedCognitiveMechanism.extensions; explicit commands only."""
    from .unified_cognition import NativeExtension
    catalog = {t.name: t for t in tasks}
    if len(catalog) != len(tasks):
        raise ValueError('duplicate task name')
    pattern = re.compile(r'ciu ([a-z][a-z0-9_.-]{1,40}) (0x[0-9a-fA-F]{1,3}|[0-9]{1,4})')

    def parse(request):
        match = pattern.fullmatch(request.strip())
        if not match or match[1] not in catalog:
            return None
        task = catalog[match[1]]
        state = int(match[2], 16 if match[2].startswith('0x') else 10)
        return (task, state) if state < (1 << task.width) else None

    def contribute(frame):
        parsed = parse(frame.request)
        if parsed is None:
            return ''
        result = runtime.execute(*parsed, acquire=acquire)
        return f'CIU {parsed[0].name}: {result.value}; source={result.source}; oracle_calls={result.oracle_calls}'

    return NativeExtension('ciu.native', lambda request: parse(request) is not None, contribute)
