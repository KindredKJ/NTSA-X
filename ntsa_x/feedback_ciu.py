"""Online measured method selection for exact, revision-scoped binary tasks."""
from array import array
from dataclasses import dataclass
from math import isfinite
from sys import getsizeof
from threading import RLock
from time import perf_counter_ns, time_ns
import sqlite3

from .capability_ciu import BinaryTask, CapabilityRuntime
from .xor_ciu import _bits


@dataclass(frozen=True)
class Observation:
    method: str
    values: tuple[int, ...]
    execution_ns: int
    preparation_ns: int
    oracle_calls: int
    exploratory: bool


class FeedbackRuntime:
    """Adapt per batch without shadow recomputation or new rule promotion.

    Timing estimates are session-local unless environment is provided. A verified rule is read at every batch
    boundary; cached tables are discarded when the receipt changes. The task
    must remain pure and its revision must reflect implementation changes.
    """
    def __init__(self, runtime: CapabilityRuntime, task: BinaryTask, *,
                 table_budget_bytes: int = 0, explore_every: int = 16,
                 alpha: float = 0.5, environment: str | None = None,
                 max_age_s: int = 86400):
        if not isinstance(runtime, CapabilityRuntime) or not isinstance(task, BinaryTask):
            raise TypeError('runtime and task required')
        if type(table_budget_bytes) is not int or not 0 <= table_budget_bytes <= 1048576:
            raise ValueError('invalid table budget')
        if type(explore_every) is not int or not 2 <= explore_every <= 1024:
            raise ValueError('invalid exploration interval')
        if type(alpha) not in (int, float) or not isfinite(alpha) or not 0 < alpha <= 1:
            raise ValueError('invalid alpha')
        if environment is not None and (not isinstance(environment, str) or not 1 <= len(environment) <= 128):
            raise ValueError('invalid environment identity')
        if type(max_age_s) is not int or not 1 <= max_age_s <= 604800:
            raise ValueError('invalid timing retention')
        self.environment, self.max_age_ns = environment, max_age_s * 10**9
        self.runtime, self.task = runtime, task
        self.budget, self.interval, self.alpha = table_budget_bytes, explore_every, alpha
        self._lock = RLock()
        self._receipt = self._table = None
        self._estimates = {}
        self._batches = self._explore_index = 0
        if environment is not None:
            with runtime.store._lock:
                runtime.store._db.execute('''CREATE TABLE IF NOT EXISTS ciu_timing_profile (
                    name TEXT, revision TEXT, width INTEGER, receipt TEXT,
                    environment TEXT, batch_size INTEGER, budget INTEGER,
                    method TEXT, ns REAL, updated_ns INTEGER,
                    PRIMARY KEY(name,revision,width,receipt,environment,batch_size,budget,method))''')

    def _profile_key(self, size):
        return (self.task.name, self.task.revision, self.task.width,
                self._receipt or '', self.environment, size, self.budget)

    def _load_profile(self, size, names):
        now = time_ns()
        with self.runtime.store._lock:
            rows = self.runtime.store._db.execute(
                'SELECT method,ns,updated_ns FROM ciu_timing_profile WHERE '
                'name=? AND revision=? AND width=? AND receipt=? AND environment=? '
                'AND batch_size=? AND budget=?', self._profile_key(size)).fetchall()
        return {method: cost for method, cost, stamp in rows
                if method in names and isinstance(cost, (int, float))
                and isfinite(cost) and cost > 0 and type(stamp) is int
                and 0 <= now - stamp <= self.max_age_ns}

    def _save_profile(self, size, method, cost):
        now = time_ns()
        with self.runtime.store._lock:
            self.runtime.store._db.execute('''INSERT INTO ciu_timing_profile
                VALUES (?,?,?,?,?,?,?,?,?,?) ON CONFLICT DO UPDATE SET
                ns=CASE WHEN updated_ns BETWEEN ? AND ? AND ns>0 AND ns<1e18
                THEN ? * excluded.ns + (1-?) * ns ELSE excluded.ns END,
                updated_ns=excluded.updated_ns''', (*self._profile_key(size), method,
                max(cost, 1e-9), now, now - self.max_age_ns, now, self.alpha, self.alpha))

    @property
    def estimates(self):
        with self._lock:
            return dict(self._estimates)

    def execute(self, states) -> Observation:
        values = []
        for state in states:
            if len(values) >= 65536:
                raise ValueError('batch limit exceeded')
            values.append(_bits(state, self.task.width))
        if not values:
            raise ValueError('empty batch')
        with self._lock:
            start = perf_counter_ns()
            task, runtime = self.task, self.runtime
            methods = {'original': lambda s: _bits(task.oracle(s), task.width)}
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
            receipt = row[0] if rule is not None and rule.width == task.width else None
            if receipt != self._receipt:
                self._table = None
                self._estimates = {}
                self._batches = self._explore_index = 0
                self._receipt = receipt
            if receipt is not None:
                methods['rule'] = rule.apply
                size = 1 << task.width
                empty = array('H')
                if getsizeof(empty) + size * empty.itemsize <= self.budget:
                    if self._table is None:
                        table = array('H', [0]) * size
                        if getsizeof(table) <= self.budget:
                            for state in range(size):
                                table[state] = rule.apply(state)
                            self._table = table
                    if self._table is not None:
                        methods['table'] = self._table.__getitem__
            names = tuple(methods)
            if self.environment is not None:
                try:
                    self._estimates = self._load_profile(len(values), names)
                except sqlite3.Error:
                    self._estimates = {}
            unseen = [name for name in names if name not in self._estimates]
            due = self._batches % self.interval == 0
            if self.environment is not None and self._batches == 0:
                due = False
            exploratory = bool(unseen) or (due and len(names) > 1)
            if unseen:
                chosen = unseen[0]
            elif exploratory:
                chosen = names[self._explore_index % len(names)]
                self._explore_index += 1
            else:
                chosen = min(names, key=self._estimates.__getitem__)
            prepared = perf_counter_ns() - start
            tick = perf_counter_ns()
            result = tuple(methods[chosen](state) for state in values)
            elapsed = perf_counter_ns() - tick
            cost = elapsed / len(values)
            old = self._estimates.get(chosen, cost)
            self._estimates[chosen] = self.alpha * cost + (1 - self.alpha) * old
            if self.environment is not None:
                try:
                    self._save_profile(len(values), chosen, cost)
                except sqlite3.Error:
                    pass  # Non-authoritative telemetry must not block results.
            self._batches += 1
            return Observation(chosen, result, elapsed, prepared,
                               len(values) if chosen == 'original' else 0, exploratory)
