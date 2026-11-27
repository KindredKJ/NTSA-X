"""Batch-scoped rule reuse and bounded hot-chain promotion."""
from collections import OrderedDict
from dataclasses import dataclass
from hashlib import sha256
import json
from threading import RLock
from time import perf_counter_ns

from .adaptive_ciu import KIND
from .nonlinear_ciu import BooleanTransition
from .capability_ciu import BinaryTask, CapabilityRuntime
from .compose_ciu import compose
from .convergence import ConvergenceReport
from .xor_ciu import _bits


@dataclass(frozen=True)
class BatchResult:
    values: tuple[int, ...]
    source: str
    receipt: str | None
    elapsed_ns: int
    promotion_attempted: bool


class HotChainRuntime:
    """Observe explicit pure task chains, not arbitrary program traces.

    One read transaction per batch; no cached rule survives a batch boundary.
    Task identity/revision must faithfully identify the original computation.
    Heat counts are bounded, session-local hints, never correctness evidence.
    """
    def __init__(self, runtime: CapabilityRuntime, *, min_batches=3,
                 min_inputs=512, max_chains=128, max_oracle_calls=65536):
        if not isinstance(runtime, CapabilityRuntime):
            raise TypeError('CapabilityRuntime required')
        for v, lo, hi in ((min_batches, 2, 1024), (min_inputs, 1, 10**7),
                          (max_chains, 1, 4096), (max_oracle_calls, 1, 65536)):
            if type(v) is not int or not lo <= v <= hi:
                raise ValueError('invalid hot-chain budget')
        self.runtime = runtime
        self.min_batches, self.min_inputs = min_batches, min_inputs
        self.max_chains, self.max_oracle_calls = max_chains, max_oracle_calls
        self._heat = OrderedDict()
        self._lock = RLock()

    def _snapshot(self, tasks):
        store = self.runtime.store
        with store._lock:
            store._db.execute('BEGIN')
            try:
                keys = [(t.name, t.revision, t.width) for t in tasks]
                rows = [store._db.execute(
                    'SELECT digest FROM ciu_capability_index WHERE name=? AND revision=? AND width=?', k
                ).fetchone() for k in keys]
                if not all(rows):
                    return (), (), None, None
                receipts = tuple(r[0] for r in rows)
                revision = sha256(json.dumps(receipts, separators=(',', ':')).encode()).hexdigest()
                name = 'chain.' + revision[:32]
                row = store._db.execute(
                    'SELECT digest FROM ciu_capability_index WHERE name=? AND revision=? AND width=?',
                    (name, revision, tasks[0].width)).fetchone()
                if row:
                    node = store.read(row[0])
                    if node is None or node.parents != tuple(dict.fromkeys(receipts)):
                        return (), (), None, None
                    p = node.payload
                    width = tasks[0].width
                    if (p.get('kind') != KIND or p.get('context_id') != name
                            or p.get('revision') != revision or p.get('width') != width
                            or p.get('verified_inputs') != 1 << width
                            or p.get('composition_order') != list(receipts)):
                        return (), (), None, None
                    fused = BooleanTransition(width, tuple((int(m, 16), int(v, 16)) for m, v in p['terms']), p['count'])
                    return receipts, (), fused, row[0]
                rules = {}
                for t, receipt in zip(tasks, receipts):
                    key = (receipt, t.name, t.revision)
                    if key not in rules:
                        rule = self.runtime.library.recall(receipt, context_id=t.name, revision=t.revision)
                        if rule is None or rule.width != t.width:
                            return (), (), None, None
                        rules[key] = rule
                return receipts, tuple(rules[(r, t.name, t.revision)] for t, r in zip(tasks, receipts)), None, None
            except (ValueError, TypeError, KeyError):
                return (), (), None, None
            finally:
                store._db.execute('ROLLBACK')  # close the read snapshot

    def execute(self, tasks: tuple[BinaryTask, ...], states, *, report=None) -> BatchResult:
        if type(tasks) is not tuple or not 2 <= len(tasks) <= 16 or any(not isinstance(t, BinaryTask) for t in tasks):
            raise ValueError('expected 2..16 binary tasks')
        width = tasks[0].width
        if any(t.width != width for t in tasks):
            raise ValueError('incompatible widths')
        inputs = []
        for state in states:
            if len(inputs) >= 65536:
                raise ValueError('batch limit exceeded')
            inputs.append(_bits(state, width))
        if not inputs:
            raise ValueError('empty batch')
        start = perf_counter_ns()
        with self._lock:
            receipts, rules, fused, digest = self._snapshot(tasks)
            attempted = False
            if fused is None and rules:
                batches, count, failed = self._heat.pop(receipts, (0, 0, False))
                batches = min(self.min_batches, batches + 1)
                count = min(self.min_inputs, count + len(inputs))
                if not failed and batches >= self.min_batches and count >= self.min_inputs and isinstance(report, ConvergenceReport) and report.converged and report.external_evidence:
                    attempted = True
                    try:
                        compose(self.runtime, tasks, report=report, max_oracle_calls=self.max_oracle_calls)
                        _, _, fused, digest = self._snapshot(tasks)
                        failed = fused is None
                    except ValueError:
                        failed = True
                        # A failed exhaustive check might indicate revision drift.
                        rules = ()
                self._heat[receipts] = (batches, count, failed)
                while len(self._heat) > self.max_chains:
                    self._heat.popitem(last=False)
            if fused is not None:
                return BatchResult(tuple(fused.apply(s) for s in inputs), 'fused', digest, perf_counter_ns()-start, attempted)
            # A failed promotion is not proof of equivalence. Use originals
            # on subsequent calls to that chain as well, until rebuilt.
            if receipts and self._heat.get(receipts, (0, 0, False))[2]:
                rules = ()
            functions = tuple(r.apply for r in rules) if rules else tuple(t.oracle for t in tasks)
            output = []
            for state in inputs:
                for fn in functions:
                    state = _bits(fn(state), width)
                output.append(state)
            return BatchResult(tuple(output), 'stages' if rules else 'original', None, perf_counter_ns()-start, attempted)
