"""Compose retained exact binary capabilities with ordered Nexus provenance."""
from dataclasses import dataclass
from hashlib import sha256
import json

from .adaptive_ciu import KIND
from .capability_ciu import BinaryTask, CapabilityRuntime
from .convergence import ConvergenceReport
from .xor_ciu import _bits


@dataclass(frozen=True)
class Composition:
    task: BinaryTask
    receipt: str
    stages: int
    verified_inputs: int
    oracle_calls: int


def compose(runtime: CapabilityRuntime, tasks: tuple[BinaryTask, ...], *,
            report: ConvergenceReport, max_terms: int = 4096,
            max_work: int = 2_000_000, max_oracle_calls: int = 65536) -> Composition:
    """Compile an explicit ordered chain of pure tasks, not arbitrary workflows.

    Each stage must already have an exact revision-bound receipt. Verify the
    composition against the original sequential oracles across the whole
    finite domain. The derived task identity includes ordered parent receipts.
    """
    if not isinstance(runtime, CapabilityRuntime) or type(tasks) is not tuple or not 2 <= len(tasks) <= 16:
        raise ValueError('expected runtime and 2..16 tasks')
    if any(not isinstance(t, BinaryTask) for t in tasks):
        raise TypeError('BinaryTask required')
    width = tasks[0].width
    if any(t.width != width for t in tasks):
        raise ValueError('incompatible widths')
    if not isinstance(report, ConvergenceReport) or not report.converged or not report.external_evidence:
        raise ValueError('independent convergence report required')
    size = 1 << width
    if type(max_oracle_calls) is not int or not size * len(tasks) <= max_oracle_calls <= 65536:
        raise ValueError('oracle budget exceeded')
    store = runtime.store
    receipts, rules = [], []
    leaf_calls = 0
    for task in tasks:
        with store._lock:
            row = store._db.execute('SELECT digest FROM ciu_capability_index WHERE name=? AND revision=? AND width=?',
                                    (task.name, task.revision, width)).fetchone()
        rule = runtime.library.recall(row[0], context_id=task.name, revision=task.revision) if row else None
        if rule is None or rule.width != width:
            raise ValueError('missing verified stage')
        node = store.read(row[0])
        count = node.payload.get('leaf_calls_per_input', 1)
        if type(count) is not int or not 1 <= count <= 65536:
            raise ValueError('invalid composition cost provenance')
        leaf_calls += count
        if leaf_calls * size > max_oracle_calls:
            raise ValueError('nested oracle budget exceeded')
        receipts.append(row[0])
        rules.append(rule)
    combined = rules[0]
    for rule in rules[1:]:
        combined = combined.then(rule, max_terms=max_terms, max_work=max_work)

    def original(state):
        for task in tasks:
            state = _bits(task.oracle(state), width)
        return state

    for state in range(size):
        if combined.apply(state) != original(state):
            raise ValueError('composition failed exhaustive verification')
    revision = sha256(json.dumps(receipts, separators=(',', ':')).encode()).hexdigest()
    name = 'chain.' + revision[:32]
    task = BinaryTask(name, revision, width, original, report)
    payload = {'kind': KIND, 'context_id': name, 'revision': revision,
               'width': width, 'terms': [[hex(m), hex(v)] for m, v in combined.terms],
               'count': combined.count, 'verified_inputs': size,
               'composition_order': receipts, 'leaf_calls_per_input': leaf_calls}
    node = store.append(payload, report=report, parents=tuple(dict.fromkeys(receipts)))
    key = (name, revision, width)
    with store._lock:
        store._db.execute('INSERT OR IGNORE INTO ciu_capability_index VALUES (?,?,?,?)', (*key, node.digest))
        selected = store._db.execute('SELECT digest FROM ciu_capability_index WHERE name=? AND revision=? AND width=?', key).fetchone()[0]
    if selected != node.digest:
        raise ValueError('composition index conflict')
    return Composition(task, node.digest, len(tasks), size, size * leaf_calls)
