"""Synthetic hot-chain batch benchmark, including acquisition and promotion."""
import json
import random
import statistics
import time
from ntsa_x.batch_ciu import HotChainRuntime
from ntsa_x.capability_ciu import BinaryTask, CapabilityRuntime
from ntsa_x.convergence import Check, ConvergenceEngine, REQUIRED_CHECKS
from ntsa_x.nexus import NexusStore


def main():
    rng = random.Random(192)
    gates = [tuple((rng.randrange(4, 8), rng.randrange(4), rng.randrange(4))
                   for _ in range(256)) for _ in range(8)]
    def make_oracle(stage):
        def oracle(s):
            for target, a, b in stage:
                s ^= (((s >> a) & (s >> b)) & 1) << target
            return s
        return oracle
    report = ConvergenceEngine().evaluate([Check(c, True, 'synthetic fixture', external=c == 'evidence') for c in REQUIRED_CHECKS])
    tasks = tuple(BinaryTask(f'stage{i}', 'v1', 8, make_oracle(g), report) for i, g in enumerate(gates))
    def original(s):
        for task in tasks:
            s = task.oracle(s)
        return s
    batches = [tuple(rng.randrange(256) for _ in range(256)) for _ in range(16)]
    def measure(fn):
        t = time.perf_counter_ns()
        result = fn()
        return (time.perf_counter_ns()-t)/1e6, result
    baseline = []
    for _ in range(5):
        t, expected = measure(lambda: [tuple(map(original, batch)) for batch in batches])
        baseline.append(t)
    with NexusStore(':memory:') as store:
        runtime = CapabilityRuntime(store)
        acquired, _ = measure(lambda: [runtime.execute(task, 0, acquire=True) for task in tasks])
        hot = HotChainRuntime(runtime)
        elapsed, results = measure(lambda: [hot.execute(tasks, b, report=report) for b in batches])
        assert [r.values for r in results] == expected
        assert [r.source for r in results[:3]] == ['stages','stages','fused']
        warm = []
        for _ in range(5):
            t, r = measure(lambda: [hot.execute(tasks, b) for b in batches])
            assert [x.values for x in r] == expected
            warm.append(t)
        # Exact finite-domain lookup is a strong baseline for this width.
        table_build, table = measure(lambda: tuple(map(original, range(256))))
        table_times = []
        for _ in range(5):
            t, r = measure(lambda: [tuple(table[s] for s in b) for b in batches])
            assert r == expected
            table_times.append(t)
        print(json.dumps(dict(inputs=sum(map(len,batches)), original_ms=statistics.median(baseline),
            acquisition_ms=acquired, first_run_ms=elapsed, warm_ms=statistics.median(warm),
            table_build_ms=table_build, table_ms=statistics.median(table_times),
            methods=[r.source for r in results], promotion_batch_ms=results[2].elapsed_ns/1e6),indent=2))


if __name__ == '__main__':
    main()
