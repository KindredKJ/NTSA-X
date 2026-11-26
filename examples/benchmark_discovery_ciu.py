"""Black-box active discovery on a structured quadratic synthetic task."""
import json
import random
import statistics
import time

from ntsa_x.discovery_ciu import discover, audit


def main():
    rng = random.Random(192)
    gates = [(rng.randrange(16, 32), rng.randrange(16), rng.randrange(16))
             for _ in range(4096)]

    def oracle(state):
        for target, a, b in gates:
            state ^= (((state >> a) & (state >> b)) & 1) << target
        return state

    start = time.perf_counter_ns()
    candidate = discover(32, oracle, degree=2)
    build_ms = (time.perf_counter_ns() - start) / 1e6
    states = [s for s in rng.sample(range(1 << 32), 1100) if s.bit_count() > 2][:1024]
    assert len(states) == 1024
    checked, mismatched = audit(candidate, oracle, states)
    assert mismatched == 0
    expected = [oracle(s) for s in states]
    times = {}
    for name, fn in [('oracle', oracle), ('candidate', candidate.rule.apply),
                     ('checked', lambda s: candidate.checked_apply(s, oracle)[0])]:
        samples = []
        for _ in range(5):
            start = time.perf_counter_ns()
            output = [fn(s) for s in states]
            samples.append((time.perf_counter_ns() - start) / 1e6)
            assert output == expected
        times[name] = statistics.median(samples)
    print(json.dumps(dict(probes=candidate.probes, exhaustive=candidate.exhaustive,
        degree=candidate.rule.degree, terms=len(candidate.rule.terms),
        holdout=checked, mismatches=mismatched, build_ms=build_ms, **times,
        candidate_speedup=times['oracle']/times['candidate'],
        notes='The workload is quadratic by construction, unknown to the learner; finite holdout alone does not certify an arbitrary oracle.'), indent=2))


if __name__ == '__main__':
    main()
