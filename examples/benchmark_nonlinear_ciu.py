"""Structured nonlinear gate reuse on distinct unseen 32-bit states."""
import json
import random
import statistics
import time

from ntsa_x.nonlinear_ciu import BooleanTransition


def sequential(state, gates):
    for op, target, *controls in gates:
        delta = 1 if op == 'not' else (state >> controls[0]) & 1
        if op == 'and_xor':
            delta &= (state >> controls[1]) & 1
        state ^= delta << target
    return state


def main():
    rng = random.Random(93)
    gates = [('and_xor', rng.randrange(16, 32), rng.randrange(16), rng.randrange(16))
             for _ in range(4096)]
    inputs = rng.sample(range(1 << 32), 1024)
    start = time.perf_counter_ns()
    rule = BooleanTransition.compile(32, gates)
    build_ms = (time.perf_counter_ns() - start) / 1e6
    # Independent simple optimization: cancel duplicate commuting gates.
    remaining = set()
    for op, target, a, b in gates:
        g = (op, target, min(a, b), max(a, b))
        if g in remaining:
            remaining.remove(g)
        else:
            remaining.add(g)
    reduced = tuple(sorted(remaining))
    expected = None
    times = {}
    for name, fn in [('sequential', lambda s: sequential(s, gates)),
                     ('canceled', lambda s: sequential(s, reduced)),
                     ('compiled', rule.apply)]:
        samples = []
        for _ in range(5):
            start = time.perf_counter_ns()
            actual = [fn(s) for s in inputs]
            samples.append((time.perf_counter_ns() - start) / 1e6)
            if expected is None:
                expected = actual
            elif actual != expected:
                raise AssertionError('result mismatch')
        times[name] = statistics.median(samples)
    saved = (times['sequential'] - times['compiled']) / len(inputs)
    print(json.dumps(dict(gates=len(gates), canceled_gates=len(reduced),
        unique_monomials=len(rule.terms), degree=rule.degree, unseen=len(inputs),
        build_ms=build_ms, **times, speedup=times['sequential']/times['compiled'],
        optimized_speedup=times['canceled']/times['compiled'],
        break_even_inputs=build_ms/saved if saved > 0 else None,
        notes='Structured commuting controlled AND-XOR gates; exact, in-memory, not neural.'), indent=2))


if __name__ == '__main__':
    main()
