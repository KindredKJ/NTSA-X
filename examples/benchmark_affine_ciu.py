"""Benchmark affine compilation on distinct, previously unseen binary inputs."""
import json
import random
import statistics
import time

from ntsa_x.affine_ciu import AffineTransition


def sequential(width, state, program):
    limit = (1 << width) - 1
    for op, arg in program:
        if op == 'xor':
            state ^= arg
        elif op == 'rotl':
            state = ((state << arg) | (state >> (width - arg))) & limit
        elif op == 'xorshl':
            state ^= (state << arg) & limit
        elif op == 'xorshr':
            state ^= state >> arg
        else:
            raise ValueError(op)
    return state


def main():
    rng = random.Random(314159)
    width = 64
    program = [(op, rng.getrandbits(64) if op == 'xor' else rng.randrange(1, 64))
               for _ in range(1024)
               for op in ('xor', 'rotl', 'xorshl', 'xorshr')]
    inputs = list(dict.fromkeys(rng.getrandbits(64) for _ in range(1024)))
    start = time.perf_counter_ns()
    rule = AffineTransition.compile(width, program)
    build = (time.perf_counter_ns() - start) / 1e6
    cache = {}
    times = {}
    expected = None
    for name, fn in [
        ('sequential', lambda s: sequential(width, s, program)),
        ('compiled', rule.apply),
        ('cache_cold', lambda s: cache.setdefault(s, sequential(width, s, program))
         if s not in cache else cache[s]),
    ]:
        samples = []
        for _ in range(5):
            cache.clear()
            start = time.perf_counter_ns()
            output = [fn(s) for s in inputs]
            samples.append((time.perf_counter_ns() - start) / 1e6)
            if name == 'sequential':
                expected = output
            elif output != expected:
                raise AssertionError('result mismatch')
        times[name] = statistics.median(samples)
    saved = (times['sequential'] - times['compiled']) / len(inputs)
    print(json.dumps(dict(
        operations=len(program), unseen=len(inputs), build_ms=build, **times,
        speedup=times['sequential'] / times['compiled'],
        break_even=build / saved if saved > 0 else None,
        notes='Synthetic affine instructions; in-memory replay; no neural benchmark.'
    ), indent=2))


if __name__ == '__main__':
    main()
