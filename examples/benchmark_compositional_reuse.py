"""SW12: measured component reuse using the existing Nexus and ConvergenceLibrary.

Run from repository root: python -m examples.benchmark_compositional_reuse --output sw12.json
Synthetic contractive four-coordinate rings only; no neural or worker speed claims.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import json
import math
from pathlib import Path
import platform
from statistics import median
from tempfile import TemporaryDirectory
from time import perf_counter_ns

from ntsa_x.convergence import Check, ConvergenceEngine
from ntsa_x.nexus import NexusStore
from ntsa_x.reuse import ConvergenceLibrary

Q = 0.875
TOL = 1e-10
CTX = "sw12/ring4"
SCOPE = "ring4/q7-8/jacobi-reverse-gs/v1"
COMPOSITE = "ring4/ordered-product/v1"
REV = "analytic-rational-residual/v1"


def fixture(i):
    x = [i + 1, -i - 2, i + 3, -i - 4]
    return [x[j] - Q * x[(j + 1) % 4] for j in range(4)]


def reference(b):
    q = Fraction(7, 8)
    f = [Fraction(v) for v in b]
    return [float(sum(q**k * f[(i + k) % 4] for k in range(4)) /
                  (1 - q**4)) for i in range(4)]


def residual(x, b):
    return max(abs(x[i] - Q * x[(i + 1) % 4] - b[i])
               for i in range(4)) / max(1.0, max(map(abs, b)))


def valid(x, b):
    return (isinstance(x, list) and len(x) == 4
            and all(type(v) in (int, float) and math.isfinite(v) for v in x)
            and residual(x, b) <= TOL
            and max(abs(a - v) for a, v in zip(x, reference(b))) <= 1e-8)


def solve(b, reverse=False):
    x = [0.0] * 4
    for step in range(1, 2049):
        if reverse:
            for i in range(3, -1, -1):
                x[i] = Q * x[(i + 1) % 4] + b[i]
        else:
            x = [Q * x[(i + 1) % 4] + b[i] for i in range(4)]
        # Absolute residual supports the fixture's absolute error requirement.
        if max(abs(x[i] - Q * x[(i + 1) % 4] - b[i])
               for i in range(4)) <= 1e-11:
            if not valid(x, b):
                raise ArithmeticError("solver failed independent reference")
            # Integer-valued synthetic fixture only; not a general quantizer.
            return [round(v) for v in x], 4 * step
    raise ArithmeticError("bounded solver did not converge")


def report(forward, reverse, bs):
    f, r = forward.get("states"), reverse.get("states")
    shape = (isinstance(f, list) and isinstance(r, list)
             and len(f) == len(r) == len(bs))
    verified = shape and all(valid(x, b) and valid(y, b)
                             for x, y, b in zip(f, r, bs))
    checks = [
        Check("state", bool(shape), "fixture/shape"),
        Check("semantic", bool(verified), "independent analytic fixture solution"),
        Check("causal", bool(verified), "ordered RHS-to-state association"),
        Check("constraint", bool(verified), "ring equation and finite state"),
        Check("evidence", bool(verified), "analytic rational oracle, outside iterative solver", external=True),
        Check("delta", bool(verified), "current residual and reference error"),
    ]
    return ConvergenceEngine().evaluate(checks, forward=forward, reverse=reverse)


def key(b):
    return ({"rhs": b}, {"rhs": b})


def component(store, lib, b):
    args = dict(context_id=CTX, scope=SCOPE, revision=REV, inputs=key(b))
    old = lib.get(**args)
    if old is not None:
        p = old.payload
        if report(p["forward"], p["reverse"], [b]).converged and p["resolved"] == p["forward"]:
            return old, 0, 1
        # Immutable index conflict is surfaced rather than silently overwritten.
        raise ValueError("current verifier rejected indexed state")
    x, n = solve(b)
    y, m = solve(b, True)
    f, r = {"states": [x]}, {"states": [y]}
    checked = report(f, r, [b])
    node = store.append({"context_id": CTX, "forward": f, "reverse": r,
                         "resolved": f}, report=checked)
    lib.register(**args, node_digest=node.digest, report=checked)
    return node, n + m, 0


def run_case(path, bs, prime):
    with NexusStore(path) as store:
        lib = ConvergenceLibrary(store)
        start = perf_counter_ns()
        for b in bs[:prime]:
            component(store, lib, b)
        prime_ns = perf_counter_ns() - start
        start = perf_counter_ns()
        parts = [component(store, lib, b) for b in bs]
        f = {"states": [p.payload["forward"]["states"][0] for p, _, _ in parts]}
        r = {"states": [p.payload["reverse"]["states"][0] for p, _, _ in parts]}
        checked = report(f, r, bs)
        args = dict(context_id=CTX, scope=COMPOSITE, revision=REV, inputs=key(bs))
        node = lib.compose(**args, members=[p.digest for p, _, _ in parts],
                           forward=f, reverse=r, resolved=f, report=checked)
        elapsed = perf_counter_ns() - start
        start = perf_counter_ns()
        recalled = lib.get(**args)
        if (recalled is None or recalled.digest != node.digest
                or recalled.payload["resolved"] != recalled.payload["forward"]
                or not report(recalled.payload["forward"], recalled.payload["reverse"], bs).converged):
            raise AssertionError("composite recall invalid")
        exact_ns = perf_counter_ns() - start
        # Deterministic negative cases, outside timing.
        if lib.get(**dict(args, revision="invalidated/v2")) is not None:
            raise AssertionError("stale revision accepted")
        wrong = {"states": [v[:] for v in f["states"]]}
        wrong["states"][0][0] += 1
        if report(wrong, wrong, bs).converged:
            raise AssertionError("wrong but agreeing trajectories accepted")
        result = dict(elapsed_ns=elapsed, prime_ns=prime_ns, exact_recall_ns=exact_ns,
                      row_updates=sum(n for _, n, _ in parts),
                      hits=sum(h for _, _, h in parts), max_residual=max(
                          residual(x, b) for x, b in zip(f["states"], bs)),
                      max_error=max(abs(a-v) for x,b in zip(f["states"],bs)
                                    for a,v in zip(x,reference(b))),
                      digest=node.digest, **store.stats())
    result["database_bytes"] = sum(p.stat().st_size for p in path.parent.glob(path.name + "*"))
    return result


def direct_baseline(bs):
    start = perf_counter_ns()
    f = {"states": [reference(b) for b in bs]}
    if not report(f, f, bs).converged:
        raise AssertionError("direct baseline invalid")
    return perf_counter_ns() - start


def iterative_baseline(bs):
    start = perf_counter_ns()
    pairs = [(solve(b), solve(b, True)) for b in bs]
    f = {"states": [a[0] for a, _ in pairs]}
    r = {"states": [b[0] for _, b in pairs]}
    if not report(f, r, bs).converged:
        raise AssertionError("iterative baseline invalid")
    return perf_counter_ns() - start


def benchmark(sizes, repeats):
    rows = []
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        for n in sizes:
            bs = [fixture(i) for i in range(n)]
            for trial in range(repeats):
                direct = direct_baseline(bs)
                iterative = iterative_baseline(bs)
                # Rotate condition order to reduce systematic timing bias.
                percentages = [0, 25, 50, 75, 100]
                for pct in percentages[trial % 5:] + percentages[:trial % 5]:
                    p = n * pct // 100
                    row = run_case(root / f"{n}-{trial}-{pct}.sqlite", bs, p)
                    if row["hits"] != p:
                        raise AssertionError("unexpected component reuse")
                    rows.append(dict(blocks=n, trial=trial, reuse_pct=pct,
                                     direct_ns=direct, iterative_no_store_ns=iterative, **row))
    summaries = []
    for n in sizes:
        cold = median(r["elapsed_ns"] for r in rows if r["blocks"] == n and r["reuse_pct"] == 0)
        for pct in [0,25,50,75,100]:
            group = [r for r in rows if r["blocks"] == n and r["reuse_pct"] == pct]
            elapsed = median(r["elapsed_ns"] for r in group)
            summaries.append(dict(blocks=n, reuse_pct=pct, median_elapsed_ns=elapsed,
                                  cold_speedup=cold/elapsed, median_row_updates=median(
                                      r["row_updates"] for r in group),
                                  median_exact_recall_ns=median(r["exact_recall_ns"] for r in group)))
    return dict(schema="ntsa-x/sw12/v1", python=platform.python_version(),
                platform=platform.platform(), rows=rows, summaries=summaries,
                limitations=["synthetic independent rings; no learned states",
                             "warm priming measured separately; exact composite recall timed separately",
                             "analytic reference is a strong direct baseline, not a neural baseline"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[4, 16, 64, 256, 1024])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("sw12.json"))
    args = parser.parse_args()
    if not 1 <= args.repeats <= 30 or any(n < 4 or n > 1024 or n % 4 for n in args.sizes):
        parser.error("sizes must be multiples of 4 in [4,1024]; repeats in [1,30]")
    output = benchmark(sorted(set(args.sizes)), args.repeats)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps(output["summaries"], indent=2))


if __name__ == "__main__":
    main()
