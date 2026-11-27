"""Synthetic end-to-end XOR replay benchmark; no neural performance claim."""
import random
import tempfile
import statistics
import json
import time
from pathlib import Path

from ntsa_x.xor_ciu import NexusXorCIU, XorTransition
from ntsa_x.nexus import NexusStore
from ntsa_x.convergence import ConvergenceEngine, Check, REQUIRED_CHECKS


def main():
    rng = random.Random(92)
    groups = [[rng.getrandbits(64) for _ in range(250_000)] for _ in range(4)]

    def seq(state, masks):
        for mask in masks:
            state ^= mask
        return state

    def full(state):
        for masks in groups:
            state = seq(state, masks)
        return state

    initial = 0xA6
    expected = full(initial)
    report = ConvergenceEngine().evaluate([
        Check(c, True, "synthetic benchmark fixture", external=c == "evidence")
        for c in REQUIRED_CHECKS
    ], forward={"state": expected}, reverse={"state": expected})

    def bench(fn, n):
        vals = []
        for _ in range(n):
            start = time.perf_counter_ns()
            value = fn()
            vals.append(time.perf_counter_ns() - start)
            if value != expected:
                raise AssertionError("result mismatch")
        return statistics.median(vals) / 1e6

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "nexus.db"
        common = dict(context_id="benchmark", revision="v1", report=report)
        with NexusStore(path) as store:
            ciu = NexusXorCIU(store)
            members = []
            state = initial
            start = time.perf_counter_ns()
            for masks in groups:
                terminal = seq(state, masks)
                members.append(ciu.record(
                    width=64, initial=state, terminal=terminal, masks=masks,
                    verify=lambda s, t=terminal: s == t, **common))
                state = terminal
            receipt = ciu.compose_receipts(
                members, initial=initial, verify=lambda s: s == expected, **common)
            build_ms = (time.perf_counter_ns() - start) / 1e6
        with NexusStore(path) as store:
            ciu = NexusXorCIU(store)

            def replay():
                return ciu.replay(receipt, initial=initial,
                    context_id="benchmark", revision="v1",
                    verify=lambda s: s == expected)

            def cached():
                node = store.read(receipt)
                p = node.payload
                if (p["initial"] != hex(initial) or p["context_id"] != "benchmark"
                        or p["revision"] != "v1"):
                    return None
                result = int(p["terminal"], 16)
                return result if result == expected else None

            seq_ms = bench(lambda: full(initial), 7)
            replay_ms = bench(replay, 1001)
            cached_ms = bench(cached, 1001)
            composed = XorTransition.compose(
                64, (mask for group in groups for mask in group))
            unseen = [rng.getrandbits(64) for _ in range(8)]
            if not all(composed.apply(s) == full(s) for s in unseen):
                raise AssertionError("fixed-mask transfer mismatch")
            print(json.dumps(dict(
                masks=1_000_000, parents=4, build_ms=build_ms,
                sequential_ms=seq_ms, nexus_replay_ms=replay_ms,
                same_receipt_cached_ms=cached_ms,
                verified_speedup=seq_ms / replay_ms,
                unseen_inputs_passed=len(unseen), nexus_nodes=store.stats()["nodes"],
                notes="Synthetic fixed masks; ancestry and terminal checks included; warm DB; cache reads same receipt."
            ), indent=2))


if __name__ == "__main__":
    main()
