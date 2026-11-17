"""Synthetic SCASI III.1 microbenchmark. Retrieval only; not model inference."""
from argparse import ArgumentParser
from dataclasses import dataclass
from statistics import median
import time
import json
from ntsa_x.scasi_iii import Objective, PredictiveSuperCycle
from ntsa_x.scasi_iii_evolution import PredictiveKnowledgeCoordinator

@dataclass(frozen=True)
class Item:
    content: str
    source: str
    created: float
    verified: bool = False

class Recall:
    def __init__(self):
        now = time.time()
        self.records = [Item(s, "synthetic", now) for s in (
            "release manifest signed", "rollback plan reviewed",
            "training data rights confirmed", "held out evaluation prepared")]
        self.calls = 0
    def recall(self, query, *, session, limit=5):
        self.calls += 1
        words = set(query.casefold().split())
        return [r for r in self.records if words & set(r.content.split())][:limit]

GOALS = (
    Objective("deploy", ("deploy service",), ("release manifest", "rollback plan")),
    Objective("training", ("train model",), ("training data rights", "held out evaluation")),
)
def q(values, p):
    s = sorted(values)
    i = (len(s)-1)*p
    a, b = int(i), min(len(s)-1, int(i)+1)
    return s[a] + (s[b]-s[a])*(i-a)

def run_benchmark(requests=300, repeats=7):
    if not 10 <= requests <= 20000 or not 1 <= repeats <= 30:
        raise ValueError("Benchmark bounds exceeded")
    workload = ["deploy service now" if i % 5 else "train model now"
                for i in range(requests)]
    trials = {x: [] for x in ("direct_recall", "cold_scasi", "warmed_scasi")}
    for _ in range(repeats):
        for mode in trials:
            mem = Recall()
            preparation_ms = 0.0
            if mode == "cold_scasi":
                core = PredictiveSuperCycle(GOALS, memory=mem)
            if mode == "warmed_scasi":
                coordinator = PredictiveKnowledgeCoordinator(GOALS, memory=mem)
                for req in workload[:20]:
                    coordinator.observe(req)
                preparation_ms = coordinator.anticipate(max_objectives=1)["preparation_ns"]/1e6
            durations = []
            for req in workload:
                start = time.perf_counter_ns()
                if mode == "direct_recall":
                    goal = GOALS[0] if req.startswith("deploy") else GOALS[1]
                    for prerequisite in goal.prerequisites:
                        mem.recall(prerequisite, session="benchmark")
                elif mode == "cold_scasi":
                    core.resolve(req, limit=1)
                else:
                    coordinator.resolve(req, limit=1)
                durations.append((time.perf_counter_ns()-start)/1e6)
            trial = {"p50_ms": q(durations, .5), "p95_ms": q(durations, .95),
                     "p99_ms": q(durations, .99), "request_total_ms": sum(durations),
                     "preparation_ms": preparation_ms,
                     "total_ms": sum(durations)+preparation_ms,
                     "memory_recall_calls": mem.calls}
            if mode == "warmed_scasi":
                trial["warm_hits"] = coordinator.controller.metrics["cache_hits"]
                trial["warm_hit_rate"] = trial["warm_hits"]/requests
            trials[mode].append(trial)
    return {"benchmark_type": "synthetic-memory-adapter-only",
            "workload": {"requests": requests, "repeats": repeats,
                         "deploy_fraction": .8, "trained_model_tokens": 0},
            "comparisons": {key: {field: round(median([x[field] for x in vals]), 6)
                                   for field in vals[0]} for key, vals in trials.items()},
            "notes": ["Preparation included in total", "Not end-to-end model inference",
                      "No EIM or knowledge correctness demonstrated"]}
def main():
    p = ArgumentParser()
    p.add_argument("--requests", type=int, default=300)
    p.add_argument("--repeats", type=int, default=7)
    a = p.parse_args()
    print(json.dumps(run_benchmark(a.requests, a.repeats), indent=2))
if __name__ == "__main__":
    main()
