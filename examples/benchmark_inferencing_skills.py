"""Seeded independent applicability benchmark, NOT Auxiliary neural accuracy.

No symbolic answer is ever returned as the model's prediction. The oracle
only labels test states; evaluated system here is host recall/fit logic.
"""
from __future__ import annotations
from argparse import ArgumentParser
from pathlib import Path
from time import perf_counter_ns
import json

from ntsa_x.inferencing_skills import SkillLedger, RecallBeforeInference


class Probe:
    architecture = "fixture-probe"


def vector(value: int) -> list[float]:
    return [float((value >> (i % 3)) & 1) + .02 for i in range(12)]


def transition(operation: str, state: int) -> int:
    if operation == "rotate-right":
        return (state >> 1) | ((state & 1) << 2)
    if operation == "clear-lowest-set-bit":
        return state & (state - 1)
    if operation == "xor-one":
        return state ^ 1
    if operation == "rotate-then-xor":
        return transition("rotate-right", state) ^ 1
    raise ValueError(operation)


def candidates(ledger, state: int, operation: str, conditions: dict, mode: str) -> list[dict]:
    if mode == "disabled":
        return []
    assessor=object.__new__(RecallBeforeInference)
    assessor.ledger=ledger
    assessor.model=Probe()
    assessor.top_k=24
    assessor.fit_threshold=.60
    return assessor.assess(vector(state),operation,conditions)


def benchmark(folder: str | Path, *, seeds=(213,311,701),
              demonstrations=(1,2,3), delays=(0,100,500)) -> dict:
    root=Path(folder)
    root.mkdir(parents=True,exist_ok=True)
    aggregates=[]
    for seed in seeds:
        rows=[]
        for count in demonstrations:
            for delay in delays:
                for mode in ("disabled","raw","consolidated"):
                    ledger=SkillLedger()
                    demos=[4,2,6][:count]
                    tests=[1,3,5,7]
                    for state in demos:
                        ledger.add_trace(architecture="fixture-probe",
                            state=vector(state),state_transition=[.1]*12,shape=(3,4),
                            operation="rotate-right",conditions={"bits":3,"mode":"rotate"},
                            action=f"{state:03b}",
                            outcome={"next":f"{transition('rotate-right',state):03b}"},
                            evidence={"attributable":True,"source":"fixture"},
                            provenance=f"seed-{seed}:demonstration-{state}")
                    for i in range(delay):
                        ledger.event("intervening_inference",{"ordinal":i,"seed":seed})
                    trial=[]
                    for family in ("rotate-right","clear-lowest-set-bit","xor-one","rotate-then-xor"):
                        for state in tests:
                            for variation,condition in (("matched",{"bits":3,"mode":"rotate"}),
                                                        ("near_match",{"bits":3,"mode":"clear"})):
                                relevant=family=="rotate-right" and variation=="matched"
                                start=perf_counter_ns()
                                assessments=candidates(ledger,state,family,condition,mode)
                                latency=perf_counter_ns()-start
                                used=[x for x in assessments if x["eligible"]]
                                row={"seed":seed,"demonstrations":count,"delay":delay,
                                     "memory_mode":mode,"family":family,"variation":variation,
                                     "state":f"{state:03b}",
                                     "oracle_result_not_model":f"{transition(family,state):03b}",
                                     "relevant_available":relevant,"applied":bool(used),
                                     "false_application":bool(used) and not relevant,
                                     "missed_recall":relevant and not bool(used),
                                     "considered_trace_ids":[x["trace_id"] for x in assessments],
                                     "used_trace_ids":[x["trace_id"] for x in used],
                                     "retrieval_latency_ns":latency,
                                     "neural_prediction":None,"neural_exact_accuracy":None}
                                trial.append(row)
                    encoded=sum(len(json.dumps(t.__dict__).encode()) for t in ledger.get_traces())
                    result={"seed":seed,"demonstrations":count,"delay":delay,"mode":mode,
                            "n_cases":len(trial),
                            "useful_selection_rate":sum(t["applied"] and t["relevant_available"] for t in trial)/sum(t["relevant_available"] for t in trial),
                            "false_application_rate":sum(t["false_application"] for t in trial)/sum(not t["relevant_available"] for t in trial),
                            "missed_recall_rate":sum(t["missed_recall"] for t in trial)/sum(t["relevant_available"] for t in trial),
                            "retrieval_p50_ns":sorted(t["retrieval_latency_ns"] for t in trial)[len(trial)//2],
                            "memory_bytes_json":encoded,
                            "neural_exact_transition_accuracy":None,
                            "actual_checkpoint":None}
                    aggregates.append(result)
                    rows.extend(trial)
                    ledger.close()
        (root/f"seed-{seed}-inferences.jsonl").write_text(
            "\n".join(json.dumps(row,sort_keys=True) for row in rows)+"\n",encoding="utf-8")
    report={"schema_version":1,"kind":"synthetic_retrieval_applicability_not_Auxiliary",
            "source_revision":"a317863526a02f3ae1a630b2e633005abd67f09e",
            "trained_checkpoint":None,"seeds":list(seeds),"aggregates":aggregates,
            "neural_accuracy_verified":False,
            "limitations":["No trained Auxiliary checkpoint supplied",
                "No measured neural transfer improvement or retention",
                "Oracle results are independent labels and never neural outputs",
                "Latency measurements are host-specific and not seed-deterministic"]}
    (root/"aggregate.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    return report


def main():
    parser=ArgumentParser()
    parser.add_argument("--output",default=".local/rbi-results")
    parser.add_argument("--seeds",type=int,nargs="+",default=[213,311,701])
    args=parser.parse_args()
    result=benchmark(args.output,seeds=args.seeds)
    print(json.dumps({"kind":result["kind"],"runs":len(result["aggregates"]),
                      "seeds":result["seeds"],"output":args.output}))


if __name__=="__main__":
    main()
