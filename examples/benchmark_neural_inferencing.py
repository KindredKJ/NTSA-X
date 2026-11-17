"""Seeded synthetic neural applicability evaluation; NOT Auxiliary task accuracy.

Three matched conditions share held-out states and decision thresholds.
This script does not instantiate a fake Auxiliary model or task-answer solver.
"""
from __future__ import annotations
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from time import perf_counter_ns
import json
import math
import platform
import statistics
import torch
from ntsa_x.neural_inferencing import NeuralSkillConfig,NeuralSkillHead,train_skill_head

FAMILIES=("matching_trace","misleading_near_match","changed_operation",
          "changed_condition","confounded_source","distant_structural")


def generate(*,seed:int,count:int,width:int,split:str)->dict:
    generator=torch.Generator().manual_seed(seed+(0 if split=="train" else 10000000))
    current=[];prior=[];meta=[];labels=[];families=[];groups=[]
    for index in range(count):
        q=torch.randn(3,width,generator=generator)
        samples=[
            ("matching_trace",q+.11*torch.randn(3,width,generator=generator),[1,1,1,1,.6],1),
            ("misleading_near_match",q+.045*torch.randn(3,width,generator=generator),[1,1,1,.12,.1],0),
            ("changed_operation",q+.045*torch.randn(3,width,generator=generator),[0,1,1,1,.6],0),
            ("changed_condition",q+.045*torch.randn(3,width,generator=generator),[1,0,0,1,.6],0),
            ("confounded_source",q+.09*torch.randn(3,width,generator=generator),[1,1,1,.18,.1],0),
            ("distant_structural",torch.randn(3,width,generator=generator),[1,1,1,1,.6],0)]
        if index%2==0:
            samples[-1]=("distant_structural",
                .5*q+.9*torch.randn(3,width,generator=generator),[1,1,1,1,.6],0)
        for name,p,m,label in samples:
            current.append(q);prior.append(p);meta.append(m)
            labels.append(float(label));families.append(name);groups.append(index)
    return {"current":torch.stack(current),"prior":torch.stack(prior),
            "metadata":torch.tensor(meta,dtype=torch.float32),
            "label":torch.tensor(labels,dtype=torch.float32),
            "family":families,"group":groups}


def evaluate(head,examples,*,condition,seed,demos,delay,threshold=.67):
    rows=[]
    head.eval()
    with torch.inference_mode():
        for i in range(len(examples["label"])):
            q=examples["current"][i:i+1]
            t=examples["prior"][i:i+1]
            m=examples["metadata"][i:i+1]
            start=perf_counter_ns()
            if condition=="disabled":
                probability=0.
            elif condition=="raw":
                cosine=float(torch.nn.functional.cosine_similarity(
                    q.flatten(1),t.flatten(1)).item())
                probability=max(0.,cosine)*float(m[0,3])
            else:
                probability=float(head.score(q,t,m).item())
            elapsed=perf_counter_ns()-start
            allowed=(all(float(m[0,j])==1 for j in (0,1,2))
                     and float(m[0,3])>=.5)
            rows.append({"seed":seed,"demonstration_groups":demos,
                "intervening_inferences":delay,"memory_condition":condition,
                "group":examples["group"][i],"family":examples["family"][i],
                "applicable":bool(examples["label"][i].item()),
                "applied":bool(allowed and probability>=threshold),
                "score":round(probability,9),"latency_ns":elapsed,
                "outcome_observed":False})
    return rows


def summarize(rows):
    groups=defaultdict(list)
    for row in rows:
        groups[(row["memory_condition"],row["demonstration_groups"],
                row["intervening_inferences"],row["family"])].append(row)
    summary=[]
    for key,records in sorted(groups.items()):
        positive=sum(r["applicable"] for r in records)
        negative=len(records)-positive
        tp=sum(r["applicable"] and r["applied"] for r in records)
        fp=sum(not r["applicable"] and r["applied"] for r in records)
        missed=positive-tp
        latencies=sorted(r["latency_ns"] for r in records)
        summary.append({"memory_condition":key[0],"demonstration_groups":key[1],
          "intervening_inferences":key[2],"family":key[3],
          "count":len(records),
          "selection_accuracy":(tp+negative-fp)/len(records),
          "useful_selection_rate":tp/positive if positive else None,
          "false_application_rate":fp/negative if negative else None,
          "missed_recall_rate":missed/positive if positive else None,
          "brier":statistics.mean((r["score"]-int(r["applicable"]))**2 for r in records),
          "latency_p50_ns":latencies[len(latencies)//2],
          "latency_p95_ns":latencies[min(len(latencies)-1,int(.95*len(latencies)))]})
    return summary


def run(output:Path,*,seeds=(213,311,701),width=8,
        demos=(24,120,360),delays=(0,64,256),epochs=20,heldout=60):
    output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    all_rows=[]
    trainings=[]
    for seed in seeds:
        test=generate(seed=seed+809,count=heldout,width=width,split="heldout")
        for n in demos:
            train=generate(seed=seed,count=n,width=width,split="train")
            torch.manual_seed(seed)
            head=NeuralSkillHead(NeuralSkillConfig(width=width,hidden=48))
            weights_before=head.network[1].weight.detach().clone()
            metrics=train_skill_head(head,{k:train[k] for k in
                 ("current","prior","metadata","label")},epochs=epochs,
                 batch_size=48,seed=seed)
            trainings.append({"seed":seed,"demonstration_groups":n,
                "labelled_pairs":len(train["label"]),"epochs":epochs,
                "gradient_steps":epochs*math.ceil(len(train["label"])/48),
                "bce_first_epoch":metrics[0]["bce"],"bce_final_epoch":metrics[-1]["bce"],
                "parameter_count":sum(p.numel() for p in head.parameters()),
                "weights_changed":not torch.equal(weights_before,head.network[1].weight)})
            for delay in delays:
                for cond in ("disabled","raw","learned"):
                    all_rows+=evaluate(head,test,condition=cond,seed=seed,demos=n,delay=delay)
    with (output/"per_inference.jsonl").open("w",encoding="utf-8") as writer:
        for r in all_rows:
            writer.write(json.dumps(r,sort_keys=True)+"\n")
    result={"name":"synthetic-neural-applicability-v1",
            "scope":"synthetic relevance, not native Auxiliary generation",
            "fixed_seeds":list(seeds),"demonstrations":list(demos),
            "delays":list(delays),"heldout_groups_per_seed":heldout,
            "families":list(FAMILIES),"labelled_pairs_per_group":6,
            "heldout_disjoint_from_train":True,
            "same_holdout_across_conditions":True,
            "labels_available_before_evaluation":True,
            "no_future_outcomes_during_inference":True,
            "native_Auxiliary_checkpoint_loaded":False,
            "native_model_task_accuracy":None,
            "native_recall_accuracy_gain":None,
            "trainable_part":"NeuralSkillHead gradient-trained parameters",
            "explicit_part":"metadata and provenance constraints",
            "torch":torch.__version__,"hardware":platform.platform(),
            "training":trainings,"per_family":summarize(all_rows),
            "inference_rows":len(all_rows)}
    (output/"aggregate.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",
                                          encoding="utf-8")
    print(json.dumps({"rows":len(all_rows),"training_runs":len(trainings),
                      "native_accuracy":None},indent=2))
    return result


if __name__=="__main__":
    parser=ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--seeds",type=int,nargs="+",default=[213,311,701])
    parser.add_argument("--epochs",type=int,default=20)
    args=parser.parse_args()
    run(args.output,seeds=tuple(args.seeds),epochs=args.epochs)
