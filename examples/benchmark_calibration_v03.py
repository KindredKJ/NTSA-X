"""Three-seed synthetic calibration and adversarial evaluation.

No genuine Auxiliary checkpoint is loaded. Results are relevance-selection
metrics only and never establish native cognitive or transition improvement.
"""
from __future__ import annotations
from argparse import ArgumentParser
from pathlib import Path
import json
import torch
from ntsa_x.neural_inferencing import NeuralSkillConfig, NeuralSkillHead, train_skill_head
from ntsa_x.neural_calibration import calibrate_and_activate


def generate(seed, groups=60, width=8):
    g=torch.Generator().manual_seed(seed)
    queries=[]; references=[]; metadata=[]; labels=[]; names=[]
    for _ in range(groups):
        q=torch.randn(3,width,generator=g)
        examples=[
            ("matching",q+.11*torch.randn(3,width,generator=g),[1,1,1,1,.6],1),
            ("misleading_near",q+.045*torch.randn(3,width,generator=g),[1,1,1,.12,.1],0),
            ("operation_mismatch",q+.045*torch.randn(3,width,generator=g),[0,1,1,1,.6],0),
            ("condition_mismatch",q+.045*torch.randn(3,width,generator=g),[1,0,0,1,.6],0),
            ("unattributable",q+.09*torch.randn(3,width,generator=g),[1,1,1,.18,.1],0),
            ("unrelated",torch.randn(3,width,generator=g),[1,1,1,1,.6],0)]
        for name,p,m,y in examples:
            queries.append(q);references.append(p);metadata.append(m)
            labels.append(y);names.append(name)
    return {"current":torch.stack(queries),"prior":torch.stack(references),
            "metadata":torch.tensor(metadata,dtype=torch.float32),
            "label":torch.tensor(labels,dtype=torch.float32),"name":names}


def fit_tensors(data):
    return {k:data[k] for k in ("current","prior","metadata","label")}


def run(path:Path,seeds=(213,311,701)):
    path.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    reports=[]
    with (path/"per_inference.jsonl").open("w",encoding="utf-8") as stream:
        for seed in seeds:
            torch.manual_seed(seed)
            head=NeuralSkillHead(NeuralSkillConfig(width=8,hidden=48))
            train=train_skill_head(head,fit_tensors(generate(seed,120)),
                                    epochs=20,batch_size=48,seed=seed)
            for scenario in ("standard","high_confidence_misleading"):
                calibration=generate(seed+809)
                blind=generate(seed+1809)
                if scenario=="high_confidence_misleading":
                    calibration["metadata"][1::6]=torch.tensor([1.,1.,1.,1.,.6])
                    blind["metadata"][1::6]=torch.tensor([1.,1.,1.,1.,.6])
                with torch.no_grad():
                    score=head.score(blind["current"],blind["prior"],blind["metadata"])
                expected=blind["label"].bool()
                applied=score>=head.config.threshold
                useful=int((applied&expected).sum())
                positives=int(expected.sum())
                false=int((applied&~expected).sum())
                negatives=int((~expected).sum())
                calibration_report=calibrate_and_activate(head,fit_tensors(calibration),
                    independently_labelled=True,blind=fit_tensors(blind),
                    blind_independently_labelled=True)
                for i in range(len(score)):
                    stream.write(json.dumps({"seed":seed,"scenario":scenario,
                        "family":blind["name"][i],"index":i,
                        "gold_applicable":bool(expected[i]),
                        "score":float(score[i]),"default_applied":bool(applied[i])},
                        sort_keys=True)+"\n")
                reports.append({"seed":seed,"scenario":scenario,
                    "scope":"synthetic relevance only, no Auxiliary checkpoint",
                    "training_pairs":120*6,"train_final_bce":train[-1]["bce"],
                    "blind_positives":positives,"blind_negatives":negatives,
                    "default_useful_recall":useful/positives,
                    "default_false_application":false/negatives,
                    "calibration":calibration_report})
    (path/"aggregate.json").write_text(json.dumps(reports,indent=2,sort_keys=True)+"\n",
                                        encoding="utf-8")
    print(json.dumps([{"seed":r["seed"],"scenario":r["scenario"],
                      "useful_recall":r["default_useful_recall"],
                      "false_application":r["default_false_application"],
                      "approved":r["calibration"]["approved"]} for r in reports],indent=2))


if __name__=="__main__":
    ap=ArgumentParser()
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--seeds",type=int,nargs="+",default=[213,311,701])
    args=ap.parse_args()
    run(args.output,tuple(args.seeds))
