"""Train and save an Auxiliary-registered skill head from attributed outcomes.

Requires real Auxiliary checkpoint and pre-existing logged inference evidence.
Will not overwrite existing model weights or existing output directories.
"""
from __future__ import annotations
from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path
import json
from ntsa_x.auxiliary import load_checkpoint
from ntsa_x.inferencing_skills import SkillLedger
from ntsa_x.neural_inferencing import (attach_neural_skill_head,
    evidence_training_pairs, save_skill_head, train_skill_head)


def main():
    parser=ArgumentParser(description="Outcome-supervised neural skill adaptation")
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--ledger",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--seed",type=int,default=213)
    parser.add_argument("--epochs",type=int,default=24)
    parser.add_argument("--batch-size",type=int,default=64)
    parser.add_argument("--learning-rate",type=float,default=.003)
    args=parser.parse_args()
    if not args.ledger.is_file():
        raise FileNotFoundError("Recorded SQLite evidence ledger required")
    digest=sha256((args.checkpoint/"weights.pt").read_bytes()).hexdigest()
    model=load_checkpoint(args.checkpoint,device="cpu")
    ledger=SkillLedger(args.ledger)
    try:
        data=evidence_training_pairs(ledger)
    finally:
        ledger.close()
    head=attach_neural_skill_head(model)
    if data["current"].shape[2]!=head.config.width:
        raise ValueError("Evidence X/Y/Z shape differs from checkpoint")
    counts={str(n):int((data["label"]==n).sum().item()) for n in (0.,1.)}
    if min(counts.values())==0:
        raise ValueError("Need both positive and negative attributable observations")
    history=train_skill_head(head,data,epochs=args.epochs,
        batch_size=args.batch_size,learning_rate=args.learning_rate,seed=args.seed)
    report={"architecture":model.architecture,"checkpoint_sha256":digest,
            "seed":args.seed,"epochs":args.epochs,"count_by_label":counts,
            "trainable_parameters":sum(p.numel() for p in head.parameters()),
            "loss_history":history,"verified_Auxiliary_accuracy_gain":None,
            "note":"Learning loss is not downstream task efficacy"}
    save_skill_head(head,args.output,base_checkpoint_sha256=digest,
        provenance={"ledger":str(args.ledger.resolve()),"seed":args.seed,
                    "training_examples":int(data["label"].numel())})
    (args.output/"training-report.json").write_text(json.dumps(report,indent=2)+"\n",
                                                   encoding="utf-8")
    print(json.dumps(report,indent=2))

if __name__=="__main__":
    main()
