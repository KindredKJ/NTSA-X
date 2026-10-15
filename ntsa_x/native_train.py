"""Train native NTSA-X recurrent parameters from consent-reviewed JSONL.

Unlike a LoRA adapter, the X/Y/Z orbit update weights themselves are optimized.
Evaluation is held out and checkpoints are isolated; no automatic promotion.
"""
from __future__ import annotations
from argparse import ArgumentParser
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import math
import os
import random
import torch
from torch import nn
from ntsa_x.native_model import ByteCodec, NativeConfig, NativeOrbitModel


def read_examples(path: str, codec: ByteCodec, context: int) -> list[tuple[list[int],list[int],str]]:
    output=[]
    for line_no,line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(),1):
        if not line.strip():
            continue
        try:
            item=json.loads(line)
            if not isinstance(item,dict) or set(item)!={"messages"}:
                raise ValueError("Expected messages list only")
            messages=item["messages"]
            ids,labels=codec.conversation(messages,max_tokens=context)
            prompt=messages[-2]["content"] if len(messages)>1 and messages[-2]["role"]=="user" else None
            if not isinstance(prompt,str) or not any(x!=-100 for x in labels):
                raise ValueError("Missing labeled assistant continuation")
            output.append((ids,labels,prompt))
        except (ValueError,TypeError,KeyError) as exc:
            raise ValueError(f"Invalid dataset at line {line_no}: {exc}") from exc
    if not output:
        raise ValueError("Dataset has no usable samples")
    return output


def make_batch(rows, device):
    width=max(len(row[0]) for row in rows)
    tokens=torch.full((len(rows),width),ByteCodec.PAD,dtype=torch.long,device=device)
    labels=torch.full((len(rows),width),-100,dtype=torch.long,device=device)
    # Teacher-forcing loss masks all prompt positions. Right-padding is safe:
    # future padding cannot affect prior state in this causal recurrence.
    for i,(ids,gold,_) in enumerate(rows):
        tokens[i,:len(ids)]=torch.tensor(ids,dtype=torch.long,device=device)
        labels[i,:len(gold)]=torch.tensor(gold,dtype=torch.long,device=device)
    return tokens,labels


@torch.no_grad()
def evaluate(model,rows,batch_size,device):
    model.eval()
    weighted_sum,count=0.0,0
    for i in range(0,len(rows),batch_size):
        tokens,labels=make_batch(rows[i:i+batch_size],device)
        n=int((labels[:,1:]!=-100).sum().item())
        loss=model.loss(tokens,labels)
        weighted_sum+=float(loss.item())*n
        count+=n
    return weighted_sum/count


def train(*,train_path,eval_path,output,width=96,layers=2,
          max_context=1024,epochs=4,batch_size=4,lr=0.001,
          gradient_clip=1.0,seed=42,device="auto"):
    if (not 1<=epochs<=100 or not 1<=batch_size<=128 or not 1e-7<=lr<=0.05
            or not 0.01<=gradient_clip<=100):
        raise ValueError("Invalid training hyperparameters")
    if device not in ("auto","cpu","cuda"):
        raise ValueError("Unsupported training device")
    if device=="cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    location=("cuda" if torch.cuda.is_available() else "cpu") if device=="auto" else device
    target=Path(output).expanduser().resolve()
    if target.exists():
        raise FileExistsError("Refusing to overwrite existing native checkpoint")
    target.parent.mkdir(parents=True,exist_ok=True)
    config=NativeConfig(width=width,layers=layers,max_context=max_context)
    codec=ByteCodec()
    training=read_examples(train_path,codec,max_context)
    evaluation=read_examples(eval_path,codec,max_context)
    train_prompts={item[2] for item in training}
    eval_prompts={item[2] for item in evaluation}
    if train_prompts & eval_prompts:
        raise ValueError("Training and evaluation share exact user prompts")
    random.seed(seed)
    torch.manual_seed(seed)
    model=NativeOrbitModel(config).to(location)
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=0.01)
    before=evaluate(model,evaluation,batch_size,location)
    history=[]
    best=float("inf")
    best_weights=None
    steps=0
    for epoch in range(epochs):
        model.train()
        random.shuffle(training)
        total_loss=0.0
        for index in range(0,len(training),batch_size):
            ids,labels=make_batch(training[index:index+batch_size],location)
            optimizer.zero_grad(set_to_none=True)
            loss=model.loss(ids,labels)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("Non-finite native training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),gradient_clip,
                                     error_if_nonfinite=True)
            optimizer.step()
            total_loss+=float(loss.detach().item())
            steps+=1
        after=evaluate(model,evaluation,batch_size,location)
        if not math.isfinite(after):
            raise RuntimeError("Non-finite held-out native loss")
        history.append({"epoch":epoch+1,"mean_training_batch_loss":
                        total_loss/math.ceil(len(training)/batch_size),
                        "heldout_loss":after})
        if after<best:
            best=after
            best_weights={k:v.detach().cpu().clone()
                          for k,v in model.state_dict().items()}
    report={
        "architecture":"ntsa-x-native-recurrent-v1",
        "transformer_layers":0,"external_teacher_calls":0,
        "trainable_parameters":sum(p.numel() for p in model.parameters()
                                   if p.requires_grad),
        "train_examples":len(training),"heldout_examples":len(evaluation),
        "optimizer_steps":steps,"device":location,"seed":seed,
        "before_heldout_loss":before,"best_heldout_loss":best,
        "best_heldout_perplexity":math.exp(min(best,50)),
        "epochs":history,
        "student_generalization_proven":False,
        "trained_model_independently_certified":False,
    }
    # Stage a complete, manifest-described checkpoint. Only best held-out
    # weights are promoted to the *new* directory, never into active serving.
    with TemporaryDirectory(prefix="ntsa-native-staging-",dir=target.parent) as stage:
        stage_path=Path(stage)
        torch.save(best_weights,stage_path/"weights.pt")
        (stage_path/"config.json").write_text(
            json.dumps(asdict(config),indent=2),encoding="utf-8")
        (stage_path/"report.json").write_text(
            json.dumps(report,indent=2),encoding="utf-8")
        os.replace(stage_path,target)
    return report


def load_checkpoint(folder,*,device="cpu"):
    path=Path(folder)
    config=NativeConfig(**json.loads((path/"config.json").read_text(encoding="utf-8")))
    model=NativeOrbitModel(config)
    # weights_only prevents loading arbitrary serialized Python objects.
    state=torch.load(path/"weights.pt",map_location="cpu",weights_only=True)
    model.load_state_dict(state,strict=True)
    return model.to(device).eval()


def main():
    parser=ArgumentParser(description="Train NTSA-X native trainable recurrence")
    parser.add_argument("--train",required=True)
    parser.add_argument("--eval",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--width",type=int,default=96)
    parser.add_argument("--layers",type=int,default=2)
    parser.add_argument("--max-context",type=int,default=1024)
    parser.add_argument("--epochs",type=int,default=4)
    parser.add_argument("--batch-size",type=int,default=4)
    parser.add_argument("--lr",type=float,default=0.001)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--device",choices=["auto","cpu","cuda"],default="auto")
    args=parser.parse_args()
    print(json.dumps(train(train_path=args.train,eval_path=args.eval,
                           output=args.output,width=args.width,layers=args.layers,
                           max_context=args.max_context,epochs=args.epochs,
                           batch_size=args.batch_size,lr=args.lr,seed=args.seed,
                           device=args.device),indent=2))


if __name__=="__main__":
    main()
