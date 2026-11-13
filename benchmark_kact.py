"""Reproducible, intentionally small K-ACT native-lane benchmark.

Each example asks for either first or last bit from 12 binary observations.
Heldout patterns are disjoint; task instructions are encoded as a single symbol.
This synthetic test does NOT establish language reasoning or production gains.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
import random
import statistics
import time
from dataclasses import asdict
from pathlib import Path
import torch
from torch.nn import functional as F
from ntsa_x.native_model import NativeConfig, NativeOrbitModel
from ntsa_x.kact import KACTConfig, KACTNativeOrbitModel


def dataset(seed=37, train_count=160, heldout_count=64):
    rng=random.Random(seed)
    unique=set()
    while len(unique)<train_count+heldout_count:
        bits=tuple(rng.randrange(2) for _ in range(12))
        mode=rng.randrange(2)
        unique.add((mode,*bits))
    rows=list(sorted(unique))
    rng.shuffle(rows)
    def encode(row):
        mode,*bits=row
        # X: recall the earliest bit; Y: use newest bit.
        tokens=[256,259,ord('X') if mode==0 else ord('Y')]+[48+b for b in bits]
        target=48+(bits[0] if mode==0 else bits[-1])
        return tokens,target
    return [encode(r) for r in rows[:train_count]],[encode(r) for r in rows[train_count:]]


def sample_batch(rows,indices,device):
    return (torch.tensor([rows[i][0] for i in indices],device=device,dtype=torch.long),
            torch.tensor([rows[i][1]-48 for i in indices],device=device,dtype=torch.long))


def class_logits(model,tokens):
    out,_=model(tokens)
    return out[:,-1,48:50]


@torch.inference_mode()
def eval_model(model,rows,device):
    model.eval()
    correct=0
    loss_total=0.0
    for start in range(0,len(rows),16):
        batch=rows[start:start+16]
        tok,y=sample_batch(batch,range(len(batch)),device)
        logits=class_logits(model,tok)
        correct+=int((logits.argmax(-1)==y).sum().item())
        loss_total+=float(F.cross_entropy(logits,y,reduction='sum').item())
    return {'accuracy':correct/len(rows),'cross_entropy':loss_total/len(rows)}


def train_one(model,train_rows,steps,seed,device):
    rng=random.Random(seed)
    opt=torch.optim.AdamW(model.parameters(),lr=0.003)
    model.train()
    start=time.perf_counter()
    for _ in range(steps):
        inds=[rng.randrange(len(train_rows)) for _ in range(16)]
        tok,y=sample_batch(train_rows,inds,device)
        opt.zero_grad(set_to_none=True)
        if isinstance(model,KACTNativeOrbitModel):
            logits,_,stats=model.forward_with_metrics(tok)
            loss=F.cross_entropy(logits[:,-1,48:50],y)
            loss+=model.config.compute_penalty*stats['expected_refinement_penalty']
        else:
            loss=F.cross_entropy(class_logits(model,tok),y)
        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError('Non-finite training loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
        opt.step()
    return time.perf_counter()-start


@torch.inference_mode()
def latency(model,device,repeat=30):
    model.eval()
    # Batch 1 exposes real branch-level early exit, avoiding batch padding artifacts.
    tokens=torch.tensor([[256,259,ord('X')]+[48,49]*6],device=device)
    for _ in range(5):
        model(tokens)
    timings=[]
    for _ in range(repeat):
        start=time.perf_counter_ns()
        model(tokens)
        timings.append((time.perf_counter_ns()-start)/1e6)
    result={'p50_ms':statistics.median(timings), 'p95_ms':sorted(timings)[int(0.95*(repeat-1))]}
    if isinstance(model,KACTNativeOrbitModel):
        _,_,stats=model.forward_with_metrics(tokens)
        result['dense_cell_calls']=stats['dense_cell_calls']
        result['logical_cell_updates']=float(stats['per_sample_logical_updates'][0].item())
    else:
        result['dense_cell_calls']=tokens.shape[1]*model.config.layers
        result['logical_cell_updates']=float(result['dense_cell_calls'])
    return result


def execute(steps:int=40,seed:int=113,device:str='cpu'):
    if steps<1 or steps>2000:
        raise ValueError('steps must be between 1 and 2000')
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    data_train,data_test=dataset(seed+1)
    data_hash=hashlib.sha256(json.dumps({'train':data_train,'heldout':data_test}).encode()).hexdigest()
    base=NativeOrbitModel(NativeConfig(width=16,layers=1,max_context=32)).to(device)
    base_initial={k:v.detach().clone() for k,v in base.state_dict().items()}
    configs={
        'native_baseline':None,
        'fixed_refine_no_memory':KACTConfig(width=16,layers=1,max_context=32,max_refinements=2,adaptive=True,coupling=False,working_memory=False),
        'adaptive_no_coupling':KACTConfig(width=16,layers=1,max_context=32,max_refinements=2,adaptive=True,coupling=False,working_memory=False),
        'adaptive_coupling':KACTConfig(width=16,layers=1,max_context=32,max_refinements=2,adaptive=True,coupling=True,working_memory=False),
        'adaptive_coupling_memory':KACTConfig(width=16,layers=1,max_context=32,max_refinements=2,adaptive=True,coupling=True,working_memory=True),
    }
    results={}
    for name,cfg in configs.items():
        native=NativeOrbitModel(NativeConfig(width=16,layers=1,max_context=32)).to(device)
        native.load_state_dict(base_initial)
        if cfg is None:
            model=native
        else:
            # Deterministic additional K-ACT parameters, same native initialization.
            torch.manual_seed(seed+7)
            model=KACTNativeOrbitModel(cfg).to(device).initialize_from_native(native)
            if name=='fixed_refine_no_memory':
                # During training, dense refinements are performed for both
                # fixed and adaptive policies; fixed is forced at eval too.
                for cell in model.cells:
                    cell.continue_head.bias.data.fill_(50)
                    cell.continue_head.weight.data.zero_()
                    for p in cell.continue_head.parameters():
                        p.requires_grad_(False)
        before=eval_model(model,data_test,device)
        elapsed=train_one(model,data_train,steps,seed+3,device)
        after=eval_model(model,data_test,device)
        timings=latency(model,device)
        results[name]={
            'trainable_params':sum(p.numel() for p in model.parameters() if p.requires_grad),
            'total_params':sum(p.numel() for p in model.parameters()),
            'initial_heldout':before, 'final_heldout':after,
            'training_wall_s':elapsed, 'inference':timings,
        }
    return {
        'experiment':'kact-v0.1-synthetic-recall','seed':seed,'train_steps_each':steps,
        'train_examples':len(data_train),'heldout_examples':len(data_test),
        'data_sha256':data_hash,
        'environment':{'python':platform.python_version(),'torch':torch.__version__,
                       'cpu':platform.processor() or 'unknown','device':device,
                       'threads':torch.get_num_threads()},
        'results':results,
        'limitations':['12-bit synthetic binary input, not natural language',
                       'one training seed, not variance estimate',
                       'batch-1 latency on single CPU, not general throughput',
                       'parameter counts differ; no matched-FLOP comparison',
                       'soft refinements always run in training',
                       'no PCK cross-process runtime integration'],
    }


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--steps',type=int,default=40)
    p.add_argument('--seed',type=int,default=113)
    p.add_argument('--output',type=Path,default=Path('kact_benchmark.json'))
    args=p.parse_args()
    result=execute(args.steps,args.seed)
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
