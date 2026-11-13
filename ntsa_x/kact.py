"""K-ACT v0.1: additive adaptive recurrence for the native NTSA-X X/Y/Z lanes.

This experimental model leaves native_model.py unchanged. It has a separate
working-memory state W, zero-initialized cross-lane adapter and a differentiable
continue controller. Training uses straight-through hard halting with dense
refinement evaluation for gradients; evaluation physically skips halted steps. Per-example logical counts are not billed as physical FLOP savings.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterator
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from ntsa_x.native_model import NativeConfig, NativeOrbitModel, OrbitCell, ByteCodec


@dataclass(frozen=True)
class KACTConfig(NativeConfig):
    max_refinements: int = 2
    halt_threshold: float = 0.5
    compute_penalty: float = 0.005
    coupling: bool = True
    working_memory: bool = True
    adaptive: bool = True

    def __post_init__(self):
        super().__post_init__()
        if not (0 <= self.max_refinements <= 8
                and 0 < self.halt_threshold < 1
                and 0 <= self.compute_penalty <= 1):
            raise ValueError("Invalid K-ACT control configuration")


@dataclass
class KACTState:
    """Native lane tensors are still [batch,3,width] per original layer."""
    lanes: tuple[Tensor, ...]
    working: tuple[Tensor, ...]


class KACTCell(nn.Module):
    def __init__(self, config: KACTConfig):
        super().__init__()
        w = config.width
        self.config = config
        self.orbit = OrbitCell(w)
        self.lane_norm = nn.LayerNorm(3*w)
        self.cross_lanes = nn.Linear(3*w, 3*w, bias=False)
        nn.init.zeros_(self.cross_lanes.weight)
        self.memory_in = nn.Linear(w, w, bias=False)
        self.memory_read = nn.Linear(w, w, bias=False)
        with torch.no_grad():
            self.memory_in.weight.copy_(torch.eye(w)*0.01)
            self.memory_read.weight.copy_(torch.eye(w)*0.01)
        self.memory_candidate = nn.Linear(2*w, w)
        self.memory_gate = nn.Linear(2*w, w)
        nn.init.constant_(self.memory_gate.bias, -2.0)
        self.continue_head = nn.Linear(2*w, 1)
        nn.init.zeros_(self.continue_head.weight)
        nn.init.constant_(self.continue_head.bias, -0.5)

    def step(self, x: Tensor, lanes: Tensor, working: Tensor):
        w = self.config.width
        if lanes.shape != (x.shape[0],3,w) or working.shape != (x.shape[0],w):
            raise ValueError("Incorrect K-ACT state shape")
        conditioned = x + (self.memory_in(working) if self.config.working_memory else 0)
        output, updated = self.orbit(conditioned,lanes)
        if self.config.coupling:
            residual = self.cross_lanes(self.lane_norm(updated.flatten(1))).reshape(-1,3,w)
            updated = updated + 0.1*torch.tanh(residual)
        if self.config.working_memory:
            pair = torch.cat((output,working),-1)
            gate = torch.sigmoid(self.memory_gate(pair))
            candidate = torch.tanh(self.memory_candidate(pair))
            working = working + gate*(candidate-working)
            output = output + 0.1*self.memory_read(working)
        return output,updated,working

    def forward(self, x: Tensor, lanes: Tensor, working: Tensor):
        out,lanes,working = self.step(x,lanes,working)
        penalties = out.new_zeros(())
        dense_calls=1
        accepted_updates=x.new_full((x.shape[0],),1.0)
        if not self.config.adaptive:
            return out,lanes,working,penalties,dense_calls,accepted_updates
        for _ in range(self.config.max_refinements):
            prob = torch.sigmoid(self.continue_head(torch.cat((out,working),-1)))
            if self.training:
                # Match the forward path taken during inference. The controller
                # still receives gradient through the straight-through surrogate.
                hard=(prob>self.config.halt_threshold).to(prob.dtype)
                route=hard.detach()+prob-prob.detach()
                penalties=penalties+route.mean()
                next_out,next_lanes,next_working=self.step(out,lanes,working)
                dense_calls+=1
                out=out+route*(next_out-out)
                lanes=lanes+route.unsqueeze(-1)*(next_lanes-lanes)
                working=working+route*(next_working-working)
                accepted_updates=accepted_updates+hard.squeeze(-1).detach()
            else:
                active = (prob > self.config.halt_threshold)
                if not bool(active.any().item()):
                    break
                next_out,next_lanes,next_working=self.step(out,lanes,working)
                dense_calls += 1
                weight=active.to(dtype=out.dtype)
                out = out + weight*(next_out-out)
                lanes = lanes + weight.unsqueeze(-1)*(next_lanes-lanes)
                working = working + weight*(next_working-working)
                accepted_updates = accepted_updates + active.squeeze(-1).to(out.dtype)
        return out,lanes,working,penalties,dense_calls,accepted_updates


class KACTNativeOrbitModel(nn.Module):
    """Drop-in next-token interface, separate state type to retain W safely."""
    def __init__(self, config: KACTConfig):
        super().__init__()
        self.config = config
        self.embedding=nn.Embedding(config.vocab_size,config.width)
        self.cells=nn.ModuleList(KACTCell(config) for _ in range(config.layers))
        self.dropout=nn.Dropout(config.dropout)
        self.norm=nn.LayerNorm(config.width)
        self.head=nn.Linear(config.width,config.vocab_size)

    def initial_state(self,batch:int,device=None):
        if not isinstance(batch,int) or batch<1:
            raise ValueError("Batch size must be positive")
        base=self.embedding.weight
        device=base.device if device is None else device
        lanes=tuple(base.new_zeros((batch,3,self.config.width),device=device)
                    for _ in self.cells)
        working=tuple(base.new_zeros((batch,self.config.width),device=device)
                      for _ in self.cells)
        return KACTState(lanes,working)

    def forward_with_metrics(self,tokens:Tensor,state:KACTState|None=None):
        if (tokens.ndim!=2 or tokens.shape[1]<1 or tokens.dtype!=torch.long
                or tokens.shape[1]>self.config.max_context):
            raise ValueError("Expected bounded long tensor [batch,length]")
        if state is None:
            state=self.initial_state(tokens.shape[0],tokens.device)
        if (not isinstance(state,KACTState) or len(state.lanes)!=len(self.cells)
                or len(state.working)!=len(self.cells)):
            raise ValueError("Invalid K-ACT state depth")
        lanes=list(state.lanes)
        working=list(state.working)
        outputs=[]
        penalty=tokens.new_zeros((),dtype=self.embedding.weight.dtype)
        calls=0
        updates=tokens.new_zeros((tokens.shape[0],),dtype=self.embedding.weight.dtype)
        embedded=self.dropout(self.embedding(tokens))
        for t in range(tokens.shape[1]):
            x=embedded[:,t,:]
            new_lanes=[]
            new_working=[]
            for i,cell in enumerate(self.cells):
                x,l,w,extra,c,u=cell(x,lanes[i],working[i])
                new_lanes.append(l)
                new_working.append(w)
                penalty=penalty+extra
                calls+=c
                updates=updates+u
            lanes,working=new_lanes,new_working
            outputs.append(self.head(self.norm(x)))
        metrics={
            "dense_cell_calls":calls,
            "per_sample_logical_updates":updates.detach(),
            "expected_refinement_penalty":penalty/(tokens.shape[1]*len(self.cells)),
        }
        return torch.stack(outputs,dim=1),KACTState(tuple(lanes),tuple(working)),metrics

    def forward(self,tokens:Tensor,state:KACTState|None=None):
        logits,new_state,_=self.forward_with_metrics(tokens,state)
        return logits,new_state

    def loss(self,tokens:Tensor,labels:Tensor):
        if tokens.shape!=labels.shape or tokens.shape[1]<2:
            raise ValueError("Invalid supervised sequence")
        logits,_,metrics=self.forward_with_metrics(tokens)
        gold=labels[:,1:].reshape(-1)
        if not bool((gold!=-100).any().item()):
            raise ValueError("No supervised labels")
        ce=F.cross_entropy(logits[:,:-1,:].reshape(-1,self.config.vocab_size),
                           gold,ignore_index=-100)
        return ce+self.config.compute_penalty*metrics["expected_refinement_penalty"]

    def initialize_from_native(self,native:NativeOrbitModel):
        """Copies compatible native weights; does not modify source checkpoint."""
        if (native.config.width!=self.config.width or
                native.config.layers!=self.config.layers or
                native.config.vocab_size!=self.config.vocab_size):
            raise ValueError("Incompatible native architecture")
        with torch.no_grad():
            self.embedding.weight.copy_(native.embedding.weight)
            self.norm.load_state_dict(native.norm.state_dict())
            self.head.load_state_dict(native.head.state_dict())
            for target,source in zip(self.cells,native.cells):
                target.orbit.load_state_dict(source.state_dict())
        return self

    def generate(self,prefix:list[int],*,max_new_tokens=128,temperature=0.0)->Iterator[int]:
        if (not prefix or len(prefix)>self.config.max_context or
                not 1<=max_new_tokens<=4096 or not 0<=temperature<=2):
            raise ValueError("Invalid generation input")
        self.eval()
        with torch.inference_mode():
            inputs=torch.tensor([prefix],device=self.embedding.weight.device,dtype=torch.long)
            logits,state=self.forward(inputs)
            scores=logits[:,-1,:]
            for _ in range(max_new_tokens):
                scores=scores.clone()
                scores[:,[ByteCodec.PAD,ByteCodec.BOS,ByteCodec.USER,
                          ByteCodec.ASSISTANT,ByteCodec.SYSTEM]]=-torch.inf
                token=(scores.argmax(-1) if temperature==0 else
                       torch.multinomial(F.softmax(scores/temperature,-1),1).squeeze(-1))
                item=int(token.item())
                if item==ByteCodec.EOS:
                    break
                yield item
                logits,state=self.forward(token.view(1,1),state)
                scores=logits[:,-1,:]
