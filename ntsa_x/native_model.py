"""Native NTSA-X differentiable recurrent byte stream (research architecture).

Three trainable X/Y/Z latent lanes and no Transformer, attention or teacher
inference. Latent lane names do not assert equivalence to verified orbit axes.
Optional PyTorch dependency: import this module directly.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import codecs
from typing import Iterator
import torch
from torch import nn, Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class NativeConfig:
    width: int = 96
    layers: int = 2
    dropout: float = 0.0
    max_context: int = 2048
    vocab_size: int = 262

    def __post_init__(self):
        if not (16 <= self.width <= 4096 and 1 <= self.layers <= 24
                and 0 <= self.dropout < 0.5 and 16 <= self.max_context <= 32768
                and self.vocab_size == 262):
            raise ValueError("Invalid native model config")


class ByteCodec:
    PAD, BOS, EOS, USER, ASSISTANT, SYSTEM = range(256, 262)
    ROLES = {"user": USER, "assistant": ASSISTANT, "system": SYSTEM}

    def conversation(self, messages: list[dict], *, generation=False,
                     max_tokens=2048) -> tuple[list[int], list[int]]:
        if not isinstance(messages, list) or not messages:
            raise ValueError("Expected a nonempty conversation")
        tokens, targets = [self.BOS], [-100]
        for i, message in enumerate(messages):
            if (not isinstance(message, dict)
                    or set(message) != {"role", "content"}
                    or message["role"] not in self.ROLES
                    or not isinstance(message["content"], str)
                    or len(message["content"]) > 16000):
                raise ValueError("Invalid conversation turn")
            role = self.ROLES[message["role"]]
            body = list(message["content"].encode("utf-8"))
            part = [role, *body, self.EOS]
            tokens.extend(part)
            targets.extend(([-100, *body, self.EOS]
                            if not generation and i == len(messages)-1
                            and role == self.ASSISTANT else [-100]*len(part)))
        if generation:
            if messages[-1]["role"] != "user":
                raise ValueError("Generation requires final user turn")
            tokens.append(self.ASSISTANT)
            targets.append(-100)
        elif messages[-1]["role"] != "assistant":
            raise ValueError("Training requires final assistant turn")
        if len(tokens) > max_tokens:
            raise ValueError("Exceeded native context")
        return tokens, targets


class OrbitCell(nn.Module):
    """h_t = h + alpha(x,h)*(candidate(x,h)-h); three trainable latent lanes."""

    def __init__(self, width):
        super().__init__()
        self.width = width
        self.norm = nn.LayerNorm(width)
        self.state_norm = nn.LayerNorm(width*3)
        self.input_candidate = nn.Linear(width, width*3)
        self.state_candidate = nn.Linear(width*3, width*3, bias=False)
        self.input_gate = nn.Linear(width, width*3)
        self.state_gate = nn.Linear(width*3, width*3, bias=False)
        self.router = nn.Linear(width*2, 3)
        self.readout = nn.Linear(width*2, width)
        self.output_norm = nn.LayerNorm(width)
        with torch.no_grad():
            self.input_gate.bias.copy_(
                torch.tensor([1.5, 0, -1.5]).repeat_interleave(width))

    def forward(self, x: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
        if x.ndim != 2 or state.shape != (x.shape[0], 3, self.width):
            raise ValueError("Incorrect recurrent state shape")
        x = self.norm(x)
        pooled = state.mean(dim=1)
        memory = self.state_norm(state.flatten(1))
        allocation = (3*F.softmax(self.router(torch.cat([x,pooled], -1)), -1)).unsqueeze(-1)
        gate = torch.sigmoid((self.input_gate(x)+self.state_gate(memory))
                             .reshape(-1,3,self.width))
        candidate = torch.tanh((self.input_candidate(x)+self.state_candidate(memory))
                                .reshape(-1,3,self.width))
        alpha = (allocation*gate).clamp(max=1.0)
        updated = state + alpha*(candidate-state)
        output = self.output_norm(self.readout(
            torch.cat([x, updated.mean(1)], dim=-1)))
        return output, updated


class NativeOrbitModel(nn.Module):
    """Causal recurrent state-space model; O(sequence length) token updates."""

    def __init__(self, config: NativeConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size,config.width)
        self.cells = nn.ModuleList(OrbitCell(config.width) for _ in range(config.layers))
        self.dropout = nn.Dropout(config.dropout)
        self.norm = nn.LayerNorm(config.width)
        self.head = nn.Linear(config.width,config.vocab_size)

    def initial_state(self,batch,device=None) -> tuple[Tensor,...]:
        if not isinstance(batch,int) or batch<1:
            raise ValueError("Batch size must be positive")
        base=self.embedding.weight
        return tuple(base.new_zeros(batch,3,self.config.width,
                                    device=device or base.device)
                     for _ in self.cells)

    def forward(self, tokens: Tensor, state: tuple[Tensor,...] | None=None
                ) -> tuple[Tensor,tuple[Tensor,...]]:
        if (tokens.ndim != 2 or tokens.shape[1]<1 or tokens.dtype != torch.long
                or tokens.shape[1]>self.config.max_context):
            raise ValueError("Expected bounded long tensor [batch,length]")
        if state is None:
            state=self.initial_state(tokens.shape[0],tokens.device)
        if len(state)!=len(self.cells):
            raise ValueError("Incorrect state depth")
        states=list(state)
        logits=[]
        embedded=self.dropout(self.embedding(tokens))
        for t in range(tokens.shape[1]):
            x=embedded[:,t,:]
            next_states=[]
            for layer,prior in zip(self.cells,states):
                x,new_state=layer(x,prior)
                next_states.append(new_state)
            states=next_states
            logits.append(self.head(self.norm(x)))
        return torch.stack(logits,dim=1),tuple(states)

    def loss(self,tokens:Tensor,labels:Tensor) -> Tensor:
        if tokens.shape!=labels.shape or tokens.shape[1]<2:
            raise ValueError("Invalid supervised sequence")
        logits,_=self.forward(tokens)
        gold=labels[:,1:].reshape(-1)
        if not bool((gold!=-100).any().item()):
            raise ValueError("No assistant labels")
        return F.cross_entropy(logits[:,:-1].reshape(-1,self.config.vocab_size),
                               gold,ignore_index=-100)

    def generate(self,prefix:list[int],*,max_new_tokens=128,
                 temperature=0.0) -> Iterator[int]:
        """Yield byte tokens from cached state, without building autograd graphs."""
        if not prefix or len(prefix)>self.config.max_context or not 1<=max_new_tokens<=4096 or not 0<=temperature<=2:
            raise ValueError("Invalid generation input")
        self.eval()
        # Context MUST be inside the iterator body, not on a decorator:
        # decorating a generator does not cover deferred next() execution.
        with torch.inference_mode():
            first=torch.tensor([prefix],dtype=torch.long,device=self.embedding.weight.device)
            logits,state=self.forward(first)
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


def utf8_stream(tokens: Iterator[int]) -> Iterator[str]:
    decoder=codecs.getincrementaldecoder("utf-8")(errors="replace")
    for token in tokens:
        if not 0<=token<256:
            raise ValueError("Invalid generated byte")
        value=decoder.decode(bytes([token]),final=False)
        if value:
            yield value
    remaining=decoder.decode(b"",final=True)
    if remaining:
        yield remaining
