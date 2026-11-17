"""Gradient-trained applicability inside Auxiliary's existing PDC response path.

This is a registered component of AuxiliaryModel; no second answer decoder.
Raw traces, provenance and explicit conditions remain inspectable. Learned
selection requires independently labelled positive and misleading examples.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
import json
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from .inferencing_skills import RecallBeforeInference, _conditions_match

@dataclass(frozen=True)
class NeuralSkillConfig:
    width: int
    hidden: int = 96
    max_influence: float = 0.12
    threshold: float = 0.67
    version: int = 1

    def __post_init__(self):
        if not (self.width >= 4 and self.hidden >= 8 and
                0 <= self.max_influence <= .25 and .50 <= self.threshold <= 1
                and self.version == 1):
            raise ValueError("Invalid neural skill configuration")


class NeuralSkillHead(nn.Module):
    """Pairwise learned relevance, not a second generative neural system."""

    def __init__(self, config: NeuralSkillConfig):
        super().__init__()
        self.config = config
        self.network = nn.Sequential(
            nn.LayerNorm(12), nn.Linear(12, config.hidden), nn.SiLU(),
            nn.Linear(config.hidden, config.hidden // 2), nn.SiLU(),
            nn.Linear(config.hidden // 2, 1))
        self.register_buffer("trained_examples", torch.zeros((), dtype=torch.long))

    def forward(self, current: Tensor, prior: Tensor, metadata: Tensor) -> Tensor:
        d = self.config.width
        if (current.ndim != 3 or current.shape[1:] != (3, d)
                or prior.shape != current.shape or metadata.shape != (current.shape[0], 5)):
            raise ValueError("Invalid neural relevance input shapes")
        if not all(torch.isfinite(x).all() for x in (current, prior, metadata)):
            raise ValueError("Nonfinite relevance inputs")
        a, b = current.flatten(1), prior.flatten(1)
        delta = a-b
        eps = 1e-6
        na = torch.linalg.vector_norm(a, dim=1, keepdim=True)
        nb = torch.linalg.vector_norm(b, dim=1, keepdim=True)
        nd = torch.linalg.vector_norm(delta, dim=1, keepdim=True)
        geometry = torch.cat((
            F.cosine_similarity(a,b,dim=1).unsqueeze(1),
            nd/(na+nb+eps),
            torch.abs(delta).mean(1,keepdim=True)/(torch.abs(a).mean(1,keepdim=True)+eps),
            na/(nb+eps),
            torch.abs(a).mean(1,keepdim=True),
            torch.abs(b).mean(1,keepdim=True),
            (a*b).mean(1,keepdim=True),
        ),dim=1)
        return self.network(torch.cat((geometry,metadata),dim=1)).squeeze(-1)

    def score(self, current: Tensor, prior: Tensor, metadata: Tensor) -> Tensor:
        return torch.sigmoid(self(current,prior,metadata))


def attach_neural_skill_head(model, *, hidden: int=96, threshold: float=.67,
                             max_influence: float=.12) -> NeuralSkillHead:
    from .auxiliary import AuxiliaryModel
    if not isinstance(model,AuxiliaryModel):
        raise TypeError("Attach only to the authoritative AuxiliaryModel")
    if "inferencing_skill_head" in model._modules:
        head = model._modules["inferencing_skill_head"]
        if not isinstance(head, NeuralSkillHead):
            raise ValueError("Incompatible skill head already exists")
        return head
    cfg = NeuralSkillConfig(width=model.config.width, hidden=hidden,
                            threshold=threshold, max_influence=max_influence)
    head = NeuralSkillHead(cfg).to(model.cortex.embedding.weight.device)
    model.add_module("inferencing_skill_head",head)
    return head


class NeuralRecallBeforeInference(RecallBeforeInference):
    """Learned fit, existing retrieval, same PDC projection, same unified decode."""

    def __init__(self, model, ledger, *, threshold: float | None=None,
                 hidden: int=96, top_k: int=8, max_influence: float=.12):
        self.head=attach_neural_skill_head(model,hidden=hidden,max_influence=max_influence)
        self.neural_threshold=(threshold if threshold is not None else self.head.config.threshold)
        super().__init__(model,ledger,top_k=top_k,
                         fit_threshold=self.neural_threshold,max_influence=max_influence)

    def assess(self,state:list[float],operation:str,conditions:dict)->list[dict]:
        if not self.ledger.get_traces():
            return []
        if not int(self.head.trained_examples.item()):
            candidates=super().assess(state,operation,conditions)
            for item in candidates:
                item["fit_source"]="untrained-explicit-relevance-fallback"
            return candidates
        old=self.fit_threshold
        self.fit_threshold=1e-9
        try:
            candidates=super().assess(state,operation,conditions)
        finally:
            self.fit_threshold=old
        if not candidates:
            return []
        traces={t.trace_id:t for t in self.ledger.get_traces()}
        device=next(self.head.parameters()).device
        query=torch.tensor(state,dtype=torch.float32,device=device).reshape(1,3,-1)
        priors=[]
        metadata=[]
        for candidate in candidates:
            trace=traces[candidate["trace_id"]]
            priors.append(torch.tensor(trace.state,dtype=torch.float32,
                                       device=device).reshape(3,-1))
            compatible,invalid=_conditions_match(conditions,trace.conditions)
            posterior,_,_=self.ledger.posterior(trace.skill_id,trace.conditions)
            metadata.append([float(operation==trace.operation),float(compatible),
                             float(not invalid),float(trace.confidence),float(posterior)])
        prior=torch.stack(priors)
        meta=torch.tensor(metadata,dtype=torch.float32,device=device)
        with torch.no_grad():
            scores=self.head.score(query.expand_as(prior),prior,meta).tolist()
        for candidate,score in zip(candidates,scores):
            trace=traces[candidate["trace_id"]]
            compatible=(not candidate["reasons"] and trace.evidence.get("attributable") is True)
            fit=float(score)*trace.confidence if compatible else 0.
            candidate["neural_fit"]=round(float(score),8)
            candidate["fit"]=round(fit,8)
            candidate["eligible"]=compatible and fit>=self.neural_threshold
            candidate["fit_source"]="neural-parameters-and-recorded-evidence"
        candidates.sort(key=lambda c:(-c["fit"],c["trace_id"]))
        return candidates

    def _stage_neural_training(self,inference_id:str,state:list[float],
                               considered:list[dict],selected:list[dict])->None:
        """Stage inference-time data; observations never leak into this event."""
        traces={t.trace_id:t for t in self.ledger.get_traces()}
        inference_event=self.ledger.events()[-1]
        staged=[]
        for candidate in considered:
            trace=traces[candidate["trace_id"]]
            compatible,invalid=_conditions_match(inference_event["conditions"],trace.conditions)
            posterior,_,_=self.ledger.posterior(trace.skill_id,trace.conditions)
            staged.append({
                "trace_id":trace.trace_id,"prior_state":trace.state,
                "metadata":[float(inference_event["operation"]==trace.operation),
                            float(compatible),float(not invalid),
                            trace.confidence,posterior],
                "selected":any(c["trace_id"]==trace.trace_id for c in selected)})
        self.ledger.event("neural_training_staged",{
            "inference_id":inference_id,"current_state":state,"candidates":staged,
            "model_architecture":self.model.architecture,"labels_available":False})


def evidence_training_pairs(ledger)->dict[str,Tensor]:
    """Only individually attributable, independently observed outcomes."""
    staged={e["inference_id"]:e for e in ledger.events()
            if e["kind"]=="neural_training_staged"}
    rows=[]
    for event in ledger.events():
        if event["kind"]!="outcome_observed" or event["attribution"] not in ("positive","negative"):
            continue
        current=staged.get(event["inference_id"])
        if not current:
            continue
        impacted={u["trace_id"] for u in event["updates"]}
        for c in current["candidates"]:
            if c["trace_id"] not in impacted or not c["selected"]:
                continue
            rows.append((current["current_state"],c["prior_state"],c["metadata"],
                         1. if event["attribution"]=="positive" else 0.))
    if not rows:
        raise ValueError("No individually attributable observed outcomes available")
    width=len(rows[0][0])//3
    if any(len(a)!=3*width or len(b)!=3*width for a,b,_,_ in rows):
        raise ValueError("Inconsistent recorded Auxiliary state shape")
    return {
        "current":torch.tensor([r[0] for r in rows],dtype=torch.float32).reshape(-1,3,width),
        "prior":torch.tensor([r[1] for r in rows],dtype=torch.float32).reshape(-1,3,width),
        "metadata":torch.tensor([r[2] for r in rows],dtype=torch.float32),
        "label":torch.tensor([r[3] for r in rows],dtype=torch.float32)}


def train_skill_head(head:NeuralSkillHead,train:dict[str,Tensor],*,
                     epochs:int=24,batch_size:int=64,learning_rate:float=.003,
                     seed:int=213)->list[dict]:
    """Optimize registered relevance head; never automatically update core weights."""
    if epochs<1 or batch_size<1 or not 0<learning_rate<1:
        raise ValueError("Invalid training configuration")
    if set(train)!={"current","prior","metadata","label"}:
        raise ValueError("Expected current, prior, metadata and label tensors")
    n=train["label"].numel()
    if n<2 or any(len(train[k])!=n for k in ("current","prior","metadata")):
        raise ValueError("Inconsistent training rows")
    if not torch.all((train["label"]==0)|(train["label"]==1)):
        raise ValueError("Expected binary applicability labels")
    old_determinism=torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    head.train()
    device=next(head.parameters()).device
    opt=torch.optim.AdamW(head.parameters(),lr=learning_rate,weight_decay=.001)
    report=[]
    try:
        for epoch in range(epochs):
            order=torch.randperm(n,generator=torch.Generator().manual_seed(seed+epoch))
            total=0.
            for start in range(0,n,batch_size):
                ix=order[start:start+batch_size]
                q,p,m,y=[train[k][ix].to(device) for k in
                         ("current","prior","metadata","label")]
                opt.zero_grad(set_to_none=True)
                loss=F.binary_cross_entropy_with_logits(head(q,p,m),y)
                loss.backward()
                nn.utils.clip_grad_norm_(head.parameters(),max_norm=1.)
                opt.step()
                total+=float(loss.item())*len(ix)
            report.append({"epoch":epoch+1,"bce":round(total/n,8)})
    finally:
        torch.use_deterministic_algorithms(old_determinism)
    head.trained_examples+=n*epochs
    head.eval()
    return report


def save_skill_head(head:NeuralSkillHead,path:str|Path,*,
                    base_checkpoint_sha256:str,provenance:dict)->None:
    target=Path(path)
    if target.exists():
        raise FileExistsError("Refusing to overwrite an existing neural checkpoint")
    if len(base_checkpoint_sha256)!=64 or any(c not in "0123456789abcdef" for c in base_checkpoint_sha256):
        raise ValueError("Expected SHA-256 of base Auxiliary model checkpoint")
    target.mkdir(parents=True)
    torch.save(head.state_dict(),target/"skill-head.pt")
    info={"schema_version":1,"module":"AuxiliaryModel.inferencing_skill_head",
          "config":asdict(head.config),"base_checkpoint_sha256":base_checkpoint_sha256,
          "trained_examples":int(head.trained_examples.item()),"provenance":provenance}
    (target/"config.json").write_text(json.dumps(info,indent=2,sort_keys=True)+"\n",
                                      encoding="utf-8")


def load_skill_head(model,path:str|Path,*,base_checkpoint_sha256:str)->NeuralSkillHead:
    info=json.loads((Path(path)/"config.json").read_text(encoding="utf-8"))
    if info["schema_version"]!=1 or info["base_checkpoint_sha256"]!=base_checkpoint_sha256:
        raise ValueError("Neural skill checkpoint not bound to current Auxiliary checkpoint")
    cfg=NeuralSkillConfig(**info["config"])
    if model.config.width!=cfg.width:
        raise ValueError("Native X/Y/Z state shape mismatch")
    head=attach_neural_skill_head(model,hidden=cfg.hidden,
                                 threshold=cfg.threshold,max_influence=cfg.max_influence)
    weights=torch.load(Path(path)/"skill-head.pt",map_location="cpu",weights_only=True)
    head.load_state_dict(weights,strict=True)
    if not int(head.trained_examples.item()):
        raise ValueError("Neural head contains no trained examples")
    return head
