"""SCASI III.1: predictive knowledge preparation and explicit unknown detection."""
from __future__ import annotations
from collections import deque
from dataclasses import asdict, dataclass
import re
import time
from typing import Any
from .scasi_iii import Objective, PredictiveSuperCycle, RecallPort, tokens

_PATTERNS = (
    re.compile(r"\b(?:i need to know|we need to know|verify|find out|unknown|missing)\s+(.{3,180}?)(?:[.?!;]|$)", re.I),
    re.compile(r"\b((?:what|why|how|when|where|which|who)\s+(?:is|are|was|were|does|do|did|can|should|would|will)\s+.{3,180}?)(?:[?!;]|$)", re.I),
)

def requested_knowledge(text: str, *, limit: int = 4) -> tuple[str, ...]:
    """Detect explicit requests for information; never invent the answer."""
    if not isinstance(text, str) or len(text) > 4096 or not 1 <= limit <= 8:
        raise ValueError("Invalid discovery input")
    result, seen = [], set()
    for pattern in _PATTERNS:
        for m in pattern.finditer(text):
            found = " ".join(m.group(1).strip(" .,?!;:\t").split())
            if len(tokens(found)) < 2 or found.casefold() in seen:
                continue
            result.append(found)
            seen.add(found.casefold())
            if len(result) == limit:
                return tuple(result)
    return tuple(result)

@dataclass(frozen=True)
class EvidenceGap:
    need: str
    state: str
    sources: tuple[str, ...]
    evidence_matches: int

class PredictiveKnowledgeCoordinator:
    """Compose existing NTSA-X memory and native cognition without weight edits."""
    def __init__(self, objectives: tuple[Objective, ...], *, memory: RecallPort,
                 session: str = "default", horizon_seconds: int = 120,
                 evidence_max_age_seconds: int = 30 * 86400, max_sources: int = 5):
        self.controller = PredictiveSuperCycle(
            objectives, memory=memory, session=session,
            horizon_seconds=horizon_seconds,
            evidence_max_age_seconds=evidence_max_age_seconds,
            max_sources=max_sources)
        self.memory, self.session = memory, session
        self.max_sources = max_sources
        self.evidence_max_age_seconds = evidence_max_age_seconds
        self._observations: deque[str] = deque(maxlen=32)

    def observe(self, request: str) -> None:
        if not isinstance(request, str) or not 1 <= len(request) <= 4096:
            raise ValueError("Invalid observed request")
        self._observations.append(request)

    def anticipate(self, *, max_objectives: int = 3) -> dict:
        if not 1 <= max_objectives <= min(self.controller.max_cache, len(self.controller.objectives)):
            raise ValueError("Invalid objective budget")
        ranked = []
        for goal in self.controller.objectives.values():
            values = [self.controller._match(goal, text) for text in self._observations]
            trend = sum(values) / len(values) if values else 0.0
            ranked.append((0.7 * trend + 0.3 * goal.priority / 10.0, goal.id))
        ids = tuple(i for _, i in sorted(ranked, key=lambda x: (-x[0], x[1]))[:max_objectives])
        start = time.perf_counter_ns()
        prepared = self.controller.warm(ids)
        return {"warmed": ids, "preparation_ns": time.perf_counter_ns() - start,
                "candidate_count": len(self.controller.objectives),
                "ranked_from_observations": bool(self._observations), "results": prepared}

    def _probe(self, need: str) -> EvidenceGap:
        records = self.memory.recall(need, session=self.session, limit=self.max_sources)
        relevant = tokens(need) - frozenset({
            "what", "why", "how", "when", "where", "which", "who",
            "does", "do", "did", "is", "can", "should", "would",
            "could", "will", "this", "that", "with", "from", "have",
            "been", "the", "are", "was"})
        matched = [r for r in records if relevant and
                   len(relevant & tokens(r.content)) / len(relevant) >= .75]
        if not matched:
            return EvidenceGap(need, "missing", (), 0)
        now = time.time()
        fresh = [r for r in matched if 0 <= now - r.created <= self.evidence_max_age_seconds]
        state = ("stale" if not fresh else
                 "attested" if all(r.verified for r in fresh) else "unverified")
        return EvidenceGap(need, state,
                           tuple(dict.fromkeys(r.source for r in matched))[:self.max_sources],
                           len(matched))

    def resolve(self, request: str, *, limit: int = 3, max_discoveries: int = 4) -> dict:
        existing = self.controller.resolve(request, limit=limit)
        needs = tuple(self._probe(s) for s in requested_knowledge(request, limit=max_discoveries))
        self.observe(request)
        return {**existing, "discovered_unknowns": [asdict(n) for n in needs],
                "all_discovered_unknowns_resolved": bool(needs) and all(
                    n.state == "attested" for n in needs),
                "unmatched_request": not bool(existing["forecasts"]), "model_weight_updates": 0}

    def native_extension(self):
        """Native cognition path also works for previously unseen questions."""
        from .unified_cognition import NativeExtension
        def supports(text: str) -> bool:
            return bool(self.controller._rank(text) or requested_knowledge(text))
        def contribute(frame) -> str:
            report = self.resolve(frame.request, limit=1)
            predicted = report["forecasts"][:1]
            goal = (f"objective={predicted[0]['objective_id']}; "
                    f"prepared={predicted[0]['prepared_before_request']}; "
                    if predicted else "objective=unmatched; ")
            prerequisites = (", ".join(f"{x['prerequisite']}:{x['state']}"
                             for x in predicted[0]["knowledge_gaps"]) if predicted else "")
            novel = ", ".join(f"{x['need']}:{x['state']}"
                              for x in report["discovered_unknowns"])
            return ("SCASI III.1 untrusted evidence hints; " + goal +
                    f"prerequisites=[{prerequisites}]; new_unknowns=[{novel}]; " +
                    "unknown/unverified does not imply resolved")[:1500]
        return NativeExtension("scasi3.predictive", supports, contribute)

    def attach_to_model(self, model: Any, *, existing_extensions: tuple = (),
                        persist_episodes: bool = False):
        from .unified_cognition import UnifiedCognitiveMechanism
        return UnifiedCognitiveMechanism(
            model, memory=self.memory,
            extensions=(*existing_extensions, self.native_extension()),
            session=self.session, persist_episodes=persist_episodes)

    def snapshot(self) -> dict:
        return {**self.controller.snapshot(),
                "observed_request_count": len(self._observations),
                "explicit_unknown_inference_only": True, "model_weight_updates": 0}
