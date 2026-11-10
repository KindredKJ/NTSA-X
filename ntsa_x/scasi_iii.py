"""SCASI III predictive preparation adapter for the NTSA-X research runtime.

This package schedules likely future knowledge needs ahead of requests. Its
bidirectional planning is an engineering heuristic, not physical retrocausality.
It does not train weights, assert truth for memory, or make external actions.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from hashlib import sha256
from threading import RLock
from typing import TYPE_CHECKING, Protocol
import json
import re
import time

if TYPE_CHECKING:
    from .memory_fabric import MemoryRecord

_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)


def tokens(value: str) -> frozenset[str]:
    return frozenset(x.casefold() for x in _TOKEN.findall(value) if len(x) > 1)


class RecallPort(Protocol):
    def recall(self, query: str, *, session: str, limit: int = 5) -> list[MemoryRecord]: ...


@dataclass(frozen=True)
class Objective:
    id: str
    signals: tuple[str, ...]
    prerequisites: tuple[str, ...]
    description: str = ""
    priority: float = 1.0

    def __post_init__(self) -> None:
        if (not re.fullmatch(r"[a-z][a-z0-9_.-]{1,63}", self.id)
                or not self.signals or not self.prerequisites
                or any(not tokens(s) or len(s) > 200 for s in self.signals + self.prerequisites)
                or not 0.01 <= self.priority <= 10.0):
            raise ValueError("Invalid bounded SCASI objective")


@dataclass(frozen=True)
class KnowledgeGap:
    prerequisite: str
    state: str    # missing, stale, unverified, or attested
    sources: tuple[str, ...]


@dataclass(frozen=True)
class Forecast:
    objective_id: str
    score: float
    forward_signal: float
    reverse_requirements: tuple[str, ...]
    knowledge_gaps: tuple[KnowledgeGap, ...]
    condensed_context: tuple[str, ...]
    prepared_before_request: bool
    provenance_digest: str

    def as_dict(self) -> dict:
        return asdict(self)


class PredictiveSuperCycle:
    """Bounded and deterministic predictor over operator-provided objectives.

    Reuses NTSA-X MemoryFabric's read-only recall interface. MemoryFabric
    currently labels recalled facts unverified; this class never promotes them
    to trusted truth. A warm cache allows preparation before a matching input.
    """

    def __init__(self, objectives: tuple[Objective, ...], *, memory: RecallPort,
                 session: str = "default", horizon_seconds: int = 120,
                 max_cache: int = 32, max_sources: int = 5):
        if (not objectives or len(objectives) > 64
                or len({x.id for x in objectives}) != len(objectives)
                or not 1 <= len(session) <= 100
                or not 1 <= horizon_seconds <= 3600
                or not 1 <= max_cache <= 256
                or not 1 <= max_sources <= 10):
            raise ValueError("Invalid bounded predictive configuration")
        self.objectives = {x.id: x for x in objectives}
        self.memory = memory
        self.session = session
        self.horizon_seconds = horizon_seconds
        self.max_cache = max_cache
        self.max_sources = max_sources
        self._cache: dict[str, tuple[float, Forecast]] = {}
        self._recent: deque[str] = deque(maxlen=16)
        self._outcomes: dict[str, deque[bool]] = defaultdict(lambda: deque(maxlen=32))
        self._lock = RLock()
        self.metrics = {"preparations": 0, "cache_hits": 0, "live_builds": 0,
                        "requests": 0}

    def _match(self, objective: Objective, request: str) -> float:
        query = tokens(request)
        if not query:
            return 0.0
        signal_sets = [tokens(s) for s in objective.signals]
        # Fraction of signal terms covered; query may be longer than signal.
        return max((len(query & s) / len(s) for s in signal_sets), default=0.0)

    def _rank(self, request: str) -> list[tuple[Objective, float, float]]:
        scored = []
        with self._lock:
            recent = tuple(self._recent)
            outcomes = {k: tuple(v) for k, v in self._outcomes.items()}
        for objective in self.objectives.values():
            forward = self._match(objective, request)
            # Recent sequential behavior influences forecasting but cannot
            # override a completely unrelated current request.
            previous = max((self._match(objective, r) for r in recent), default=0.0)
            history = outcomes.get(objective.id, ())
            success = sum(history) / len(history) if history else 0.5
            score = (0.80 * forward + 0.10 * previous + 0.10 * success)
            score *= min(objective.priority, 2.0) / 2.0
            if forward > 0:
                scored.append((objective, round(score, 6), round(forward, 6)))
        return sorted(scored, key=lambda x: (-x[1], x[0].id))

    def _build(self, objective: Objective, score: float = 0.0,
               forward: float = 0.0, *, prepared: bool = False) -> Forecast:
        gaps: list[KnowledgeGap] = []
        contexts: list[str] = []
        seen: set[str] = set()
        for prerequisite in objective.prerequisites:
            records = self.memory.recall(
                prerequisite, session=self.session, limit=self.max_sources)
            required = tokens(prerequisite)
            # Partial lexical match is not evidence of the prerequisite.
            matched = [r for r in records if required <= tokens(r.content)]
            state = "missing"
            if matched:
                fresh = [r for r in matched
                         if time.time() - r.created <= self.horizon_seconds]
                chosen = fresh or matched
                state = ("stale" if not fresh else
                         "attested" if all(r.verified for r in fresh) else "unverified")
                # Exact normalized dedup only: semantic similarity is not
                # permission to merge conflicting or unrelated assertions.
                for record in chosen:
                    content = " ".join(record.content.split())[:400]
                    normalized = content.casefold()
                    if normalized not in seen and len(contexts) < self.max_sources * 4:
                        seen.add(normalized)
                        contexts.append(f"[untrusted:{record.source[:100]}] {content}")
            gaps.append(KnowledgeGap(prerequisite, state,
                                     tuple(dict.fromkeys(r.source for r in matched))[:self.max_sources]))
        payload = {"objective": objective.id, "requirements": objective.prerequisites,
                   "gaps": [asdict(x) for x in gaps], "context": contexts}
        digest = sha256(json.dumps(payload, sort_keys=True,
                                   ensure_ascii=False).encode("utf-8")).hexdigest()
        return Forecast(objective.id, score, forward, objective.prerequisites,
                        tuple(gaps), tuple(contexts), prepared, digest)

    def warm(self, objective_ids: tuple[str, ...] | None = None) -> dict:
        """Perform read-only knowledge preparation before next user request."""
        ids = objective_ids or tuple(self.objectives)
        if len(ids) > self.max_cache or any(i not in self.objectives for i in ids):
            raise ValueError("Invalid warm objectives")
        prepared = [(ident, self._build(self.objectives[ident])) for ident in ids]
        with self._lock:
            expiry = time.monotonic() + self.horizon_seconds
            for ident, result in prepared:
                if len(self._cache) >= self.max_cache and ident not in self._cache:
                    self._cache.pop(next(iter(self._cache)))
                self._cache[ident] = (expiry, result)
            self.metrics["preparations"] += len(prepared)
        return {ident: item.as_dict() for ident, item in prepared}

    def resolve(self, request: str, *, limit: int = 3) -> dict:
        """Return matching precomputed gaps; never hallucinate missing facts."""
        if not isinstance(request, str) or len(request) > 4096 or not 1 <= limit <= 10:
            raise ValueError("Invalid request or limit")
        ranked = self._rank(request)[:limit]
        results = []
        now = time.monotonic()
        with self._lock:
            self.metrics["requests"] += 1
        for objective, score, forward in ranked:
            with self._lock:
                cached = self._cache.get(objective.id)
            if cached and cached[0] >= now:
                original = cached[1]
                result = Forecast(original.objective_id, score, forward,
                                  original.reverse_requirements,
                                  original.knowledge_gaps,
                                  original.condensed_context, True,
                                  original.provenance_digest)
                with self._lock:
                    self.metrics["cache_hits"] += 1
            else:
                result = self._build(objective, score, forward)
                with self._lock:
                    self.metrics["live_builds"] += 1
            results.append(result.as_dict())
        with self._lock:
            self._recent.append(request)
        return {"architecture": "SCASI-III x NTSA-X", "mechanism":
                "predictive-preparation-and-bidirectional-goal-constraints",
                "request": request, "forecasts": results,
                "candidate_facts_are_verified": False,
                "physical_negative_latency_claimed": False}

    def record_outcome(self, objective_id: str, *, success: bool) -> dict:
        """Adapt ranking from observed outcomes, without training model weights."""
        if objective_id not in self.objectives or type(success) is not bool:
            raise ValueError("Unknown objective or invalid outcome")
        with self._lock:
            self._outcomes[objective_id].append(success)
            history = tuple(self._outcomes[objective_id])
            return {"objective_id": objective_id, "samples": len(history),
                    "success_fraction": sum(history) / len(history),
                    "parameter_updates": 0}

    def snapshot(self) -> dict:
        with self._lock:
            return {"metrics": dict(self.metrics),
                    "warmed_objectives": list(self._cache),
                    "outcome_samples": {k: len(v) for k, v in self._outcomes.items()},
                    "trained_model": False,
                    "verified_intelligence_multiplier": None}


def native_scasi_extension(controller: PredictiveSuperCycle):
    """Optional additive hook into existing UnifiedCognitiveMechanism."""
    from .unified_cognition import NativeExtension

    def supports(request: str) -> bool:
        return bool(controller._rank(request))

    def contribute(frame) -> str:
        result = controller.resolve(frame.request, limit=1)
        forecasts = result["forecasts"]
        if not forecasts:
            return ""
        first = forecasts[0]
        gaps = ", ".join(f"{g['prerequisite']}:{g['state']}"
                         for g in first["knowledge_gaps"])
        context = " ".join(first["condensed_context"])[:700]
        return (f"SCASI III prediction ({first['objective_id']}); "
                f"prepared={first['prepared_before_request']}; "
                f"knowledge prerequisites: {gaps[:500]}; "
                f"untrusted recall: {context}")[:1500]

    return NativeExtension("scasi3.predictive", supports, contribute)
