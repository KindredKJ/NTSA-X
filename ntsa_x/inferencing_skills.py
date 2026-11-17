"""Auxiliary Recall-Before-Inference: evidence-addressed trace guidance.

The existing Auxiliary PDC/cortex remains the only neural inference engine.
No rule solver or answer cache. Reliability learning is an explicit Bayesian
evidence counter (not trained neural skill parameters).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from hashlib import sha256
from pathlib import Path
from time import perf_counter_ns
from typing import Any
import json
import math
import sqlite3

SCHEMA_VERSION = 1


def stable_id(namespace: str, payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return namespace + "_" + sha256(raw.encode("utf-8")).hexdigest()[:24]


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x*y for x, y in zip(a, b))
    norm = math.sqrt(sum(x*x for x in a)*sum(y*y for y in b))
    return max(-1.0, min(1.0, dot/norm)) if norm > 1e-12 else 0.0


def _conditions_match(current: dict, prior: dict) -> tuple[bool, list[str]]:
    conflicts = [key for key, value in prior.items()
                 if key in current and current[key] != value]
    unknown = [key for key in prior if key not in current]
    return not conflicts and not unknown, [*("conflict:"+k for k in conflicts),
                                           *("unknown:"+k for k in unknown)]


@dataclass(frozen=True)
class ExperienceTrace:
    trace_id: str
    schema_version: int
    model_architecture: str
    state_shape: tuple[int, int]
    state: tuple[float, ...]
    state_transition: tuple[float, ...]
    operation: str
    conditions: dict[str, Any]
    action: str
    outcome: dict[str, Any]
    evidence: dict[str, Any]
    provenance: str
    sequence: int
    confidence: float
    skill_id: str
    related_trace_ids: tuple[str, ...] = ()
    counterexample_ids: tuple[str, ...] = ()


class SkillLedger:
    """Versioned append-only event log, evidence traces, contextual posteriors."""
    def __init__(self, path: str | Path = ":memory:"):
        self.path = str(path)
        self.db = sqlite3.connect(self.path)
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS traces(id TEXT PRIMARY KEY, document TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS skill_contexts(
          skill_id TEXT NOT NULL, context_key TEXT NOT NULL,
          positive INTEGER NOT NULL DEFAULT 0, negative INTEGER NOT NULL DEFAULT 0,
          PRIMARY KEY(skill_id, context_key));
        CREATE TABLE IF NOT EXISTS events(
          sequence INTEGER PRIMARY KEY AUTOINCREMENT, event_id TEXT UNIQUE NOT NULL,
          kind TEXT NOT NULL, document TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS outcomes(
          inference_id TEXT PRIMARY KEY, document TEXT NOT NULL);
        """)
        self.db.commit()

    def close(self):
        self.db.close()

    def event(self, kind: str, payload: dict) -> str:
        next_seq = self.db.execute("SELECT COALESCE(MAX(sequence),0)+1 FROM events").fetchone()[0]
        document = {"schema_version": SCHEMA_VERSION, "sequence": next_seq,
                    "kind": kind, **payload}
        event_id = stable_id("evt", document)
        self.db.execute("INSERT INTO events(event_id,kind,document) VALUES(?,?,?)",
                        (event_id, kind, _json({"event_id": event_id, **document})))
        self.db.commit()
        return event_id

    def events(self) -> list[dict]:
        return [json.loads(row[0]) for row in self.db.execute(
            "SELECT document FROM events ORDER BY sequence")]

    def add_trace(self, *, architecture: str, state: list[float],
                  state_transition: list[float], shape: tuple[int, int],
                  operation: str, conditions: dict, action: str,
                  outcome: dict, evidence: dict, provenance: str,
                  confidence: float = 1.0, related: tuple[str, ...] = (),
                  counterexamples: tuple[str, ...] = ()) -> ExperienceTrace:
        if (not architecture or not operation or not provenance or not isinstance(conditions, dict)
            or not 0 <= confidence <= 1 or len(shape) != 2 or shape[0] != 3
            or len(state) != shape[0]*shape[1] or len(state_transition) != len(state)
            or not all(math.isfinite(x) for x in state+state_transition)):
            raise ValueError("Invalid versioned Auxiliary experience")
        skill_id = stable_id("skill", {"operation": operation,
                                       "conditions": conditions,
                                       "architecture": architecture})
        seq = self.db.execute("SELECT COALESCE(MAX(sequence),0)+1 FROM events").fetchone()[0]
        value = {"schema_version": SCHEMA_VERSION, "model_architecture": architecture,
                 "state_shape": shape, "state": state, "state_transition": state_transition,
                 "operation": operation, "conditions": conditions, "action": action,
                 "outcome": outcome, "evidence": evidence, "provenance": provenance,
                 "sequence": seq, "confidence": confidence, "skill_id": skill_id,
                 "related_trace_ids": related, "counterexample_ids": counterexamples}
        trace_id = stable_id("trace", value)
        trace = ExperienceTrace(trace_id=trace_id, **value)
        self.db.execute("INSERT OR IGNORE INTO traces VALUES(?,?)", (trace_id, _json(asdict(trace))))
        self.db.commit()
        self.event("trace_recorded", {"trace_id": trace_id, "skill_id": skill_id,
                                      "source": provenance})
        return trace

    def get_traces(self) -> list[ExperienceTrace]:
        result = []
        for (document,) in self.db.execute("SELECT document FROM traces ORDER BY rowid"):
            row = json.loads(document)
            row["state_shape"] = tuple(row["state_shape"])
            for key in ("state", "state_transition", "related_trace_ids", "counterexample_ids"):
                row[key] = tuple(row[key])
            result.append(ExperienceTrace(**row))
        return result

    def posterior(self, skill_id: str, conditions: dict) -> tuple[float, int, int]:
        key = stable_id("ctx", conditions)
        row = self.db.execute("SELECT positive,negative FROM skill_contexts WHERE skill_id=? AND context_key=?",
                              (skill_id, key)).fetchone() or (0, 0)
        return (1 + row[0]) / (2 + row[0] + row[1]), row[0], row[1]

    def observe(self, inference_id: str, *, observed: str, baseline: str | None,
                evidence_source: str, confounded: bool = False,
                outcome_verified: bool = False) -> dict:
        if not evidence_source:
            raise ValueError("Outcome evidence provenance is required")
        if type(outcome_verified) is not bool:
            raise ValueError("outcome_verified must be a bool")
        if self.db.execute("SELECT 1 FROM outcomes WHERE inference_id=?", (inference_id,)).fetchone():
            raise ValueError("Outcome already committed; no duplicate reinforcement")
        row = self.db.execute("SELECT document FROM events WHERE event_id=? AND kind='inference'",
                              (inference_id,)).fetchone()
        if not row:
            raise ValueError("Unknown inference receipt")
        event = json.loads(row[0])
        # Only a native pre-outcome paired decoder result may constitute the
        # baseline. Caller-provided strings cannot independently establish gain.
        recorded = event.get("counterfactual_prediction")
        trusted = (recorded is not None and
                   event.get("counterfactual_protocol") == "auxiliary-native-paired-v1" and
                   event.get("counterfactual_seed") == event.get("seed") and
                   event.get("counterfactual_before_outcome") is True and
                   event.get("counterfactual_model_architecture") == event.get("architecture"))
        if trusted and baseline is not None and baseline != recorded:
            raise ValueError("Caller baseline differs from native paired result")
        comparison_known = trusted and outcome_verified and not confounded
        assisted = event["prediction"] == observed
        unassisted = recorded == observed if trusted else None
        attribution = ("positive" if comparison_known and assisted and not unassisted else
                       "negative" if comparison_known and unassisted and not assisted else
                       "provisional")
        update = []
        # Separate ablations are needed to attribute multi-skill outcomes.
        if len(event["used"]) != 1:
            attribution = "provisional"
        if attribution != "provisional":
            for used in event["used"]:
                for trace_id in used["trace_ids"]:
                    trace = next(t for t in self.get_traces() if t.trace_id == trace_id)
                    key = stable_id("ctx", trace.conditions)
                    self.db.execute("""INSERT INTO skill_contexts(skill_id,context_key,positive,negative)
                        VALUES(?,?,?,?) ON CONFLICT(skill_id,context_key) DO UPDATE SET
                        positive=positive+excluded.positive, negative=negative+excluded.negative""",
                        (trace.skill_id, key, int(attribution == "positive"),
                         int(attribution == "negative")))
                    update.append({"skill_id": trace.skill_id, "trace_id": trace_id,
                                   "classification": attribution})
        receipt = {"inference_id": inference_id, "observed": observed,
                   "comparison_baseline": recorded if trusted else None,
                   "counterfactual_verified_in_receipt": bool(trusted),
                   "outcome_verified": outcome_verified,
                   "evidence_source": evidence_source,
                   "confounded": confounded, "attribution": attribution, "updates": update}
        self.db.execute("INSERT INTO outcomes VALUES(?,?)", (inference_id, _json(receipt)))
        self.db.commit()
        self.event("outcome_observed", receipt)
        return receipt

    def snapshot(self, filename: str | Path) -> None:
        target = sqlite3.connect(str(filename))
        try:
            self.db.backup(target)
        finally:
            target.close()


class RecallBeforeInference:
    """Runs inside Auxiliary's existing PDC and COG-FU recurrence only."""
    def __init__(self, model, ledger: SkillLedger, *, top_k: int = 8,
                 fit_threshold: float = 0.65, max_influence: float = 0.12):
        from .auxiliary import AuxiliaryModel
        if not isinstance(model, AuxiliaryModel):
            raise TypeError("Expected the authoritative AuxiliaryModel")
        if not (1 <= top_k <= 32 and 0 < fit_threshold <= 1
                and 0 <= max_influence <= 0.25):
            raise ValueError("Invalid recall/guidance budget")
        self.model, self.ledger = model, ledger
        self.top_k, self.fit_threshold = top_k, fit_threshold
        self.max_influence = max_influence

    def assess(self, state: list[float], operation: str, conditions: dict) -> list[dict]:
        scored = []
        for trace in self.ledger.get_traces():
            similarity = max(0.0, _cosine(state, list(trace.state)))
            compatible, reasons = _conditions_match(conditions, trace.conditions)
            if operation != trace.operation:
                compatible = False
                reasons.append("operation_mismatch")
            if trace.model_architecture != self.model.architecture or len(state) != len(trace.state):
                compatible = False
                reasons.append("state_or_architecture_mismatch")
            if trace.evidence.get("attributable") is not True:
                compatible = False
                reasons.append("attribution_unverified")
            posterior, pos, neg = self.ledger.posterior(trace.skill_id, trace.conditions)
            evidence_reliability = trace.confidence * (0.5 + 0.5*posterior)
            # Structural compatibility permits transfer to novel states;
            # latent similarity ranks the otherwise applicable experiences.
            fit = (0.8 + 0.2*similarity) * evidence_reliability if compatible else 0.0
            scored.append({"trace_id": trace.trace_id, "skill_id": trace.skill_id,
                           "similarity": round(similarity, 8),
                           "reliability": round(evidence_reliability, 8),
                           "fit": round(fit, 8), "reasons": reasons,
                           "positive_updates": pos, "negative_updates": neg,
                           "eligible": compatible and fit >= self.fit_threshold})
        scored.sort(key=lambda x: (-x["similarity"], x["trace_id"]))
        candidates = scored[:self.top_k]
        # Different outputs for different input states are NOT contradictions.
        by_id = {t.trace_id: t for t in self.ledger.get_traces()}
        conflict_ids = set()
        for i, a in enumerate(candidates):
            if not a["eligible"]:
                continue
            for b in candidates[i+1:]:
                if (b["eligible"] and by_id[a["trace_id"]].action == by_id[b["trace_id"]].action
                    and by_id[a["trace_id"]].conditions == by_id[b["trace_id"]].conditions
                    and by_id[a["trace_id"]].outcome != by_id[b["trace_id"]].outcome):
                    conflict_ids.update((a["trace_id"], b["trace_id"]))
        for item in candidates:
            if item["trace_id"] in conflict_ids:
                item["eligible"] = False
                item["reasons"].append("contradictory_evidence")
        return candidates

    def infer(self, prefix: list[int], *, operation: str, conditions: dict,
              seed: int = 0, max_new_tokens: int = 32,
              memory_mode: str = "consolidated",
              measure_counterfactual: bool = False) -> tuple[list[int], dict]:
        """Only prior evidence may influence this inference; never future outcomes."""
        import torch
        from .cog_fu import CogFuModel
        if memory_mode not in ("off", "raw", "consolidated"):
            raise ValueError("Memory mode must be off/raw/consolidated")
        with torch.inference_mode():
            self.model.eval()
            start = perf_counter_ns()
            projection, receipt = self.model.resolve_projections(prefix, seed=seed)
            state = projection.flatten().detach().cpu().tolist()
            self.ledger.event("situation_encoded", {"architecture": self.model.architecture,
                                                      "state_shape": list(projection.shape),
                                                      "seed": seed})
            recall_start = perf_counter_ns()
            considered = [] if memory_mode == "off" else self.assess(state, operation, conditions)
            retrieval_ns = perf_counter_ns() - recall_start
            self.ledger.event("candidate_recall", {"considered": [c["trace_id"] for c in considered]})
            self.ledger.event("applicability_assessed", {"candidates": considered})
            selected = [c for c in considered if c["eligible"]]
            selected.sort(key=lambda c: (-c["fit"], c["trace_id"]))
            selected = selected[:3]
            unmodified_projection = projection.clone()
            if selected and self.max_influence:
                traces = {t.trace_id: t for t in self.ledger.get_traces()}
                weighted = torch.zeros_like(projection)
                total_weight = 0.0
                for c in selected:
                    trace = traces[c["trace_id"]]
                    weight = c["fit"] if memory_mode == "raw" else c["fit"]*c["reliability"]
                    weighted += torch.tensor(trace.state_transition, dtype=projection.dtype,
                                             device=projection.device).reshape_as(projection)*weight
                    total_weight += weight
                if total_weight:
                    guidance = weighted / total_weight
                    scale = projection.norm().clamp_min(1e-6)*self.max_influence
                    guidance = guidance * (scale / guidance.norm().clamp_min(1e-6)).clamp(max=1)
                    projection = projection + guidance
            used = [{"skill_id": c["skill_id"], "trace_ids": [c["trace_id"]],
                     "fit": c["fit"]} for c in selected] if self.max_influence else []
            self.ledger.event("guidance_applied", {"used": used,
                                                    "influence_cap": self.max_influence})
            # Projection readout and response are the existing unified model,
            # not a second solver, independent assistant, or answer cache.
            context = self.model.projected_context(projection)
            self.model.last_projection_receipt = receipt
            counterfactual_prediction = None
            counterfactual_ns = 0
            if measure_counterfactual and selected:
                baseline_context = self.model.projected_context(unmodified_projection)
                paired_start = perf_counter_ns()
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(seed)
                    baseline_output = CogFuModel.unified_draft(
                        self.model, prefix, max_new_tokens=max_new_tokens,
                        temperature=0.0, cached_context=baseline_context)
                counterfactual_prediction = bytes(baseline_output).decode(
                    "utf-8", errors="replace")
                counterfactual_ns = perf_counter_ns()-paired_start
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                output = CogFuModel.unified_draft(
                    self.model, prefix, max_new_tokens=max_new_tokens,
                    temperature=0.0, cached_context=context)
            prediction = bytes(output).decode("utf-8", errors="replace")
            self.ledger.event("prediction_generated", {"prediction": prediction, "seed": seed})
            inference_payload = {"architecture": self.model.architecture,
                                 "operation": operation, "conditions": conditions,
                                 "memory_mode": memory_mode, "considered": considered,
                                 "used": used, "prediction": prediction,
                                 "retrieval_ns": retrieval_ns,
                                 "inference_ns": perf_counter_ns()-start,
                                 "pdc": receipt.to_dict(),
                                 "seed": seed,
                                 "counterfactual_prediction": counterfactual_prediction,
                                 "counterfactual_protocol": (
                                     "auxiliary-native-paired-v1" if counterfactual_prediction is not None else None),
                                 "counterfactual_seed": (seed if counterfactual_prediction is not None else None),
                                 "counterfactual_model_architecture": (
                                     self.model.architecture if counterfactual_prediction is not None else None),
                                 "counterfactual_before_outcome": counterfactual_prediction is not None,
                                 "counterfactual_ns": counterfactual_ns,
                                 "attribution": "not_yet_observed"}
            inference_id = self.ledger.event("inference", inference_payload)
            stage = getattr(self, "_stage_neural_training", None)
            if stage is not None:
                # The already-scanned latent state is staged before any outcome
                # observation; this never performs an online learning update.
                stage(inference_id, state, considered, selected)
            return output, {"inference_id": inference_id, **inference_payload}
