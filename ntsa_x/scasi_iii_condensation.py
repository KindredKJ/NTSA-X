"""SCASI III.2: source-preserving condensation, validated small-skill induction,
and matched NTSA-X native inference evaluation.

This is an additive research module, not model-weight training, an autonomous
verifier, or an established intelligence-density improvement.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
from hashlib import sha256
from pathlib import Path
from statistics import median
from threading import RLock
from typing import Iterable
import json
import re
import sqlite3
import time

MAX_ABS = 10**9
_ID = re.compile(r"[a-z][a-z0-9_.-]{1,63}\Z")


def _digest(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                             separators=(",", ":")).encode("utf-8")).hexdigest()


def _canonical(text: str) -> str:
    return " ".join(text.split()).casefold()


@dataclass(frozen=True)
class EvidenceClaim:
    topic: str
    claim: str
    source: str
    observed_at: float
    externally_attested: bool = False

    def __post_init__(self):
        if not all(isinstance(x, str) and 1 <= len(x.strip()) <= 512
                   for x in (self.topic, self.claim, self.source)):
            raise ValueError("Invalid evidence claim")
        if not isinstance(self.observed_at, (int, float)) or not 0 < self.observed_at < 1e12:
            raise ValueError("Invalid evidence time")
        if type(self.externally_attested) is not bool:
            raise ValueError("Invalid attestation flag")


def condense_claims(claims: Iterable[EvidenceClaim], *, max_items: int = 256) -> dict:
    """Exact dedup only. A label from a caller is not independent verification."""
    items = list(claims)
    if not 1 <= max_items <= 4096 or len(items) > max_items:
        raise ValueError("Evidence budget exceeded")
    if any(not isinstance(x, EvidenceClaim) for x in items):
        raise ValueError("Expected typed evidence claims")
    groups: dict[tuple[str, str], list[EvidenceClaim]] = {}
    topics: dict[str, set[str]] = {}
    for item in items:
        key = (_canonical(item.topic), _canonical(item.claim))
        groups.setdefault(key, []).append(item)
        topics.setdefault(key[0], set()).add(key[1])
    clusters = []
    for (topic, claim), cluster in sorted(groups.items()):
        sources = tuple(sorted({item.source for item in cluster}))
        conflict = len(topics[topic]) > 1
        clusters.append({"topic": topic, "claim": claim, "sources": sources,
                         "source_count": len(sources), "duplicate_count": len(cluster),
                         "conflicting_claims_same_topic": conflict,
                         "evidence_status": "conflicting" if conflict else "unverified",
                         "caller_attestation_labels": sum(i.externally_attested for i in cluster),
                         "lineage_digest": _digest([asdict(i) for i in cluster])})
    return {"input_count": len(items), "cluster_count": len(clusters),
            "exact_dedup_savings": len(items) - len(clusters),
            "conflict_topics": tuple(sorted(t for t, v in topics.items() if len(v) > 1)),
            "clusters": clusters, "externally_verified_by_engine": False}


@dataclass(frozen=True)
class SkillExample:
    skill_id: str
    input_value: int
    output_value: int
    partition: str    # train, validation, test
    source: str
    reviewed: bool

    def __post_init__(self):
        if (not isinstance(self.skill_id, str) or not _ID.fullmatch(self.skill_id)
                or self.partition not in {"train", "validation", "test"}
                or not isinstance(self.source, str) or not 1 <= len(self.source) <= 180
                or type(self.input_value) is not int or type(self.output_value) is not int
                or abs(self.input_value) > MAX_ABS or abs(self.output_value) > MAX_ABS
                or type(self.reviewed) is not bool):
            raise ValueError("Invalid skill example")


@dataclass(frozen=True)
class CondensedSkill:
    skill_id: str
    coefficient: int
    intercept: int
    kind: str
    train_count: int
    validation_count: int
    sources: tuple[str, ...]
    evidence_digest: str
    revision: int = 1
    parent_digest: str | None = None

    def apply(self, value: int) -> int:
        if type(value) is not int or abs(value) > MAX_ABS:
            raise ValueError("Bounded integer input required")
        outcome = self.coefficient * value + self.intercept
        if abs(outcome) > 10**12:
            raise OverflowError("Derived result exceeds bounds")
        return outcome


def induce_affine_skill(examples: Iterable[SkillExample]) -> CondensedSkill:
    """Learn a constrained affine skill, validated on disjoint reviewed examples.

    Validation examples must never participate in coefficient fitting.
    Explicit test examples cannot be supplied to this training function.
    """
    rows = tuple(examples)
    if not 5 <= len(rows) <= 256 or not all(isinstance(x, SkillExample) for x in rows):
        raise ValueError("Requires bounded, typed training and validation examples")
    ids = {x.skill_id for x in rows}
    if len(ids) != 1 or any(x.partition == "test" or not x.reviewed for x in rows):
        raise ValueError("Training examples must be reviewed and test-independent")
    train = [x for x in rows if x.partition == "train"]
    valid = [x for x in rows if x.partition == "validation"]
    if (len(train) < 3 or len(valid) < 2
            or len({x.input_value for x in train}) < 3
            or len({x.input_value for x in valid}) < 2
            or {x.input_value for x in train} & {x.input_value for x in valid}
            or {x.source for x in train} & {x.source for x in valid}):
        raise ValueError("Need distinct training/validation values and sources")
    first = train[0]
    other = next(x for x in train if x.input_value != first.input_value)
    a = Fraction(other.output_value - first.output_value,
                 other.input_value - first.input_value)
    b = Fraction(first.output_value) - a * first.input_value
    if a.denominator != 1 or b.denominator != 1:
        raise ValueError("No integral affine skill fits this training pair")
    coefficient, intercept = int(a), int(b)
    if abs(coefficient) > 10**6 or abs(intercept) > 10**9:
        raise ValueError("Skill parameters exceed allowed bounds")
    if any(coefficient * x.input_value + intercept != x.output_value for x in train):
        raise ValueError("Training examples contradict proposed skill")
    if any(coefficient * x.input_value + intercept != x.output_value for x in valid):
        raise ValueError("Independent validation failed; skill not promoted")
    kind = ("identity" if (coefficient, intercept) == (1, 0) else
            "offset" if coefficient == 1 else
            "scale" if intercept == 0 else "affine")
    return CondensedSkill(first.skill_id, coefficient, intercept, kind,
                          len(train), len(valid),
                          tuple(sorted({x.source for x in rows})),
                          _digest([asdict(x) for x in rows]))


class SkillBank:
    """Versioned SQLite store for *validated synthetic rules*, not verified facts."""
    def __init__(self, path: str | Path = ":memory:", *, writable: bool = False):
        self.writable = writable
        self._lock = RLock()
        self._db = sqlite3.connect(str(path), isolation_level=None, timeout=10,
                                   check_same_thread=False)
        if str(path) != ":memory:":
            self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("CREATE TABLE IF NOT EXISTS skill_versions ("
                         "skill_id TEXT NOT NULL, revision INTEGER NOT NULL, "
                         "payload TEXT NOT NULL, PRIMARY KEY(skill_id,revision))")
    def close(self):
        with self._lock:
            self._db.close()
    def __enter__(self):
        return self
    def __exit__(self, *_):
        self.close()
    def promote(self, skill: CondensedSkill) -> CondensedSkill:
        if not self.writable or not isinstance(skill, CondensedSkill):
            raise PermissionError("Requires authorized validated skill write")
        with self._lock:
            prior = self.get(skill.skill_id)
            if prior and prior.evidence_digest == skill.evidence_digest:
                return prior
            revision = prior.revision + 1 if prior else 1
            upgraded = CondensedSkill(skill.skill_id, skill.coefficient,
                                      skill.intercept, skill.kind, skill.train_count,
                                      skill.validation_count, skill.sources,
                                      skill.evidence_digest, revision,
                                      prior.evidence_digest if prior else None)
            self._db.execute("INSERT INTO skill_versions VALUES(?,?,?)",
                             (skill.skill_id, revision, json.dumps(asdict(upgraded),
                                                                      sort_keys=True)))
            return upgraded
    def get(self, skill_id: str) -> CondensedSkill | None:
        if not isinstance(skill_id, str) or not _ID.fullmatch(skill_id):
            raise ValueError("Invalid skill identifier")
        with self._lock:
            row = self._db.execute("SELECT payload FROM skill_versions "
                                   "WHERE skill_id=? ORDER BY revision DESC LIMIT 1",
                                   (skill_id,)).fetchone()
        if row is None:
            return None
        payload = json.loads(row[0])
        payload["sources"] = tuple(payload["sources"])
        return CondensedSkill(**payload)
    def count_versions(self, skill_id: str) -> int:
        with self._lock:
            return self._db.execute("SELECT count(*) FROM skill_versions WHERE skill_id=?",
                                    (skill_id,)).fetchone()[0]


def skill_extension(bank: SkillBank):
    """Read-only native cognition extension for explicit compute commands."""
    from .unified_cognition import NativeExtension
    pattern = re.compile(r"^compute ([a-z][a-z0-9_.-]{1,63}) (-?\d{1,10})$", re.I)
    def supports(text: str) -> bool:
        match = pattern.fullmatch(text.strip())
        return bool(match and bank.get(match.group(1)))
    def contribute(frame) -> str:
        match = pattern.fullmatch(frame.request.strip())
        if not match:
            return ""
        skill = bank.get(match.group(1))
        if skill is None:
            return ""
        result = skill.apply(int(match.group(2)))
        return (f"SCASI reviewed-example derived calculation: {result}; "
                f"skill={skill.skill_id}; rule={skill.coefficient}*x+{skill.intercept}; "
                f"revision={skill.revision}; digest={skill.evidence_digest}; "
                "not externally verified general knowledge")
    return NativeExtension("scasi3.condensation", supports, contribute)


@dataclass(frozen=True)
class HeldoutCase:
    prompt: str
    expected: str
    group: str
    def __post_init__(self):
        if any(not isinstance(x, str) or not 1 <= len(x) <= 300
               for x in (self.prompt, self.expected, self.group)):
            raise ValueError("Invalid held-out case")


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    i = (len(ordered) - 1) * .95
    lo = int(i)
    hi = min(len(ordered) - 1, lo + 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (i - lo)


def benchmark_native_model(model, cases: Iterable[HeldoutCase], *, bank: SkillBank,
                           repeats: int = 1, max_new_tokens: int = 64,
                           preparation_ns: int = 0) -> dict:
    """Measure actual model.generated bytes under matched baseline and extension.

    Requires caller-supplied loaded model; fixtures establish interface only.
    Does not use held-out answers during skill fitting or prompt construction.
    """
    from .unified_cognition import UnifiedCognitiveMechanism, BOS, USER, EOS, ASSISTANT
    rows = tuple(cases)
    if (not rows or len(rows) > 256 or not 1 <= repeats <= 10
            or not 1 <= max_new_tokens <= 512 or preparation_ns < 0
            or any(not isinstance(x, HeldoutCase) for x in rows)
            or len({x.prompt for x in rows}) != len(rows)):
        raise ValueError("Invalid evaluation set or budget")
    arms = {"baseline": UnifiedCognitiveMechanism(model),
            "condensed": UnifiedCognitiveMechanism(model,
                extensions=(skill_extension(bank),))}
    reports = {}
    for label, runtime in arms.items():
        durations, outcomes, generated_bytes, native_uses = [], [], 0, 0
        for _ in range(repeats):
            for case in rows:
                prefix = [BOS, USER, *case.prompt.encode("utf-8"), EOS, ASSISTANT]
                started = time.perf_counter_ns()
                result = bytes(runtime.generate(prefix, max_new_tokens=max_new_tokens,
                                                temperature=0.0)).decode("utf-8")
                durations.append((time.perf_counter_ns() - started) / 1e6)
                generated_bytes += len(result.encode("utf-8"))
                native_uses += int("scasi3.condensation" in runtime.last_receipt["native_extensions"])
                outcomes.append(result.strip() == case.expected.strip())
        reports[label] = {"exact_match_accuracy": sum(outcomes) / len(outcomes),
                          "correct": sum(outcomes), "total": len(outcomes),
                          "p50_response_ms": median(durations),
                          "p95_response_ms": _p95(durations),
                          "request_total_ms": sum(durations),
                          "generated_bytes": generated_bytes,
                          "condensation_extension_uses": native_uses}
    reports["condensed"]["preparation_ms"] = preparation_ns / 1e6
    reports["condensed"]["total_ms_with_preparation"] = (
        reports["condensed"]["request_total_ms"] + preparation_ns / 1e6)
    return {"evaluation_type": "matched_native_model_calls",
            "model_architecture": str(getattr(model, "architecture", type(model).__name__)),
            "checkpoint_verified": False,
            "heldout_count": len(rows), "repeats": repeats,
            "arms": reports,
            "accuracy_delta": (reports["condensed"]["exact_match_accuracy"]
                               - reports["baseline"]["exact_match_accuracy"]),
            "effective_intelligence_multiplier": None,
            "notes": "Synthetic/deterministic model outputs are not pretrained-model evidence."}
