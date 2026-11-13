"""SCASI III.3: leakage-checked, compute-matched native learning experiment.

Trains *separate* COG-FU checkpoints from the same approved examples and
parent checkpoint. Contrastive condition adds a condensed, independently
validated symbolic rule to the training-only system message. The control
receives a neutral equal-byte system message. Held-out inference receives
neither message. No oracle extensions are used during evaluation.

Preparing this experiment never alters model weights or contacts providers.
Actual continuation is an explicit CLI operation that writes new checkpoints.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from statistics import median
from typing import Iterable
import json
import math
import random
import re
import time

from .scasi_iii_condensation import (
    CondensedSkill, HeldoutCase, SkillExample, induce_affine_skill,
)

_QUERY = re.compile(r"compute ([a-z][a-z0-9_.-]{1,63}) (-?(?:0|[1-9][0-9]{0,9}))\Z")


def _hash(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                             separators=(",", ":")).encode("utf-8")).hexdigest()


def _conversation(question: str, answer: str, system: str | None = None) -> dict:
    turns = [] if system is None else [{"role": "system", "content": system}]
    turns += [{"role": "user", "content": question},
              {"role": "assistant", "content": answer}]
    return {"messages": turns}


def _neutral_message(length: int) -> str:
    # ASCII ensures exactly matching UTF-8 bytes and identical token counts
    # in the native ByteCodec (which tokenizes UTF-8 bytes directly).
    return "Control data, no task rule. " + "-" * (length - len("Control data, no task rule. "))


@dataclass(frozen=True)
class AssimilationDataset:
    control_train: tuple[dict, ...]
    condensed_train: tuple[dict, ...]
    validation: tuple[dict, ...]
    skills: tuple[CondensedSkill, ...]
    evidence_digest: str
    train_samples: int
    validation_samples: int
    heldout_samples: int
    training_byte_difference: int


def prepare_assimilation(examples: Iterable[SkillExample],
                         holdout: Iterable[HeldoutCase]) -> AssimilationDataset:
    """Prepare paired datasets. Holdout is inspected for leakage, never fitted.

    Rules are fit on train split and independently checked on validation split.
    Output labels, prompt set, training order, and total byte-token counts are
    matched between control and condensed arms. Only system meaning differs.
    """
    rows = tuple(examples)
    exams = tuple(holdout)
    if not 5 <= len(rows) <= 256 or not 1 <= len(exams) <= 256:
        raise ValueError("Bounded train/validation and heldout sets are required")
    if any(not isinstance(r, SkillExample) for r in rows) or any(
            not isinstance(r, HeldoutCase) for r in exams):
        raise TypeError("Require typed reviewed examples and heldout cases")
    if any(not r.reviewed or r.partition == "test" for r in rows):
        raise ValueError("Only reviewed train/validation examples are permitted")
    seen: set[tuple[str, int]] = set()
    for item in rows:
        key = (item.skill_id, item.input_value)
        if key in seen:
            raise ValueError("Training/validation inputs overlap or are duplicated")
        seen.add(key)
    prompts: set[str] = set()
    for test in exams:
        if test.prompt in prompts:
            raise ValueError("Duplicate heldout prompts")
        prompts.add(test.prompt)
        match = _QUERY.fullmatch(test.prompt)
        if match is None:
            raise ValueError("Heldout prompts must use the compute skill_id integer format")
        if (match.group(1), int(match.group(2))) in seen:
            raise ValueError("Heldout overlaps train/validation numeric inputs")
    groups: dict[str, list[SkillExample]] = {}
    for item in rows:
        groups.setdefault(item.skill_id, []).append(item)
    if any(_QUERY.fullmatch(test.prompt).group(1) not in groups for test in exams):
        raise ValueError("Heldout refers to an untrained skill")

    control: list[dict] = []
    condensed: list[dict] = []
    validation: list[dict] = []
    skills: list[CondensedSkill] = []
    for skill_id in sorted(groups):
        group = groups[skill_id]
        skill = induce_affine_skill(group)
        skills.append(skill)
        rule_note = (f"Validated practice rule for {skill_id}: "
                     f"output = {skill.coefficient} * input + {skill.intercept}. "
                     "Use the user's input to compute the answer.")
        if not rule_note.isascii() or len(rule_note) < 28:
            raise ValueError("Rule note must be bounded ASCII")
        neutral = _neutral_message(len(rule_note))
        for example in sorted(group, key=lambda x: (x.partition, x.input_value)):
            q = f"compute {skill_id} {example.input_value}"
            answer = str(example.output_value)
            if example.partition == "train":
                control.append(_conversation(q, answer, neutral))
                condensed.append(_conversation(q, answer, rule_note))
            else:
                validation.append(_conversation(q, answer))
    # Native byte-tokenization budgets are exactly matched across control and
    # condensed arms, even for different per-skill rule descriptions.
    def count_bytes(collection: list[dict]) -> int:
        return sum(len(m["content"].encode("utf-8"))
                   for row in collection for m in row["messages"])
    difference = count_bytes(condensed) - count_bytes(control)
    if difference != 0 or len(control) != len(condensed):
        raise AssertionError("Paired training budgets not byte-matched")
    digest = _hash({"examples": [asdict(r) for r in rows],
                    "learned_rules": [asdict(s) for s in skills]})
    return AssimilationDataset(tuple(control), tuple(condensed),
                               tuple(validation), tuple(skills), digest,
                               len(control), len(validation), len(exams), difference)


def materialize_datasets(dataset: AssimilationDataset, destination: str | Path) -> dict:
    """Create new deterministic training data without overwriting any output."""
    target = Path(destination).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir(parents=False, exist_ok=False)
    outputs = {}
    for name, data in (("control_train", dataset.control_train),
                       ("condensed_train", dataset.condensed_train),
                       ("validation", dataset.validation)):
        path = target / f"{name}.jsonl"
        with path.open("x", encoding="utf-8") as stream:
            for row in data:
                stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        outputs[name] = str(path)
    manifest = {"schema": "scasi-iii3-training-v1",
                "train_samples_per_arm": dataset.train_samples,
                "validation_samples": dataset.validation_samples,
                "heldout_samples": dataset.heldout_samples,
                "training_byte_difference": dataset.training_byte_difference,
                "evidence_digest": dataset.evidence_digest,
                "skill_count": len(dataset.skills),
                "train_arm_sha256": {
                    name: sha256(Path(path).read_bytes()).hexdigest()
                    for name, path in outputs.items()}}
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {**outputs, "manifest": str(target / "manifest.json")}


def _percentile(samples: list[float], q: float) -> float:
    values = sorted(samples)
    i = (len(values) - 1) * q
    lo = int(i)
    hi = min(len(values) - 1, lo + 1)
    return values[lo] + (values[hi] - values[lo]) * (i - lo)


def _wilson(correct: int, total: int) -> tuple[float, float]:
    z = 1.959963984540054
    p = correct / total
    denom = 1.0 + z*z/total
    center = (p + z*z/(2*total))/denom
    error = z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/denom
    return (max(0.0, center-error), min(1.0, center+error))


def evaluate_checkpoint_arms(models: dict[str, object],
                            cases: Iterable[HeldoutCase], *,
                            max_new_tokens: int = 32, repeats: int = 1) -> dict:
    """Paired native *model-only* evaluation: no memory or skill extensions.

    Inputs are identical across arms; model output is the sole answer source.
    Stand-in models may be used to test wiring, never to claim neural gains.
    """
    from .unified_cognition import UnifiedCognitiveMechanism, BOS, USER, EOS, ASSISTANT
    rows = tuple(cases)
    if (not rows or len(rows) > 256 or len({c.prompt for c in rows}) != len(rows)
            or not 1 <= max_new_tokens <= 512 or not 1 <= repeats <= 10
            or any(not isinstance(c, HeldoutCase) for c in rows)
            or not {"original", "control", "assimilated"} <= models.keys()):
        raise ValueError("Invalid paired model evaluation")
    outcomes = {}
    for arm in ("original", "control", "assimilated"):
        model = models[arm]
        runtime = UnifiedCognitiveMechanism(model)
        elapsed: list[float] = []
        answers: list[str] = []
        generations = 0
        for _ in range(repeats):
            for case in rows:
                prefix = [BOS, USER, *case.prompt.encode("utf-8"), EOS, ASSISTANT]
                start = time.perf_counter_ns()
                # COG-FU unified_draft is deterministic at temperature=0.
                response = bytes(runtime.generate(prefix, max_new_tokens=max_new_tokens,
                                                  temperature=0.0)).decode("utf-8").strip()
                elapsed.append((time.perf_counter_ns() - start) / 1e6)
                answers.append(response)
                generations += len(response.encode("utf-8"))
                if runtime.last_receipt["native_extensions"]:
                    raise RuntimeError("Contaminated neural-only experiment")
        # Evaluate all repetitions, not just the first.
        correct = [answers[i] == rows[i % len(rows)].expected.strip()
                   for i in range(len(answers))]
        passed = sum(correct)
        outcomes[arm] = {"correct": passed, "total": len(correct),
                         "exact_match_accuracy": passed / len(correct),
                         "wilson95": _wilson(passed, len(correct)),
                         "p50_ms": median(elapsed), "p95_ms": _percentile(elapsed, .95),
                         "p99_ms": _percentile(elapsed, .99),
                         "total_response_ms": sum(elapsed),
                         "generated_bytes": generations,
                         "per_question_correct": [
                             all(correct[j] for j in range(i, len(correct), len(rows)))
                             for i in range(len(rows))]}
    a = outcomes["assimilated"]["per_question_correct"]
    b = outcomes["control"]["per_question_correct"]
    return {"kind": "native-checkpoint-weight-ablation",
            "model_oracle_extensions": False,
            "heldout_questions": len(rows), "repeats": repeats,
            "arms": outcomes,
            "assimilation_minus_control_accuracy": (
                outcomes["assimilated"]["exact_match_accuracy"]
                - outcomes["control"]["exact_match_accuracy"]),
            "paired_discordance": {"assimilation_only_correct": sum(x and not y for x, y in zip(a, b)),
                                   "control_only_correct": sum(y and not x for x, y in zip(a, b))},
            "eim": None, "general_reasoning_improvement_proven": False}
