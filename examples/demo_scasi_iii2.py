"""Offline SCASI III.2 skill compression and unseen-input demonstration."""
import json
import time
from ntsa_x.scasi_iii_condensation import (
    EvidenceClaim, SkillBank, SkillExample, condense_claims, induce_affine_skill)


def main():
    def example(x, partition, source):
        return SkillExample("demo-calibration", x, 3*x+2, partition, source, True)
    examples = [example(x, "train", "reviewed-train") for x in (0, 2, 3, 6)]
    examples += [example(x, "validation", "separate-validation") for x in (4, 8)]
    started = time.perf_counter_ns()
    skill = induce_affine_skill(examples)
    with SkillBank(writable=True) as bank:
        stored = bank.promote(skill)
        value = bank.get("demo-calibration").apply(19)
    elapsed = time.perf_counter_ns() - started
    now = time.time()
    evidence = condense_claims((
        EvidenceClaim("status", "reviewed locally", "example-a", now),
        EvidenceClaim("status", "REVIEWED  locally", "example-b", now),
    ))
    print(json.dumps({"mode": "offline-deterministic-skill-fixture",
                      "skill": stored.skill_id, "revision": stored.revision,
                      "train_count": stored.train_count,
                      "validation_count": stored.validation_count,
                      "withheld_input": 19, "withheld_output": value,
                      "expected_withheld_output": 59,
                      "matched": value == 59,
                      "evidence_clusters": evidence["cluster_count"],
                      "preparation_ms": elapsed/1e6,
                      "trained_model_inference": False,
                      "superintelligence_or_eim_verified": False}, indent=2))


if __name__ == "__main__":
    main()
