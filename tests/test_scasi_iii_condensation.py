"""SCASI III.2: isolated held-out transfer and source lineage tests."""
from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import re
import sys
import tempfile
import time
import unittest

from ntsa_x.scasi_iii_condensation import (
    EvidenceClaim, HeldoutCase, SkillBank, SkillExample, benchmark_native_model,
    condense_claims, induce_affine_skill, skill_extension)


def samples(skill_id="repeatable", a=2, b=3):
    train = [SkillExample(skill_id, x, a*x+b, "train", "training-corpus", True)
             for x in (0, 2, 5, 8)]
    valid = [SkillExample(skill_id, x, a*x+b, "validation", "independent-review", True)
             for x in (3, 7)]
    return train + valid


class CondensationTests(unittest.TestCase):
    def test_exact_merge_lineage_preserved(self):
        now = time.time()
        result = condense_claims([
            EvidenceClaim("Release", "Plan approved", "a", now),
            EvidenceClaim(" release ", " plan  APPROVED ", "b", now),
            EvidenceClaim("Other", "Unrelated", "c", now)])
        self.assertEqual(result["input_count"], 3)
        self.assertEqual(result["cluster_count"], 2)
        self.assertEqual(result["exact_dedup_savings"], 1)
        release = next(x for x in result["clusters"] if x["topic"] == "release")
        self.assertEqual(release["sources"], ("a", "b"))
        self.assertEqual(release["evidence_status"], "unverified")
        self.assertFalse(result["externally_verified_by_engine"])

    def test_conflicting_claims_cannot_merge(self):
        now = time.time()
        result = condense_claims([
            EvidenceClaim("Deploy", "allowed", "a", now),
            EvidenceClaim("Deploy", "not allowed", "b", now, True)])
        self.assertEqual(result["cluster_count"], 2)
        self.assertEqual(result["conflict_topics"], ("deploy",))
        self.assertTrue(all(x["evidence_status"] == "conflicting" for x in result["clusters"]))

    def test_rule_induction_and_unseen_value_transfer(self):
        skill = induce_affine_skill(samples())
        self.assertEqual((skill.coefficient, skill.intercept), (2, 3))
        self.assertEqual(skill.kind, "affine")
        self.assertEqual(skill.apply(19), 41)
        self.assertEqual(skill.train_count, 4)
        self.assertEqual(skill.validation_count, 2)
        self.assertTrue(skill.evidence_digest)

    def test_invalid_validation_and_leakage_rejected(self):
        wrong = samples()
        wrong[-1] = SkillExample("repeatable", 7, 999, "validation", "independent-review", True)
        with self.assertRaises(ValueError):
            induce_affine_skill(wrong)
        unreviewed = samples()
        unreviewed[1] = SkillExample("repeatable", 2, 7, "train", "training-corpus", False)
        with self.assertRaises(ValueError):
            induce_affine_skill(unreviewed)
        leaked = samples()
        leaked[-1] = SkillExample("repeatable", 2, 7, "validation", "independent-review", True)
        with self.assertRaises(ValueError):
            induce_affine_skill(leaked)
        exposed = samples() + [SkillExample("repeatable", 100, 203, "test", "heldout", True)]
        with self.assertRaises(ValueError):
            induce_affine_skill(exposed)

    def test_durable_bank_revision_and_no_repeat(self):
        with tempfile.TemporaryDirectory() as tmp:
            location = Path(tmp) / "skills.sqlite"
            with SkillBank(location, writable=True) as bank:
                a = bank.promote(induce_affine_skill(samples()))
                self.assertEqual(a.revision, 1)
                self.assertEqual(bank.promote(induce_affine_skill(samples())).revision, 1)
                b = bank.promote(induce_affine_skill(samples(a=3, b=1)))
                self.assertEqual(b.revision, 2)
                self.assertEqual(b.parent_digest, a.evidence_digest)
                self.assertEqual(bank.count_versions("repeatable"), 2)
            with SkillBank(location, writable=False) as reopened:
                self.assertEqual(reopened.get("repeatable").apply(10), 31)
                with self.assertRaises(PermissionError):
                    reopened.promote(induce_affine_skill(samples()))

    def test_explicit_rule_extension_is_bounded(self):
        import types
        with SkillBank(writable=True) as bank:
            bank.promote(induce_affine_skill(samples()))
            replacement = types.ModuleType("ntsa_x.unified_cognition")
            replacement.NativeExtension = lambda name, supports, contribute: SimpleNamespace(
                name=name, supports=supports, contribute=contribute)
            with patch.dict(sys.modules, {"ntsa_x.unified_cognition": replacement}):
                extension = skill_extension(bank)
            self.assertFalse(extension.supports("hi"))
            self.assertFalse(extension.supports("compute another 10"))
            self.assertTrue(extension.supports("compute repeatable 10"))
            self.assertIn("calculation: 23", extension.contribute(SimpleNamespace(request="compute repeatable 10")))

    def test_bad_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            SkillExample("bad", 10**20, 7, "train", "x", True)
        with self.assertRaises(ValueError):
            condense_claims(["not a structured claim"])
        with self.assertRaises(ValueError):
            HeldoutCase("", "0", "arithmetic")


class FixtureModel:
    config = SimpleNamespace(max_context=8192)
    architecture = "fixture-do-not-report-as-trained"
    def unified_draft(self, prompt, *, max_new_tokens, temperature):
        context = bytes(x for x in prompt if 0 <= x <= 255).decode("utf-8", errors="ignore")
        m = re.search(r"SCASI reviewed-example derived calculation: (-?\d+)", context)
        answer = m.group(1) if m else "unavailable"
        return list(answer.encode())[:max_new_tokens]


class NativeInterfaceTests(unittest.TestCase):
    def test_matched_native_model_evaluation(self):
        try:
            import ntsa_x.unified_cognition
        except ImportError:
            self.skipTest("Full NTSA-X runtime available in GitHub CI")
        with SkillBank(writable=True) as bank:
            bank.promote(induce_affine_skill(samples()))
            heldout = [HeldoutCase("compute repeatable 11", "25", "heldout"),
                       HeldoutCase("compute repeatable 19", "41", "heldout")]
            out = benchmark_native_model(FixtureModel(), heldout, bank=bank, repeats=2)
            self.assertEqual(out["arms"]["baseline"]["exact_match_accuracy"], 0.0)
            self.assertEqual(out["arms"]["condensed"]["exact_match_accuracy"], 1.0)
            self.assertEqual(out["arms"]["condensed"]["condensation_extension_uses"], 4)
            self.assertIsNone(out["effective_intelligence_multiplier"])
            self.assertFalse(out["checkpoint_verified"])


if __name__ == "__main__":
    unittest.main()
