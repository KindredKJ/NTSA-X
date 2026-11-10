"""SCASI III.3 leakage and controlled neural evaluation regression tests."""
from __future__ import annotations
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
import sys
import tempfile
import unittest

from ntsa_x.scasi_iii_condensation import HeldoutCase, SkillExample
from ntsa_x.scasi_iii_assimilation import (
    evaluate_checkpoint_arms, materialize_datasets, prepare_assimilation,
)


def fixtures():
    examples = [SkillExample("ruleone", x, 3*x + 2, "train", "reviewed-train", True)
                for x in (0, 1, 3, 5)]
    examples += [SkillExample("ruleone", x, 3*x + 2, "validation", "external-validation", True)
                 for x in (4, 6)]
    heldout = [HeldoutCase(f"compute ruleone {x}", str(3*x+2), "sealed")
               for x in (9, 17, 21)]
    return examples, heldout


class AssimilationDataTests(unittest.TestCase):
    def test_matched_budget_labels_and_prompt_order(self):
        samples, sealed = fixtures()
        paired = prepare_assimilation(samples, sealed)
        self.assertEqual(paired.training_byte_difference, 0)
        self.assertEqual(paired.train_samples, 4)
        self.assertEqual(paired.validation_samples, 2)
        self.assertEqual(paired.heldout_samples, 3)
        for control, condensed in zip(paired.control_train, paired.condensed_train):
            self.assertEqual(control["messages"][1:], condensed["messages"][1:])
            self.assertEqual(len(control["messages"][0]["content"].encode()),
                             len(condensed["messages"][0]["content"].encode()))
            self.assertIn("Validated practice rule", condensed["messages"][0]["content"])
            self.assertNotIn("Validated practice rule", control["messages"][0]["content"])
        self.assertFalse(any("21" in str(v) for v in paired.condensed_train))

    def test_versioned_inputs_produce_stable_digest(self):
        a, b = fixtures()
        self.assertEqual(prepare_assimilation(a, b).evidence_digest,
                         prepare_assimilation(a, b).evidence_digest)

    def test_leaked_train_input_disallowed(self):
        a, _ = fixtures()
        with self.assertRaisesRegex(ValueError, "overlaps"):
            prepare_assimilation(a, [HeldoutCase("compute ruleone 3", "11", "sealed")])

    def test_duplicate_train_validation_disallowed(self):
        a, b = fixtures()
        a.append(SkillExample("ruleone", 3, 11, "validation", "other", True))
        with self.assertRaisesRegex(ValueError, "overlap"):
            prepare_assimilation(a, b)

    def test_unreviewed_rules_disallowed(self):
        a, b = fixtures()
        a[0] = SkillExample("ruleone", 0, 2, "train", "reviewed-train", False)
        with self.assertRaises(ValueError):
            prepare_assimilation(a, b)

    def test_unknown_heldout_skill_disallowed(self):
        a, _ = fixtures()
        with self.assertRaises(ValueError):
            prepare_assimilation(a, [HeldoutCase("compute unknown 10", "99", "sealed")])

    def test_bad_heldout_format_disallowed(self):
        a, _ = fixtures()
        with self.assertRaises(ValueError):
            prepare_assimilation(a, [HeldoutCase("evaluate ruleone please", "20", "sealed")])

    def test_independent_validation_still_required(self):
        a, b = fixtures()
        a[-1] = SkillExample("ruleone", 6, 999, "validation", "external-validation", True)
        with self.assertRaises(ValueError):
            prepare_assimilation(a, b)

    def test_native_cog_fu_parser_when_available(self):
        try:
            from ntsa_x.cog_fu import CogFuConfig
            from ntsa_x.cog_fu_train import read_conversations
        except ImportError:
            self.skipTest("PyTorch native COG-FU modules are available in GitHub native CI")
        a, b = fixtures()
        with tempfile.TemporaryDirectory() as tmp:
            output = materialize_datasets(prepare_assimilation(a,b), Path(tmp)/"data")
            config = CogFuConfig(width=16, layers=1, max_context=512,
                                 max_answer=32, diffusion_layers=1, denoise_steps=1)
            normal = read_conversations(output["control_train"], config)
            guided = read_conversations(output["condensed_train"], config)
            dev = read_conversations(output["validation"], config)
            self.assertEqual(len(normal), len(guided))
            self.assertEqual(len(dev), 2)
            self.assertEqual([row[-2:] for row in normal],
                             [row[-2:] for row in guided])

    def test_materialize_never_overwrites(self):
        a, b = fixtures()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prepared"
            files = materialize_datasets(prepare_assimilation(a,b), path)
            self.assertEqual(len(Path(files["control_train"]).read_text().splitlines()), 4)
            manifest = json.loads(Path(files["manifest"]).read_text())
            self.assertEqual(manifest["training_byte_difference"], 0)
            with self.assertRaises(FileExistsError):
                materialize_datasets(prepare_assimilation(a,b), path)


class FakeNativeModel:
    architecture = "test-model-not-a-trained-NTSA-model"
    config = SimpleNamespace(max_context=512)
    def __init__(self, answer):
        self.answer = answer
    def unified_draft(self, prefix, *, max_new_tokens, temperature):
        return list(self.answer.encode()[:max_new_tokens])


class FakeMechanism:
    def __init__(self, model):
        self.model = model
        self.last_receipt = {"native_extensions": []}
    def generate(self, prefix, *, max_new_tokens, temperature):
        yield from self.model.unified_draft(prefix, max_new_tokens=max_new_tokens,
                                          temperature=temperature)


class AssimilationEvalTests(unittest.TestCase):
    def test_three_identical_input_arms_and_discordance(self):
        # Only mock the native model interface. Never present fixture gains
        # as measured neural improvements.
        stub = ModuleType("ntsa_x.unified_cognition")
        stub.UnifiedCognitiveMechanism = FakeMechanism
        stub.BOS, stub.USER, stub.EOS, stub.ASSISTANT = 256, 259, 258, 260
        with patch.dict(sys.modules, {"ntsa_x.unified_cognition": stub}):
            result = evaluate_checkpoint_arms({
                "original": FakeNativeModel("no"),
                "control": FakeNativeModel("no"),
                "assimilated": FakeNativeModel("29"),
            }, [HeldoutCase("compute ruleone 9", "29", "sealed")], repeats=2)
        self.assertEqual(result["arms"]["original"]["exact_match_accuracy"], 0.0)
        self.assertEqual(result["arms"]["control"]["exact_match_accuracy"], 0.0)
        self.assertEqual(result["arms"]["assimilated"]["exact_match_accuracy"], 1.0)
        self.assertEqual(result["paired_discordance"]["assimilation_only_correct"], 1)
        self.assertIsNone(result["eim"])
        self.assertFalse(result["general_reasoning_improvement_proven"])

    def test_evaluator_rejects_duplicate_heldout(self):
        stub = ModuleType("ntsa_x.unified_cognition")
        stub.UnifiedCognitiveMechanism = FakeMechanism
        stub.BOS, stub.USER, stub.EOS, stub.ASSISTANT = 256, 259, 258, 260
        rows = [HeldoutCase("compute ruleone 9", "29", "sealed")]*2
        with patch.dict(sys.modules, {"ntsa_x.unified_cognition": stub}):
            with self.assertRaises(ValueError):
                evaluate_checkpoint_arms({"original": FakeNativeModel("29"),
                                          "control": FakeNativeModel("29"),
                                          "assimilated": FakeNativeModel("29")}, rows)


if __name__ == "__main__":
    unittest.main()
