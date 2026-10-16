"""Regression gates for COG-FU collapsed output diagnostic and decoder ablation."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, skipUnless
from unittest.mock import patch
import importlib.util
import json

from ntsa_x.cog_fu_diagnose import generation_metrics

HAS_TORCH = importlib.util.find_spec("torch") is not None


class GenerationMetricsTests(TestCase):
    def test_repeated_e_is_flagged_without_claiming_reasoning(self):
        report = generation_metrics("eeeeeeeeeeeeeeeeeeeerieeeeeeeee")
        self.assertTrue(report["repetition_collapse_warning"])
        self.assertEqual(report["dominant_character"], "e")
        self.assertGreater(report["dominant_fraction"], 0.65)
        self.assertGreater(report["longest_same_character_run"], 12)

    def test_balanced_short_text_is_not_collapse(self):
        report = generation_metrics("Hello, I can read Python source.")
        self.assertFalse(report["repetition_collapse_warning"])
        self.assertEqual(report["characters"], 32)

    def test_empty_text_warns(self):
        report = generation_metrics("")
        self.assertTrue(report["repetition_collapse_warning"])
        self.assertEqual(report["warning_reason"], "empty_response")


@skipUnless(HAS_TORCH, "Decoder comparison requires optional PyTorch")
class NativeDecoderCompareTests(TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)

    def test_decoder_selection_is_explicit_and_weight_preserving(self):
        import torch
        from ntsa_x.cog_fu import CogFuConfig, CogFuModel
        model = CogFuModel(CogFuConfig(
            width=16, layers=1, max_context=128,
            max_answer=16, diffusion_layers=1, denoise_steps=2))
        before = [v.clone() for v in model.state_dict().values()]
        with patch.object(model, "draft", return_value=[ord("e")] * 5):
            model.inference_decoder = "diffusion"
            self.assertEqual(bytes(model.generate(
                [256], max_new_tokens=5)), b"eeeee")
        with patch.object(model.cortex, "generate", return_value=iter(
                [ord("H"), ord("i"), ord("!")])) as native:
            model.inference_decoder = "cortex"
            self.assertEqual(bytes(model.generate(
                [256], max_new_tokens=5)), b"Hi!")
            native.assert_called_once()
        model.inference_decoder = "invalid"
        with self.assertRaisesRegex(ValueError, "decoder"):
            bytes(model.generate([256], max_new_tokens=5))
        for old, new in zip(before, model.state_dict().values()):
            torch.testing.assert_close(old, new)

    def test_checkpoint_diagnosis_uses_both_decoders_and_framings(self):
        import torch
        from dataclasses import asdict
        from ntsa_x.cog_fu import COG_FU_ARCH, CogFuConfig, CogFuModel
        from ntsa_x.cog_fu_diagnose import diagnose
        cfg = CogFuConfig(width=16, layers=1, max_context=128,
                          max_answer=16, diffusion_layers=1,
                          denoise_steps=2)
        with TemporaryDirectory() as directory:
            path = Path(directory)
            torch.manual_seed(4)
            original = CogFuModel(cfg)
            torch.save(original.state_dict(), path / "weights.pt")
            (path / "config.json").write_text(json.dumps({
                "architecture": COG_FU_ARCH,
                "config": asdict(cfg),
            }), encoding="utf-8")
            report = diagnose(checkpoint=path, prompt="Hello?",
                              max_new_tokens=5, context_limit=1024)
            self.assertEqual(len(report["results"]), 4)
            self.assertEqual(
                {(r["framing"], r["decoder"]) for r in report["results"]},
                {("chat", "cortex"), ("chat", "diffusion"),
                 ("text", "cortex"), ("text", "diffusion")})
            self.assertFalse(report["checkpoint_modified"])
            self.assertEqual(report["teacher_calls"], 0)
            self.assertEqual(report["training_max_context"], 128)
            self.assertEqual(report["serving_max_context"], 1024)
            for row in report["results"]:
                self.assertLessEqual(len(row["response"]), 5)
                self.assertIn("repetition_collapse_warning", row["metrics"])
            for key, old in original.state_dict().items():
                torch.testing.assert_close(
                    old, torch.load(path / "weights.pt",
                                    weights_only=True)[key])

    def test_diagnostic_rejects_excessive_outputs_and_invalid_prompt(self):
        import torch
        from dataclasses import asdict
        from ntsa_x.cog_fu import COG_FU_ARCH, CogFuConfig, CogFuModel
        from ntsa_x.cog_fu_diagnose import diagnose
        cfg = CogFuConfig(width=16, layers=1, max_context=128,
                          max_answer=16, diffusion_layers=1)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            torch.save(CogFuModel(cfg).state_dict(), root / "weights.pt")
            (root / "config.json").write_text(json.dumps({
                "architecture": COG_FU_ARCH, "config": asdict(cfg)}))
            with self.assertRaisesRegex(ValueError, "Prompt"):
                diagnose(checkpoint=root, prompt="")
            with self.assertRaisesRegex(ValueError, "max_answer"):
                diagnose(checkpoint=root, prompt="Hi", max_new_tokens=48)
