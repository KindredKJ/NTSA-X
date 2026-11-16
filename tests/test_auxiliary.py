"""Structural, training-path, checkpoint, and task-benchmark tests for Auxiliary."""
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from unittest import TestCase, skipUnless

HAS_TORCH = importlib.util.find_spec("torch") is not None


@skipUnless(HAS_TORCH, "Auxiliary PDC tests require PyTorch")
class AuxiliaryTests(TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)

    def setUp(self):
        import torch
        from ntsa_x.auxiliary import AuxiliaryConfig, AuxiliaryModel
        self.torch = torch
        torch.manual_seed(114)
        self.config = AuxiliaryConfig(
            width=16, layers=1, max_context=128, max_answer=24,
            diffusion_layers=1, denoise_steps=2, pdc_max_cycles=4,
            pdc_min_cycles=2, pdc_tolerance=0.01)
        self.model = AuxiliaryModel(self.config)
        self.messages = [
            {"role": "user", "content": "Give the exact result of 2+2."},
            {"role": "assistant", "content": "4"},
        ]

    def test_config_rejects_unbounded_or_invalid_cycle_budgets(self):
        from ntsa_x.auxiliary import AuxiliaryConfig
        with self.assertRaises(ValueError):
            AuxiliaryConfig(width=16, pdc_max_cycles=65)
        with self.assertRaises(ValueError):
            AuxiliaryConfig(width=16, pdc_max_cycles=3, pdc_min_cycles=4)

    def test_projection_diffusion_is_deterministic_bounded_and_finite(self):
        prefix, _ = self.model.codec.conversation(
            [self.messages[0]], generation=True, max_tokens=self.config.max_context)
        first, receipt_a = self.model.resolve_projections(
            prefix, max_cycles=4, adaptive=False, seed=55)
        second, receipt_b = self.model.resolve_projections(
            prefix, max_cycles=4, adaptive=False, seed=55)
        self.torch.testing.assert_close(first, second)
        self.assertEqual(first.shape, (1, 3, self.config.width))
        self.assertTrue(self.torch.isfinite(first).all())
        self.assertEqual(receipt_a, receipt_b)
        self.assertEqual(receipt_a.cycles, 4)

    def test_adaptive_diffusion_stays_within_configured_budget(self):
        from ntsa_x.auxiliary import AuxiliaryConfig, AuxiliaryModel
        self.model.config = AuxiliaryConfig(
            **{**self.config.to_dict(), "pdc_tolerance": 1.0})
        prefix, _ = self.model.codec.conversation(
            [self.messages[0]], generation=True, max_tokens=self.config.max_context)
        _, receipt = self.model.resolve_projections(
            prefix, max_cycles=4, adaptive=True, seed=9)
        self.assertGreaterEqual(receipt.cycles, self.model.config.pdc_min_cycles)
        self.assertLessEqual(receipt.cycles, self.model.config.pdc_max_cycles)

    def test_joint_training_reaches_pdc_cortex_and_response_branches(self):
        result = self.model.joint_loss(self.messages, seed=12)
        self.assertTrue(self.torch.isfinite(result["loss"]))
        self.assertTrue(self.torch.isfinite(result["pdc_loss"]))
        result["loss"].backward()
        parameters = dict(self.model.named_parameters())
        for name in (
            "cortex.cells.0.input_candidate.weight",
            "projection_denoiser.network.0.weight",
            "projection_readout.1.weight",
            "pdc_context_gate",
            "diffusion_head.weight",
        ):
            self.assertIsNotNone(parameters[name].grad, name)
            self.assertGreater(float(parameters[name].grad.abs().sum()), 0, name)

    def test_draft_resolves_projection_then_emits_only_byte_values(self):
        prefix, _ = self.model.codec.conversation(
            [self.messages[0]], generation=True, max_tokens=self.config.max_context)
        output = self.model.draft(
            prefix, max_new_tokens=8, temperature=0.0, pdc_cycles=3, seed=4)
        self.assertLessEqual(len(output), 8)
        self.assertTrue(all(0 <= item < 256 for item in output))
        self.assertEqual(self.model.last_projection_receipt.cycles, 3)

    def test_overlapped_projection_refinement_is_bounded_and_repeatable(self):
        prefix, _ = self.model.codec.conversation(
            [self.messages[0]], generation=True,
            max_tokens=self.config.max_context)
        first, receipt_a = self.model.resolve_projections_overlapped(
            prefix, max_cycles=4, seed=41)
        second, receipt_b = self.model.resolve_projections_overlapped(
            prefix, max_cycles=4, seed=41)
        self.torch.testing.assert_close(first, second)
        self.assertTrue(self.torch.isfinite(first).all())
        self.assertEqual(receipt_a, receipt_b)
        self.assertEqual(receipt_a.cycles, 4)

    def test_overlapped_generation_keeps_direct_scanner_and_fixed_budget(self):
        prefix, _ = self.model.codec.conversation(
            [self.messages[0]], generation=True,
            max_tokens=self.config.max_context)
        with patch.object(self.model.cortex, "forward",
                          wraps=self.model.cortex.forward) as legacy_forward:
            output = self.model.draft(
                prefix, max_new_tokens=8, temperature=0.0,
                pdc_cycles=4, adaptive_pdc=False, seed=73,
                pdc_schedule="overlap")
        legacy_forward.assert_not_called()
        self.assertTrue(all(0 <= item < 256 for item in output))
        self.assertEqual(self.model.last_projection_receipt.cycles, 4)
        with self.assertRaises(ValueError):
            self.model.draft(
                prefix, max_new_tokens=8, pdc_cycles=4,
                adaptive_pdc=True, pdc_schedule="overlap")

    def test_direct_scanner_reads_input_without_legacy_encoder_forward(self):
        prefix, _ = self.model.codec.conversation(
            [self.messages[0]], generation=True,
            max_tokens=self.config.max_context)
        tokens = self.torch.tensor([prefix], dtype=self.torch.long)
        projections, signal = self.model.scan_experience(tokens)
        self.assertEqual(projections.shape, (1, 3, self.config.width))
        self.assertEqual(signal.shape, (1, self.config.width))
        self.assertTrue(self.torch.isfinite(projections).all())
        self.assertTrue(self.torch.isfinite(signal).all())

        with patch.object(self.model.cortex, "forward",
                          wraps=self.model.cortex.forward) as legacy_forward:
            output = self.model.draft(
                prefix, max_new_tokens=8, temperature=0.0,
                pdc_cycles=2, seed=31)
        legacy_forward.assert_not_called()
        self.assertTrue(all(0 <= item < 256 for item in output))

    def test_auxiliary_checkpoint_round_trip(self):
        from ntsa_x.auxiliary import load_checkpoint
        with TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "config.json").write_text(json.dumps({
                "architecture": self.model.architecture,
                "config": self.config.to_dict()}), encoding="utf-8")
            self.torch.save(self.model.state_dict(), root / "weights.pt")
            restored = load_checkpoint(root)
            self.assertEqual(restored.config, self.config)
            self.assertEqual(restored.state_dict().keys(), self.model.state_dict().keys())

    def test_general_skill_benchmark_reports_category_and_cost(self):
        from examples.benchmark_auxiliary import benchmark
        case = {"category": "general_skill", "messages": [self.messages[0]],
                "expected": "unused exact-match baseline"}
        result = benchmark(self.model, [case], max_new_tokens=8,
                           pdc_cycles=2, adaptive=False, seed=2)
        self.assertEqual(result["cases"], 1)
        self.assertIn("general_skill", result["categories"])
        metrics = result["categories"]["general_skill"]
        self.assertEqual(metrics["cases"], 1)
        self.assertEqual(metrics["mean_pdc_cycles"], 2)
        self.assertGreaterEqual(metrics["latency_p50_seconds"], 0)
        self.assertIn("no live tools", result["execution_note"])

    def test_schedule_comparison_reports_quality_and_latency_together(self):
        from examples.benchmark_auxiliary import compare_schedules
        case = {"category": "general_skill", "messages": [self.messages[0]],
                "expected": "unused exact-match baseline"}
        result = compare_schedules(
            self.model, [case], max_new_tokens=8, pdc_cycles=3, seed=8)
        self.assertEqual(result["serial"]["schedule"], "serial")
        self.assertEqual(result["overlap"]["schedule"], "overlap")
        category = result["deltas"]["general_skill"]
        self.assertIn("p50_latency_speedup", category)
        self.assertIn("exact_match_delta", category)

    def test_auxiliary_integrates_with_memory_and_registered_execution(self):
        from ntsa_x.unified_cognition import (
            UnifiedCognitiveMechanism, scalar_extension,
            BOS, USER, EOS, ASSISTANT)
        runtime = UnifiedCognitiveMechanism(
            self.model, extensions=(scalar_extension(),))
        request = "scalar increment 7 by 2"
        prompt = [BOS, USER, *request.encode("utf-8"), EOS, ASSISTANT]
        bytes(list(runtime.generate(prompt, max_new_tokens=4)))
        self.assertEqual(runtime.last_receipt["architecture"], self.model.architecture)
        self.assertEqual(runtime.last_receipt["native_extensions"], ["scalar.native"])
        self.assertIsNotNone(self.model.last_projection_receipt)
        self.assertTrue(runtime.last_receipt["background_levels"]
                        ["L0_unified_decode_denoise"] == "executed")

    def test_training_entrypoint_stages_and_loads_auxiliary_checkpoint(self):
        from ntsa_x.auxiliary import AuxiliaryConfig, load_checkpoint
        from ntsa_x.auxiliary_train import train
        with TemporaryDirectory() as folder:
            root = Path(folder)
            training = [{"messages": [
                {"role": "user", "content": "Add 2 and 3."},
                {"role": "assistant", "content": "5"}]}]
            heldout = [{"messages": [
                {"role": "user", "content": "Add 4 and 1."},
                {"role": "assistant", "content": "5"}]}]
            (root / "train.jsonl").write_text(
                "\n".join(json.dumps(row) for row in training), encoding="utf-8")
            (root / "eval.jsonl").write_text(
                "\n".join(json.dumps(row) for row in heldout), encoding="utf-8")
            config = AuxiliaryConfig(
                width=16, layers=1, max_context=128, max_answer=24,
                diffusion_layers=1, denoise_steps=2, pdc_max_cycles=2,
                pdc_min_cycles=1)
            report = train(
                train_path=root / "train.jsonl", eval_path=root / "eval.jsonl",
                output=root / "aux-v1", config=config, epochs=1,
                batch_size=1, device="cpu", seed=7)
            self.assertEqual(report["architecture"], self.model.architecture)
            self.assertTrue((root / "aux-v1" / "weights.pt").is_file())
            restored = load_checkpoint(root / "aux-v1")
            self.assertEqual(restored.config, config)


if __name__ == "__main__":
    import unittest
    unittest.main()
