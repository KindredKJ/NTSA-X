"""Regression and real-model tests for the Krystal single-cycle path.

Uses a real, randomly initialized PyTorch CogFuModel. Passing these structural
tests does NOT establish language competence, factual correctness, or savings.
"""
from __future__ import annotations

import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipUnless(torch is not None, "PyTorch optional native-model tests")
class KrystalCycleTests(unittest.TestCase):
    def setUp(self):
        from ntsa_x.cog_fu import CogFuConfig, CogFuModel
        torch.manual_seed(7)
        self.model = CogFuModel(CogFuConfig(
            width=16, layers=1, max_context=512, max_answer=12,
            diffusion_layers=1, denoise_steps=1))
        self.model.eval()

    def test_real_heads_share_one_trajectory_without_extra_generation(self):
        from unittest.mock import patch
        from ntsa_x.krystal_cycle import KrystalSingleCycle
        from ntsa_x.native_model import ByteCodec

        prompt, _ = ByteCodec().conversation([
            {"role": "user", "content": "Hello"}
        ], generation=True, max_tokens=512)
        trace = []
        original = self.model.unified_draft(
            prompt, max_new_tokens=8, temperature=0)
        observed = self.model.unified_draft(
            prompt, max_new_tokens=8, temperature=0,
            phase_trace=trace)
        self.assertEqual(original, observed)
        self.assertGreaterEqual(len(trace), 1)
        for step in trace:
            self.assertIn("forward_token", step)
            self.assertIn("counter_token", step)
            self.assertIn("head_agreement", step)
            self.assertGreaterEqual(step["forward_confidence"], 0)
            self.assertLessEqual(step["forward_confidence"], 1)
        with patch.object(
            self.model.cortex, "forward",
            wraps=self.model.cortex.forward
        ) as forward:
            result = KrystalSingleCycle(self.model).run(
                "Hello", max_new_tokens=8)
        # One context encoding and one incremental recurrent update per
        # emitted byte: there is no second inference pass.
        self.assertEqual(forward.call_count, result.selected_bytes + 1)
        self.assertFalse(result.externally_verified)
        self.assertTrue(0 <= result.head_agreement_fraction <= 1)
        self.assertGreaterEqual(result.head_challenges, 0)
        self.assertEqual(result.cognition_graph.kind, "cognition")
        self.assertEqual(result.diffusion_graph.kind, "diffusion")

    def test_recall_is_real_read_only_and_context_bounded(self):
        from ntsa_x.krystal_cycle import KrystalSingleCycle
        from ntsa_x.memory_fabric import MemoryFabric
        with MemoryFabric(":memory:", writable=True) as store:
            ident = store.remember(
                "semantic", "The orchard has green apples",
                source="reviewed-user")
            count_before = store.summary()["counts"].copy()
            result = KrystalSingleCycle(
                self.model, memory=store).run(
                    "Where are the apples?", max_new_tokens=8)
            self.assertIn(ident, result.recalled_ids)
            self.assertEqual(result.memory_graph.nodes[1].key, ident)
            self.assertEqual(store.summary()["counts"], count_before)
            self.assertFalse(result.externally_verified)

    def test_invalid_request_rejected_without_model_execution(self):
        from ntsa_x.krystal_cycle import KrystalSingleCycle
        engine = KrystalSingleCycle(self.model)
        with self.assertRaises(ValueError):
            engine.run("", max_new_tokens=8)
        with self.assertRaises(ValueError):
            engine.run("text", max_new_tokens=0)
        with self.assertRaises(ValueError):
            engine.run("text", max_new_tokens=13)

if __name__ == "__main__":
    unittest.main()
