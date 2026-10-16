"""M15: one cognitive state drives causal decoding and masked-byte refinement."""
import importlib.util
import unittest

@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch optional")
class UnifiedNativeFusionTests(unittest.TestCase):
    def test_fused_path_validates_inputs_and_keeps_state_dictionary(self):
        import torch
        from ntsa_x.cog_fu import CogFuConfig, CogFuModel
        from ntsa_x.native_model import ByteCodec
        torch.manual_seed(15)
        model = CogFuModel(CogFuConfig(
            width=16, layers=1, max_context=64,
            max_answer=8, diffusion_layers=1, denoise_steps=1))
        source = {k: value.clone() for k, value in model.state_dict().items()}
        prefix = [ByteCodec.BOS, *b"hello"]
        for mix in (0.0, 0.5, 1.0):
            emitted = model.unified_draft(
                prefix, max_new_tokens=5,
                temperature=0.0, diffusion_mix=mix)
            self.assertLessEqual(len(emitted), 5)
            self.assertTrue(all(0 <= item < 256 for item in emitted))
        self.assertEqual(
            bytes(model.unified_draft(
                prefix, max_new_tokens=5, diffusion_mix=0.5)),
            bytes(model.unified_draft(
                prefix, max_new_tokens=5, diffusion_mix=0.5)))
        model.inference_decoder = "unified"
        self.assertTrue(isinstance(
            bytes(model.generate(prefix, max_new_tokens=5)),
            bytes))
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, source[key]))
        with self.assertRaises(ValueError):
            model.unified_draft(prefix, diffusion_mix=-0.1)
        with self.assertRaises(ValueError):
            model.unified_draft(prefix, max_new_tokens=9)
        with self.assertRaises(ValueError):
            model.unified_draft([], max_new_tokens=2)


if __name__ == "__main__":
    unittest.main()
