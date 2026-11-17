"""Actual NTSA-X/CogFu ABI checks when the authoritative repo is importable."""
import importlib.util
import unittest

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from plugcore import PlugCoreBank, PlugCoreTrainer, SkillSpec


@unittest.skipUnless(HAS_TORCH and importlib.util.find_spec("ntsa_x"),
                     "Real upstream integration requires NTSA-X and optional PyTorch")
class ActualNTSAIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_native_orbit_gradients_checkpoint_and_restore(self):
        from ntsa_x.native_model import NativeOrbitModel, NativeConfig, ByteCodec
        torch.manual_seed(271)
        model = NativeOrbitModel(NativeConfig(width=16, layers=1, max_context=64))
        bank = PlugCoreBank()
        bank.register(SkillSpec(id="auxiliary_skill", version="0.2.0", width=16,
                                rank=8, layer=0, scale=1))
        original = {key: value.clone() for key, value in model.state_dict().items()}
        ids, labels = ByteCodec().conversation([
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"}], max_tokens=64)
        tokens = torch.tensor([ids]); targets = torch.tensor([labels])
        before = model.loss(tokens, targets).item()
        bank.attach(model)
        self.assertAlmostEqual(model.loss(tokens, targets).item(), before, places=6)
        with PlugCoreTrainer(model, bank, lr=0.02) as trainer:
            for _ in range(20):
                trainer.step(lambda: model.loss(tokens, targets))
        after = model.loss(tokens, targets).item()
        self.assertLess(after, before)
        self.assertEqual(model.state_dict().keys(), original.keys())
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, original[key], atol=0, rtol=0)
        bank.detach()
        self.assertAlmostEqual(model.loss(tokens, targets).item(), before, places=6)
        model.load_state_dict(original, strict=True)

    def test_cog_fu_joint_loss_single_cortex_and_unmodified_decoder(self):
        from ntsa_x.cog_fu import CogFuModel, CogFuConfig
        from ntsa_x.native_model import ByteCodec
        torch.manual_seed(272)
        model = CogFuModel(CogFuConfig(width=16, layers=1,
                max_context=64, max_answer=8, diffusion_layers=1, denoise_steps=1))
        before_keys = tuple(model.state_dict().keys())
        bank = PlugCoreBank()
        bank.register(SkillSpec(id="cogfu_skill",version="0.2.0",width=16,rank=4,layer=0))
        messages = [{"role":"user","content":"a"}, {"role":"assistant","content":"b"}]
        bank.attach(model)
        with PlugCoreTrainer(model, bank, lr=0.005) as trainer:
            loss = trainer.step(lambda: model.joint_loss(messages, seed=15)["loss"])
        self.assertGreater(loss, 0)
        self.assertEqual(before_keys, tuple(model.state_dict().keys()))
        prefix,_=ByteCodec().conversation([messages[0]], generation=True, max_tokens=64)
        generated = model.unified_draft(prefix, max_new_tokens=3)
        self.assertTrue(all(0 <= b < 256 for b in generated))
        bank.detach()


if __name__ == "__main__":
    unittest.main()
