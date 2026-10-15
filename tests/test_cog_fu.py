"""COG-FU: joint gradient, discrete corruption, denoising and checkpoint gates.

PyTorch is an optional dependency of the original NTSA-X reference kernel.
"""
import importlib.util
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, skipUnless

HAS_TORCH = importlib.util.find_spec("torch") is not None


@skipUnless(HAS_TORCH, "COG-FU native tests require PyTorch")
class CogFuTests(TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)

    def setUp(self):
        import torch
        from ntsa_x.cog_fu import CogFuConfig, CogFuModel
        self.torch = torch
        self.config = CogFuConfig(
            width=16, layers=1, max_context=128, max_answer=24,
            diffusion_layers=1, denoise_steps=3)
        torch.manual_seed(1234)
        self.model = CogFuModel(self.config)
        self.messages = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4."}]

    def test_cognitive_diffusion_loss_flows_to_both_branches(self):
        result = self.model.joint_loss(self.messages, seed=7)
        self.assertTrue(self.torch.isfinite(result["loss"]))
        self.assertGreater(float(result["diffusion_loss"].item()), 0)
        result["loss"].backward()
        paths = [
            "cortex.cells.0.input_candidate.weight",
            "cortex.cells.0.router.weight",
            "context_projection.1.weight",
            "blocks.0.mix.weight",
            "diffusion_head.weight",
        ]
        params = dict(self.model.named_parameters())
        for name in paths:
            self.assertIsNotNone(params[name].grad, name)
            self.assertGreater(float(params[name].grad.abs().sum()), 0, name)

    def test_full_mask_cannot_see_target_answer(self):
        from ntsa_x.cog_fu import MASK_ID
        codec = self.model.codec
        prefix, _ = codec.conversation([self.messages[0]], generation=True)
        prompt = self.torch.tensor([prefix], dtype=self.torch.long)
        masked = self.torch.full((1, 3), MASK_ID, dtype=self.torch.long)
        time = self.torch.tensor([1.0])
        self.model.eval()
        with self.torch.inference_mode():
            output_one = self.model.denoise_logits(prompt, masked, time)
            output_two = self.model.denoise_logits(prompt, masked, time)
        self.torch.testing.assert_close(output_one, output_two)
        self.assertEqual(output_one.shape, (1, 3, 262))
        self.assertTrue(self.torch.isfinite(output_one).all())

    def test_deterministic_multiple_step_denoising(self):
        from ntsa_x.cog_fu import MASK_ID
        prefix, _ = self.model.codec.conversation(
            [self.messages[0]], generation=True)
        a = self.model.draft(prefix, max_new_tokens=9, temperature=0, steps=4)
        b = self.model.draft(prefix, max_new_tokens=9, temperature=0, steps=4)
        self.assertEqual(a, b)
        self.assertLessEqual(len(a), 9)
        self.assertTrue(all(0 <= item < 256 for item in a))
        self.assertNotIn(MASK_ID, a)

    def test_training_improves_repeated_joint_sample(self):
        torch = self.torch
        torch.manual_seed(7)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.01)
        before = float(self.model.joint_loss(
            self.messages, seed=99, noise_fraction=1.0)["loss"])
        for _ in range(20):
            optimizer.zero_grad(set_to_none=True)
            loss = self.model.joint_loss(
                self.messages, seed=99, noise_fraction=1.0)["loss"]
            loss.backward()
            optimizer.step()
        after = float(self.model.joint_loss(
            self.messages, seed=99, noise_fraction=1.0)["loss"])
        self.assertLess(after, before * 0.8)

    def test_training_checkpoint_resume_native_cortex(self):
        from ntsa_x.cog_fu_train import train, load_checkpoint
        from ntsa_x.native_model import NativeConfig, NativeOrbitModel
        torch = self.torch
        with TemporaryDirectory() as folder:
            root = Path(folder)
            native = NativeOrbitModel(NativeConfig(
                width=16, layers=1, max_context=128))
            old = root / "native-v1"
            old.mkdir()
            (old / "config.json").write_text(json.dumps({
                "width": 16, "layers": 1, "max_context": 128,
                "dropout": 0.0, "vocab_size": 262}))
            torch.save(native.state_dict(), old / "weights.pt")

            for file, offset in (("train.jsonl", 0), ("eval.jsonl", 40)):
                with (root / file).open("w", encoding="utf-8") as stream:
                    for i in range(3):
                        example = [{"role": "user",
                                    "content": f"repeat {offset+i}"},
                                   {"role": "assistant", "content": "yes"}]
                        stream.write(json.dumps({"messages": example}) + "\n")
            first = root / "cog-fu-v1"
            report = train(
                train_path=root / "train.jsonl", eval_path=root / "eval.jsonl",
                output=first, config=self.config, epochs=1,
                batch_size=2, device="cpu", initialize_cortex=old)
            self.assertEqual(report["from"], "native-cortex")
            self.assertTrue(report["trained_cortex_and_denoiser_jointly"])
            self.assertEqual(report["teacher_calls"], 0)
            self.assertTrue(math.isfinite(report["best_holdout_joint_loss"]))
            self.assertTrue((first / "weights.pt").exists())
            loaded = load_checkpoint(first)
            self.assertEqual(loaded.config, self.config)
            second = root / "cog-fu-v2"
            continued = train(
                train_path=root / "train.jsonl", eval_path=root / "eval.jsonl",
                output=second, config=self.config, epochs=1, batch_size=2,
                device="cpu", initialize_from=first,
                replay_path=root / "train.jsonl")
            self.assertEqual(continued["rehearsal_samples"], 3)
            self.assertLessEqual(
                continued["best_holdout_joint_loss"],
                continued["initial_holdout"]["loss"] + 1e-7)
            self.assertEqual(len(continued["previous_weights_sha256"]), 64)
            with self.assertRaises(FileExistsError):
                train(train_path=root / "train.jsonl",
                      eval_path=root / "eval.jsonl",
                      output=first, config=self.config, epochs=1,
                      device="cpu")

    def test_train_eval_contamination_is_rejected(self):
        from ntsa_x.cog_fu_train import train
        with TemporaryDirectory() as folder:
            root = Path(folder)
            item = json.dumps({"messages": self.messages}) + "\n"
            (root / "train.jsonl").write_text(item)
            (root / "eval.jsonl").write_text(item)
            with self.assertRaisesRegex(ValueError, "overlap"):
                train(train_path=root / "train.jsonl",
                      eval_path=root / "eval.jsonl",
                      output=root / "no-model", config=self.config,
                      epochs=1, device="cpu")

    def test_existing_local_sse_server_works(self):
        from contextlib import closing
        from http.client import HTTPConnection
        from http.server import ThreadingHTTPServer
        from threading import Thread
        from ntsa_x.native_server import handler_factory
        import socket

        with closing(socket.socket()) as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        server = ThreadingHTTPServer(
            ("127.0.0.1", port),
            handler_factory(self.model, "ntsa-cog-fu-test", port, 7))
        runner = Thread(target=server.serve_forever, daemon=True)
        runner.start()
        try:
            conn = HTTPConnection("127.0.0.1", port, timeout=5)
            conn.request("POST", "/v1/chat/completions",
                         body=json.dumps({
                             "model": "ntsa-cog-fu-test",
                             "stream": True,
                             "max_tokens": 7,
                             "temperature": 0,
                             "messages": [self.messages[0]]}),
                         headers={"Content-Type": "application/json"})
            response = conn.getresponse()
            self.assertEqual(response.status, 200)
            self.assertIn("data: [DONE]", response.read().decode())
            conn.close()
            conn = HTTPConnection("127.0.0.1", port, timeout=5)
            conn.request("GET", "/health")
            response = conn.getresponse()
            info = json.loads(response.read())
            self.assertEqual(info["architecture"], "ntsa-x-cog-fu-discrete-v1")
            self.assertFalse(info["teacher_loaded"])
            conn.close()
        finally:
            server.shutdown()
            server.server_close()
            runner.join(timeout=5)


if __name__ == "__main__":
    import unittest
    unittest.main()
