"""M14 guards: compatibility, no exam leakage, strict new-output versioning."""
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from ntsa_x.academy import export
from ntsa_x.academy_bridge import plan, prepare


class AcademyBridgeTests(unittest.TestCase):
    def setUp(self):
        self.folder = TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.academy = self.root / "academy"
        export(self.academy)
        self.checkpoint = self.root / "checkpoint"
        self.checkpoint.mkdir()
        (self.checkpoint / "config.json").write_text(json.dumps({
            "architecture": "ntsa-x-cog-fu-discrete-v1",
            "config": {"max_context": 128, "max_answer": 48}
        }), encoding="utf-8")

    def test_context_increase_is_calculated_without_truncation(self):
        report = plan(self.academy, self.checkpoint)
        self.assertEqual(report["training_samples"], 12)
        self.assertEqual(report["development_samples"], 12)
        self.assertEqual(report["reserved_exam_examples_used"], 0)
        self.assertGreater(report["required_full_sequence_bytes"], 128)
        self.assertLessEqual(report["required_full_sequence_bytes"],
                             report["suggested_training_context"])
        self.assertEqual(report["suggested_training_context"] % 64, 0)
        self.assertLessEqual(report["longest_answer_bytes_including_eos"], 48)

    def test_prepare_is_separate_disjoint_and_never_overwrites(self):
        out = self.root / "m14"
        receipt = prepare(self.academy, self.checkpoint, out)
        train_rows = [json.loads(x) for x in
                      (out / "train.jsonl").read_text().splitlines()]
        dev_rows = [json.loads(x) for x in
                    (out / "dev.jsonl").read_text().splitlines()]
        self.assertEqual((len(train_rows), len(dev_rows)), (12, 12))
        self.assertTrue(set(r["messages"][0]["content"] for r in train_rows)
                          .isdisjoint(set(r["messages"][0]["content"]
                                          for r in dev_rows)))
        self.assertEqual(len(receipt["prepared_sha256"]), 3)
        self.assertNotIn("exam", " ".join(p.name for p in out.iterdir()))
        with self.assertRaises(FileExistsError):
            prepare(self.academy, self.checkpoint, out)

    def test_reject_export_tampering_and_excess_answer(self):
        path = self.academy / "train-chat.jsonl"
        raw = path.read_text()
        path.write_text(raw.replace("A noun names", "A noun REALLY names", 1))
        with self.assertRaisesRegex(ValueError, "differs"):
            plan(self.academy, self.checkpoint)
        path.write_text(raw)
        config_path = self.checkpoint / "config.json"
        config = json.loads(config_path.read_text())
        config["config"]["max_answer"] = 8
        config_path.write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, "bounds"):
            plan(self.academy, self.checkpoint)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch optional")
    def test_versioned_context_expansion_preserves_source_weights(self):
        import hashlib
        import torch
        from ntsa_x.cog_fu import CogFuConfig, CogFuModel, COG_FU_ARCH
        from ntsa_x.cog_fu_train import train

        original = CogFuConfig(width=16, layers=1, max_context=32,
                               max_answer=16, diffusion_layers=1,
                               denoise_steps=1)
        larger = CogFuConfig(width=16, layers=1, max_context=64,
                             max_answer=16, diffusion_layers=1,
                             denoise_steps=1)
        model = CogFuModel(original)
        weights_path = self.checkpoint / "weights.pt"
        torch.save(model.state_dict(), weights_path)
        (self.checkpoint / "config.json").write_text(
            json.dumps({"architecture": COG_FU_ARCH,
                        "config": original.to_dict()}))
        before = hashlib.sha256(weights_path.read_bytes()).hexdigest()
        for name, marker in [("train", "A"), ("eval", "B")]:
            message = {"messages": [
                {"role": "user", "content": marker * 35},
                {"role": "assistant", "content": "yes"}]}
            (self.root / (name + ".jsonl")).write_text(
                json.dumps(message) + "\n", encoding="utf-8")
        args = dict(train_path=self.root / "train.jsonl",
                    eval_path=self.root / "eval.jsonl",
                    output=self.root / "trained-v2", config=larger,
                    epochs=1, batch_size=1, device="cpu",
                    initialize_from=self.checkpoint)
        with self.assertRaisesRegex(ValueError, "configuration mismatch"):
            train(**args)
        result = train(**args, allow_context_expansion=True)
        self.assertEqual(result["context_expanded_from"], 32)
        self.assertEqual(result["trained_max_context"], 64)
        self.assertTrue((self.root / "trained-v2" / "weights.pt").exists())
        self.assertEqual(before,
                         hashlib.sha256(weights_path.read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
