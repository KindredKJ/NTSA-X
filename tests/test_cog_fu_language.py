"""Milestone 10: licensed local corpus, native language loss, Unicode wire safety.

Research tests use synthetic original text; no external dataset downloads.
"""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, skipUnless
from hashlib import sha256
import importlib.util
import json

HAS_TORCH = importlib.util.find_spec("torch") is not None


@skipUnless(HAS_TORCH, "Native acquisition requires optional PyTorch")
class LanguageAcquisitionTests(TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)

    def setUp(self):
        from ntsa_x.cog_fu import CogFuConfig
        self.config = CogFuConfig(
            width=16, layers=1, max_context=64, max_answer=24,
            diffusion_layers=1, denoise_steps=2)

    def corpus(self, root, *, duplicate=False):
        root = Path(root)
        folder = root / "texts"
        folder.mkdir(exist_ok=True)
        texts = [
            "One small document with a research example. Text generation is experimental.",
            "A distinct training passage talks about blue glass and yellow sunlight.",
            "Held out documents describe another independent series of words."]
        if duplicate:
            texts[-1] = texts[0]
        rows = []
        for i, text in enumerate(texts):
            relative = f"texts/source{i}.txt"
            (root / relative).write_text(text, encoding="utf-8")
            rows.append({"path": relative, "license": "user-owned-original",
                         "rights_confirmed": True})
        path = root / "manifest.json"
        path.write_text(json.dumps({
            "version": 1, "documents": rows}), encoding="utf-8")
        return path

    def test_document_disjoint_utf8_windows_and_no_duplicate_holdout(self):
        from ntsa_x.cog_fu_language_data import load_corpus, build_windows
        with TemporaryDirectory() as folder:
            manifest = self.corpus(folder)
            train, held, report = load_corpus(manifest, self.config)
            self.assertEqual(report["document_count"], 3)
            self.assertEqual(report["holdout_documents"], 1)
            self.assertFalse(
                {w.document_sha256 for w in train} &
                {w.document_sha256 for w in held})
            for w in [*train, *held]:
                self.assertLessEqual(len(w.prefix)+len(w.continuation),
                                     self.config.max_context)
                self.assertLessEqual(len(w.continuation),
                                     self.config.max_answer)
            emoji = "🔬Café" * 12
            windows = build_windows(emoji, sha256(emoji.encode()).hexdigest(),
                                    self.config)
            self.assertGreater(len(windows), 1)
            joined = b"".join(bytes(x for x in w.continuation if x < 256)
                              for w in windows).decode("utf-8")
            self.assertEqual(joined, emoji)
            self.assertFalse(report["near_duplicate_documents_checked"])
        with TemporaryDirectory() as folder:
            manifest = self.corpus(folder, duplicate=True)
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                load_corpus(manifest, self.config)

    def test_disallows_unapproved_and_outside_files(self):
        from ntsa_x.cog_fu_language_data import load_corpus
        with TemporaryDirectory() as folder:
            manifest = self.corpus(folder)
            data = json.loads(manifest.read_text())
            data["documents"][1]["rights_confirmed"] = False
            manifest.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "explicit rights"):
                load_corpus(manifest, self.config)
            data["documents"][1]["rights_confirmed"] = True
            data["documents"][1]["path"] = "../outside.txt"
            manifest.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                load_corpus(manifest, self.config)

    def test_language_gradients_are_joint_and_targets_causal(self):
        import torch
        from ntsa_x.cog_fu import CogFuModel
        torch.manual_seed(2)
        model = CogFuModel(self.config)
        prefix = [256, ord("A"), ord("B")]
        answer = list("hello".encode())
        outputs = model.language_loss(prefix, answer, seed=7)
        outputs["loss"].backward()
        parameters = dict(model.named_parameters())
        for key in ("cortex.cells.0.router.weight",
                    "context_projection.1.weight",
                    "diffusion_head.weight"):
            grad = parameters[key].grad
            self.assertIsNotNone(grad)
            self.assertGreater(float(grad.abs().sum()), 0)
        with self.assertRaises(ValueError):
            model.language_loss(prefix, list(range(200)), seed=7)

    def test_utf8_safe_bytes_for_corrupt_diffusion_draft(self):
        from ntsa_x.cog_fu import CogFuModel
        from ntsa_x.native_model import utf8_stream
        model = CogFuModel(self.config)
        model.draft = lambda *a, **kw: [0xFF, 0xC3, 0x28, ord("A")]
        raw = bytes(model.generate([256], max_new_tokens=4))
        self.assertEqual(raw.decode("utf-8"), "??(A")
        self.assertNotIn("\ufffd", raw.decode("utf-8"))
        self.assertEqual("".join(utf8_stream(iter(raw))), "??(A")

    def test_pretrain_checkpoint_continuation_and_equal_step_baseline(self):
        from ntsa_x.cog_fu_pretrain import pretrain
        from ntsa_x.cog_fu_train import load_checkpoint
        from ntsa_x.native_train import load_checkpoint as load_baseline
        with TemporaryDirectory() as folder:
            root = Path(folder)
            manifest = self.corpus(root)
            hybrid = root / "hybrid"
            base = root / "recurrent"
            result = pretrain(
                manifest=manifest, output=hybrid, config=self.config,
                epochs=1, batch_size=2, device="cpu",
                baseline_output=base,
                max_train_windows=3, max_eval_windows=2)
            self.assertEqual(result["teacher_calls"], 0)
            self.assertEqual(result["transformer_layers"], 0)
            self.assertGreater(result["steps"], 0)
            self.assertTrue(result["baseline"]["trained"])
            self.assertTrue(result["baseline"]
                            ["same_training_windows_and_step_budget"])
            self.assertTrue((hybrid / "weights.pt").is_file())
            self.assertTrue((base / "weights.pt").is_file())
            self.assertEqual(load_checkpoint(hybrid).config, self.config)
            self.assertEqual(load_baseline(base).config.width, 16)
            next_dir = root / "hybrid-next"
            followup = pretrain(
                manifest=manifest, output=next_dir, config=self.config,
                initialize_from=hybrid, epochs=1,
                batch_size=2, device="cpu",
                max_train_windows=3, max_eval_windows=2)
            self.assertEqual(len(followup["previous_weights_sha256"]), 64)
            self.assertLessEqual(
                followup["holdout"]["best"]["joint_loss"],
                followup["holdout"]["initial"]["joint_loss"])
            with self.assertRaises(FileExistsError):
                pretrain(manifest=manifest, output=hybrid,
                         config=self.config, epochs=1, device="cpu")


if __name__ == "__main__":
    import unittest
    unittest.main()
