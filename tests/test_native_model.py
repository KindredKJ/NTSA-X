"""Native model proof tests. Skipped when optional PyTorch is not installed."""
import importlib.util
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, skipUnless

HAS_TORCH = importlib.util.find_spec("torch") is not None


@skipUnless(HAS_TORCH, "Native training tests require optional PyTorch")
class NativeOrbitTests(TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)

    def setUp(self):
        import torch
        from ntsa_x.native_model import NativeConfig, NativeOrbitModel, ByteCodec
        self.torch = torch
        torch.manual_seed(7)
        self.model = NativeOrbitModel(
            NativeConfig(width=16, layers=2, max_context=128))
        self.codec = ByteCodec()

    def test_role_framing_and_no_prompt_supervision(self):
        ids, targets = self.codec.conversation([
            {"role": "user", "content": "what?"},
            {"role": "assistant", "content": "✓"}], max_tokens=128)
        self.assertEqual(ids[0], self.codec.BOS)
        self.assertEqual(ids[-1], self.codec.EOS)
        self.assertEqual(targets[-1], self.codec.EOS)
        self.assertEqual(targets.count(-100), len(ids)-len("✓".encode())-1)
        self.assertEqual(self.codec.conversation(
            [{"role":"user","content":"hi"}],generation=True)[0][-1],
                         self.codec.ASSISTANT)
        with self.assertRaises(ValueError):
            self.codec.conversation([
                {"role":"assistant","content":"no"}],generation=True)

    def test_recurrent_parameters_receive_nonzero_gradients(self):
        torch=self.torch
        ids,gold=self.codec.conversation([
            {"role":"user","content":"zero"},
            {"role":"assistant","content":"one"}],max_tokens=128)
        loss=self.model.loss(torch.tensor([ids]),torch.tensor([gold]))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for name in ("input_candidate.weight","state_candidate.weight",
                     "input_gate.weight","router.weight"):
            param=dict(self.model.cells[0].named_parameters())[name]
            self.assertIsNotNone(param.grad)
            self.assertGreater(float(param.grad.abs().sum().item()),0.0,name)

    def test_chunked_recurrence_matches_single_pass(self):
        torch=self.torch
        self.model.eval()
        ids,_=self.codec.conversation([
            {"role":"user","content":"abc"},
            {"role":"assistant","content":"def"}],max_tokens=128)
        tokens=torch.tensor([ids],dtype=torch.long)
        with torch.inference_mode():
            reference, full_state=self.model(tokens)
            first, state=self.model(tokens[:,:4])
            second, state=self.model(tokens[:,4:],state)
        torch.testing.assert_close(torch.cat([first,second],dim=1),reference,
                                   atol=1e-6,rtol=1e-6)
        for observed,expected in zip(state,full_state):
            torch.testing.assert_close(observed,expected,atol=1e-6,rtol=1e-6)

    def test_rejects_invalid_shape_and_context(self):
        torch=self.torch
        with self.assertRaises(ValueError):
            self.model(torch.tensor([1,2,3]))
        with self.assertRaises(ValueError):
            self.model(torch.zeros((1,129),dtype=torch.long))
        with self.assertRaises(ValueError):
            self.model.loss(torch.tensor([[1,2]]),torch.tensor([[-100,-100]]))

    def test_incremental_generation_and_utf8_boundary(self):
        from ntsa_x.native_model import utf8_stream
        encoded=list("🔬".encode("utf-8"))
        self.assertEqual("".join(utf8_stream(iter(encoded))),"🔬")
        prefix,_=self.codec.conversation([
            {"role":"user","content":"hello"}],generation=True,max_tokens=128)
        emitted=list(self.model.generate(prefix,max_new_tokens=8))
        self.assertTrue(all(0<=x<=255 for x in emitted))
        self.assertLessEqual(len(emitted),8)

    def test_training_checkpoint_roundtrip_and_holdout(self):
        from ntsa_x.native_train import train,load_checkpoint
        torch=self.torch
        with TemporaryDirectory() as folder:
            root=Path(folder)
            for fname,offset in (("train.jsonl",0),("eval.jsonl",20)):
                with (root/fname).open("w",encoding="utf-8") as stream:
                    for i in range(4):
                        messages=[{"role":"user","content":f"say {i+offset}"},
                                  {"role":"assistant","content":f"value {i+offset}"}]
                        stream.write(json.dumps({"messages":messages})+"\n")
            destination=root/"native-weights"
            summary=train(train_path=root/"train.jsonl",
                          eval_path=root/"eval.jsonl",output=destination,
                          width=16,layers=1,max_context=128,
                          epochs=1,batch_size=2,device="cpu")
            self.assertEqual(summary["external_teacher_calls"],0)
            self.assertEqual(summary["transformer_layers"],0)
            self.assertEqual(summary["train_examples"],4)
            self.assertTrue(math.isfinite(summary["best_heldout_loss"]))
            loaded=load_checkpoint(destination)
            self.assertEqual(loaded.config.width,16)
            prompt,_=self.codec.conversation([
                {"role":"user","content":"new task"}],generation=True,max_tokens=128)
            with torch.no_grad():
                logits,states=loaded(torch.tensor([prompt]))
            self.assertEqual(logits.shape[-1],262)
            self.assertEqual(states[0].shape,(1,3,16))
            with self.assertRaises(FileExistsError):
                train(train_path=root/"train.jsonl",eval_path=root/"eval.jsonl",
                      output=destination,width=16,layers=1,max_context=128,
                      epochs=1,device="cpu")

    def test_prompt_overlap_blocks_leakage(self):
        from ntsa_x.native_train import train
        with TemporaryDirectory() as folder:
            root=Path(folder)
            record={"messages":[{"role":"user","content":"same prompt"},
                                {"role":"assistant","content":"answer"}]}
            for name in ("train.jsonl","eval.jsonl"):
                (root/name).write_text(json.dumps(record)+"\n",encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"share exact user prompts"):
                train(train_path=root/"train.jsonl",eval_path=root/"eval.jsonl",
                      output=root/"output",width=16,layers=1,max_context=128,
                      epochs=1,device="cpu")


if __name__=="__main__":
    import unittest
    unittest.main()
