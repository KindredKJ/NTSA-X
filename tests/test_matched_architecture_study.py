"""Controls for native-vs-Transformer experiment. Never interpreted as AGI scores."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

TORCH_PRESENT=importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_PRESENT,"PyTorch is optional in the standard runtime")
class ControlledArchitectureTests(unittest.TestCase):
    def test_causal_transformer_does_not_see_future_tokens(self):
        import torch
        from ntsa_x.matched_architecture_study import CausalByteTransformer
        torch.manual_seed(11)
        model=CausalByteTransformer(16,16).eval()
        a=torch.tensor([[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]])
        b=torch.tensor([[1,2,3,4,99,98,97,96,9,10,11,12,13,14,15,16]])
        with torch.inference_mode():
            logits_a=model(a)[:,:4]
            logits_b=model(b)[:,:4]
        self.assertTrue(torch.allclose(logits_a,logits_b,atol=1e-5))

    def test_native_reset_state_is_real_ablation(self):
        import torch
        from ntsa_x.native_model import NativeConfig,NativeOrbitModel
        from ntsa_x.matched_architecture_study import native_without_recurrent_memory
        torch.manual_seed(9)
        model=NativeOrbitModel(NativeConfig(width=16,layers=1,max_context=16)).eval()
        tokens=torch.tensor([[1,2,3,1,2,3]])
        with torch.inference_mode():
            recurrent=model(tokens)[0]
            no_memory=native_without_recurrent_memory(model,tokens)
        self.assertEqual(recurrent.shape,no_memory.shape)
        self.assertFalse(torch.allclose(recurrent[:,2:],no_memory[:,2:]))

    def test_full_experiment_uses_exact_public_training_split_not_evaluation(self):
        from ntsa_x.matched_architecture_study import run
        with tempfile.TemporaryDirectory() as td:
            directory=Path(td)
            for i in range(1000):
                item={"train":[{"input":[[i%10]],"output":[[(i+1)%10]]}],
                      "test":[{"input":[[(i+2)%10]],"output":[[(i+3)%10]]}]}
                (directory/f"{i:04d}.json").write_text(json.dumps(item))
            result=run(directory,seeds=(3,),steps=1,sequence_length=16,
                       batch_size=1,heldout_batches=1)
            self.assertEqual(result["protocol"]["training_files"],850)
            self.assertEqual(result["protocol"]["heldout_files"],150)
            self.assertTrue(result["token_exposure_matched"])
            self.assertFalse(result["model_comparison_is_exact_flop_matched"])
            self.assertIsNone(result["official_arc_puzzle_accuracy"])
            self.assertFalse(result["trained_agi_claim"])
            self.assertLessEqual(result["parameter_delta_percent"],10)


if __name__=="__main__":
    unittest.main()
