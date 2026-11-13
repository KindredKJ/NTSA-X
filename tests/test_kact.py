"""K-ACT interface, causality, gradient and early-exit regression tests."""
import unittest
import torch
from ntsa_x.native_model import NativeConfig, NativeOrbitModel, ByteCodec
from ntsa_x.kact import KACTConfig, KACTNativeOrbitModel, KACTState


class KACTTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(17)
        self.model=KACTNativeOrbitModel(KACTConfig(width=16,layers=2,max_context=64))
        self.tokens=torch.tensor([[257,259,65,66,258]],dtype=torch.long)

    def test_shapes_and_distinct_working_state(self):
        logits,state=self.model(self.tokens)
        self.assertEqual(tuple(logits.shape),(1,5,262))
        self.assertIsInstance(state,KACTState)
        for lane,working in zip(state.lanes,state.working):
            self.assertEqual(tuple(lane.shape),(1,3,16))
            self.assertEqual(tuple(working.shape),(1,16))
            self.assertTrue(torch.isfinite(lane).all())

    def test_causal_chunking_in_eval(self):
        self.model.eval()
        with torch.inference_mode():
            full,state_all=self.model(self.tokens)
            first,state=self.model(self.tokens[:,:3])
            last,state=self.model(self.tokens[:,3:],state)
        torch.testing.assert_close(torch.cat((first,last),dim=1),full,atol=1e-6,rtol=1e-6)
        for a,b in zip(state.lanes,state_all.lanes):
            torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-6)
        for a,b in zip(state.working,state_all.working):
            torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-6)

    def test_gradients_flow_to_coupling_memory_controller(self):
        self.model.train()
        codec=ByteCodec()
        tokens,targets=codec.conversation([{"role":"user","content":"a"},{"role":"assistant","content":"b"}])
        loss=self.model.loss(torch.tensor([tokens]),torch.tensor([targets]))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for key in ("cross_lanes.weight","memory_candidate.weight","continue_head.weight"):
            grad=dict(self.model.cells[0].named_parameters())[key].grad
            self.assertIsNotNone(grad,key)
            self.assertTrue(torch.isfinite(grad).all(),key)
            self.assertGreater(grad.abs().sum().item(),0,key)

    def test_migrates_native_weights_without_mutation(self):
        source=NativeOrbitModel(NativeConfig(width=16,layers=2,max_context=64))
        original={k:v.clone() for k,v in source.state_dict().items()}
        self.model.initialize_from_native(source)
        for cell,src in zip(self.model.cells,source.cells):
            for name,tensor in src.state_dict().items():
                torch.testing.assert_close(cell.orbit.state_dict()[name],tensor)
        for name,tensor in source.state_dict().items():
            torch.testing.assert_close(source.state_dict()[name],original[name])

    def test_batch_early_exit_and_work_counts(self):
        self.model.eval()
        for cell in self.model.cells:
            cell.continue_head.bias.data.fill_(-50)
        with torch.inference_mode():
            _,_,metrics=self.model.forward_with_metrics(self.tokens)
        self.assertEqual(metrics["dense_cell_calls"],self.tokens.shape[1]*2)
        self.assertEqual(metrics["per_sample_logical_updates"].item(),10)
        for cell in self.model.cells:
            cell.continue_head.bias.data.fill_(50)
        with torch.inference_mode():
            _,_,metrics=self.model.forward_with_metrics(self.tokens)
        self.assertEqual(metrics["dense_cell_calls"],self.tokens.shape[1]*2*3)
        self.assertEqual(metrics["per_sample_logical_updates"].item(),30)

    def test_invalid_state_and_config_rejected(self):
        with self.assertRaises(ValueError):
            KACTConfig(max_refinements=9)
        with self.assertRaises(ValueError):
            self.model(torch.tensor([[1.0,2.0]]))
        with self.assertRaises(ValueError):
            self.model(self.tokens,KACTState(tuple(),tuple()))
        with self.assertRaises(ValueError):
            self.model.initial_state(0)
        with self.assertRaises(ValueError):
            self.model.initialize_from_native(NativeOrbitModel(NativeConfig(width=32)))

    def test_training_evaluation_loss_not_inclusive(self):
        codec=ByteCodec()
        tokens,labels=codec.conversation([{"role":"user","content":"Hello"},{"role":"assistant","content":"World"}])
        tok=torch.tensor([tokens],dtype=torch.long)
        target=torch.tensor([labels],dtype=torch.long)
        orig=self.model.loss(tok,target)
        self.assertTrue(torch.isfinite(orig))
        self.model.eval()
        eval_loss=self.model.loss(tok,target)
        self.assertTrue(torch.isfinite(eval_loss))
        self.assertGreaterEqual(float(orig.item()),0)

if __name__=='__main__':
    unittest.main()
