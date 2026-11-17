"""Neural skill learning tests; native path uses actual Auxiliary when present."""
from __future__ import annotations
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, skipUnless
import torch
from ntsa_x.neural_inferencing import (NeuralSkillConfig, NeuralSkillHead,
     train_skill_head, evidence_training_pairs)
from ntsa_x.inferencing_skills import SkillLedger


def pairs(seed=213,count=120,width=8):
    g=torch.Generator().manual_seed(seed)
    qs=[];ps=[];ms=[];ys=[]
    for _ in range(count):
        q=torch.randn(3,width,generator=g)
        for prior,meta,y in (
            (q+.10*torch.randn(3,width,generator=g),[1,1,1,1,.6],1.),
            (q+.03*torch.randn(3,width,generator=g),[0,1,1,1,.6],0.),
            (q+.03*torch.randn(3,width,generator=g),[1,0,0,1,.6],0.),
            (q+.03*torch.randn(3,width,generator=g),[1,1,1,.1,.1],0.),
            (torch.randn(3,width,generator=g),[1,1,1,1,.6],0.)):
            qs.append(q);ps.append(prior);ms.append(meta);ys.append(y)
    return {"current":torch.stack(qs),"prior":torch.stack(ps),
            "metadata":torch.tensor(ms,dtype=torch.float32),
            "label":torch.tensor(ys,dtype=torch.float32)}


class NeuralHeadTests(TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_train_gradient_and_reject_hard_negatives(self):
        torch.manual_seed(213)
        head=NeuralSkillHead(NeuralSkillConfig(width=8,hidden=48))
        original=head.network[1].weight.detach().clone()
        history=train_skill_head(head,pairs(213,180),epochs=20,seed=213)
        self.assertLess(history[-1]["bce"],history[0]["bce"])
        self.assertFalse(torch.equal(original,head.network[1].weight))
        heldout=pairs(9123,25)
        with torch.no_grad():
            score=head.score(heldout["current"],heldout["prior"],heldout["metadata"])
        for i in range(25):
            self.assertGreater(float(score[5*i]),.67)
            self.assertLess(float(score[5*i+1]),.67)
            self.assertLess(float(score[5*i+2]),.67)
            self.assertLess(float(score[5*i+3]),.67)

    def test_replay_exact_seed_and_state_dict(self):
        batch=pairs(701,50)
        def run():
            torch.manual_seed(71)
            head=NeuralSkillHead(NeuralSkillConfig(width=8,hidden=24))
            report=train_skill_head(head,batch,epochs=6,seed=71)
            return report,{k:v.detach().clone() for k,v in head.state_dict().items()}
        a,aw=run();b,bw=run()
        self.assertEqual(a,b)
        self.assertTrue(all(torch.equal(aw[k],bw[k]) for k in aw))

    def test_confounded_evidence_excluded_and_contextual_signs_preserved(self):
        ledger=SkillLedger()
        trace=ledger.add_trace(architecture="fixture-probe",state=[.1]*24,
            state_transition=[0.]*24,shape=(3,8),operation="rotate-right",
            conditions={"bits":3},action="001",outcome={"next":"100"},
            evidence={"attributable":True},provenance="fixture")
        def stage(pred, baseline):
            receipt=ledger.event("inference",{"prediction":pred,
                "seed":213,"architecture":"fixture-probe",
                "counterfactual_prediction":baseline,
                "counterfactual_seed":213,
                "counterfactual_protocol":"auxiliary-native-paired-v1",
                "counterfactual_model_architecture":"fixture-probe",
                "counterfactual_before_outcome":True,
                "used":[{"skill_id":trace.skill_id,"trace_ids":[trace.trace_id]}]})
            ledger.event("neural_training_staged",{
                "inference_id":receipt,"current_state":[.2]*24,
                "candidates":[{"trace_id":trace.trace_id,"prior_state":[.1]*24,
                    "metadata":[1.,1.,1.,1.,.5],"selected":True}],
                "labels_available":False})
            return receipt
        positive=stage("100","000")
        with self.assertRaises(ValueError):
            evidence_training_pairs(ledger)
        ledger.observe(positive,observed="100",baseline="000",evidence_source="test",outcome_verified=True)
        provisional=stage("100","100")
        ledger.observe(provisional,observed="100",baseline="100",evidence_source="test",outcome_verified=True)
        negative=stage("000","100")
        ledger.observe(negative,observed="100",baseline="100",evidence_source="test",outcome_verified=True)
        data=evidence_training_pairs(ledger)
        self.assertEqual(data["label"].tolist(),[1.,0.])
        ledger.close()

    def test_checkpoint_restores_predictions_bitwise(self):
        torch.manual_seed(81)
        one=NeuralSkillHead(NeuralSkillConfig(width=8,hidden=24))
        train_skill_head(one,pairs(99,35),epochs=5,seed=81)
        with TemporaryDirectory() as d:
            target=Path(d)/"weights.pt"
            torch.save(one.state_dict(),target)
            two=NeuralSkillHead(NeuralSkillConfig(width=8,hidden=24))
            two.load_state_dict(torch.load(target,map_location="cpu",weights_only=True),strict=True)
            batch=pairs(239,3)
            with torch.no_grad():
                x=one.score(batch["current"],batch["prior"],batch["metadata"])
                y=two.score(batch["current"],batch["prior"],batch["metadata"])
            self.assertTrue(torch.equal(x,y))

try:
    from ntsa_x.auxiliary import AuxiliaryModel,AuxiliaryConfig
    NATIVE=True
except ImportError:
    NATIVE=False

@skipUnless(NATIVE,"Real Auxiliary source required")
class RealNativePathTests(TestCase):
    def test_neural_head_is_registered_and_single_inference_path(self):
        from ntsa_x.neural_inferencing import NeuralRecallBeforeInference
        torch.set_num_threads(1);torch.manual_seed(17)
        model=AuxiliaryModel(AuxiliaryConfig(width=16,layers=1,
            max_context=128,max_answer=24,diffusion_layers=1,
            denoise_steps=2,pdc_max_cycles=2,pdc_min_cycles=1))
        before=set(model.state_dict())
        ledger=SkillLedger()
        runtime=NeuralRecallBeforeInference(model,ledger)
        self.assertTrue(before.issubset(set(model.state_dict())))
        self.assertTrue(any(k.startswith("inferencing_skill_head.")
                            for k in model.state_dict()))
        train_skill_head(runtime.head,pairs(213,50,16),epochs=10,seed=213)
        prefix,_=model.codec.conversation(
            [{"role":"user","content":"Rotate three bits"}],
            generation=True,max_tokens=model.config.max_context)
        _,receipt=runtime.infer(prefix,operation="rotate-right",
                                conditions={"bits":3},seed=213,max_new_tokens=4)
        self.assertEqual(receipt["architecture"],model.architecture)
        events=ledger.events()
        self.assertEqual(events[-2]["kind"],"inference")
        self.assertEqual(events[-1]["kind"],"neural_training_staged")
        ledger.close()
