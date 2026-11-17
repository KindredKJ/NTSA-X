"""Counterfactual evidence integrity and adversarial calibration regression tests."""
import unittest
import torch
from ntsa_x.inferencing_skills import SkillLedger
from ntsa_x.neural_inferencing import NeuralSkillConfig, NeuralSkillHead, train_skill_head
from ntsa_x.neural_calibration import calibrate_and_activate


def samples(seed,count=60,width=8):
    g=torch.Generator().manual_seed(seed)
    q=[];p=[];m=[];y=[]
    for _ in range(count):
        current=torch.randn(3,width,generator=g)
        items=[
            (current+.11*torch.randn(3,width,generator=g),[1,1,1,1,.6],1),
            (current+.045*torch.randn(3,width,generator=g),[1,1,1,.12,.1],0),
            (current+.045*torch.randn(3,width,generator=g),[0,1,1,1,.6],0),
            (current+.045*torch.randn(3,width,generator=g),[1,0,0,1,.6],0),
            (current+.09*torch.randn(3,width,generator=g),[1,1,1,.18,.1],0),
            (torch.randn(3,width,generator=g),[1,1,1,1,.6],0)]
        for prior,meta,label in items:
            q.append(current);p.append(prior);m.append(meta);y.append(label)
    return {"current":torch.stack(q),"prior":torch.stack(p),
            "metadata":torch.tensor(m,dtype=torch.float32),
            "label":torch.tensor(y,dtype=torch.float32)}


class ReceiptTests(unittest.TestCase):
    def setUp(self):
        self.ledger=SkillLedger()
        self.trace=self.ledger.add_trace(
            architecture="probe",state=[.1]*24,state_transition=[0.]*24,
            shape=(3,8),operation="rotate-right",conditions={"bits":3},
            action="001",outcome={"next":"100"},
            evidence={"attributable":True},provenance="fixture")
    def tearDown(self): self.ledger.close()

    def test_arbitrary_baseline_does_not_reinforce(self):
        receipt=self.ledger.event("inference",{"prediction":"100","used":[
            {"skill_id":self.trace.skill_id,"trace_ids":[self.trace.trace_id]}]})
        result=self.ledger.observe(receipt,observed="100",baseline="000",
            evidence_source="caller",outcome_verified=True)
        self.assertEqual(result["attribution"],"provisional")
        self.assertEqual(self.ledger.posterior(self.trace.skill_id,self.trace.conditions)[1:],(0,0))

    def test_paired_evidence_requires_validated_outcome(self):
        def record():
            return self.ledger.event("inference",{
                "prediction":"100","seed":213,"architecture":"probe",
                "counterfactual_prediction":"000",
                "counterfactual_protocol":"auxiliary-native-paired-v1",
                "counterfactual_seed":213,
                "counterfactual_model_architecture":"probe",
                "counterfactual_before_outcome":True,
                "used":[{"skill_id":self.trace.skill_id,"trace_ids":[self.trace.trace_id]}]})
        token=record()
        with self.assertRaises(ValueError):
            self.ledger.observe(token,observed="100",baseline="111",
                evidence_source="fixture",outcome_verified=True)
        no_proof=self.ledger.observe(token,observed="100",baseline=None,
            evidence_source="fixture",outcome_verified=False)
        self.assertEqual(no_proof["attribution"],"provisional")
        proof=self.ledger.observe(record(),observed="100",baseline=None,
            evidence_source="fixture",outcome_verified=True)
        self.assertEqual(proof["attribution"],"positive")


class CalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)
    def head(self):
        torch.manual_seed(213)
        model=NeuralSkillHead(NeuralSkillConfig(width=8,hidden=48))
        train_skill_head(model,samples(213,120),epochs=20,batch_size=48,seed=213)
        return model

    def test_spoofed_near_match_refuses_certification(self):
        head=self.head()
        validation=samples(922,60)
        validation["metadata"][1::6]=torch.tensor([1.,1.,1.,1.,.6])
        report=calibrate_and_activate(head,validation,independently_labelled=True)
        self.assertFalse(report["approved"],report)
        self.assertFalse(head.activation_certified)

    def test_indistinguishable_opposing_labels_abstain(self):
        head=self.head()
        validation=samples(999,60)
        base={k:validation[k][::6] for k in ("current","prior","metadata")}
        collision={k:torch.cat((v,v)) for k,v in base.items()}
        collision["label"]=torch.cat((torch.ones(60),torch.zeros(60)))
        result=calibrate_and_activate(head,collision,independently_labelled=True)
        self.assertEqual(result["reason"],"indistinguishable_opposing_outcomes")
        self.assertEqual(result["feature_collisions"],60)
        self.assertFalse(head.activation_certified)

    def test_external_label_trust_is_required(self):
        head=self.head()
        result=calibrate_and_activate(head,samples(81,60),
                                      independently_labelled=False)
        self.assertFalse(result["approved"])
        self.assertFalse(head.activation_certified)


if __name__=="__main__":
    unittest.main()
