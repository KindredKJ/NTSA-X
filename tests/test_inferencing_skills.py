"""Regression suite for Auxiliary's evidence-ledger and native-model guidance."""
from __future__ import annotations
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, skipUnless

from ntsa_x.inferencing_skills import SkillLedger, RecallBeforeInference


def vec(value: int) -> list[float]:
    # Independent synthetic feature fixture, never an Auxiliary checkpoint.
    return [float((value >> (i % 3)) & 1) + .02 for i in range(12)]


def insert(ledger, operation, prior, result, *, conditions=None, evidence=None):
    return ledger.add_trace(
        architecture="fixture-probe", state=vec(prior),
        state_transition=[.1]*12, shape=(3, 4),
        operation=operation, conditions=conditions or {"bits": 3},
        action=f"{prior:03b}", outcome={"next": f"{result:03b}"},
        evidence=evidence or {"attributable": True, "source": "fixture"},
        provenance=f"fixture:{operation}:{prior}:{result}")


class FixtureModel:
    architecture = "fixture-probe"


def assess(ledger, value, operation, conditions):
    evaluator = object.__new__(RecallBeforeInference)
    evaluator.ledger, evaluator.model = ledger, FixtureModel()
    evaluator.top_k, evaluator.fit_threshold = 16, .60
    return evaluator.assess(vec(value), operation, conditions)


class EvidenceLedgerTests(TestCase):
    def test_bit_rotation_and_mismatch(self):
        ledger = SkillLedger()
        a = insert(ledger, "rotate-right", 0b100, 0b010)
        b = insert(ledger, "rotate-right", 0b010, 0b001)
        self.assertEqual(((0b001 >> 1) | ((0b001 & 1) << 2)), 0b100)
        self.assertTrue(any(r["eligible"] for r in assess(ledger, 0b001, "rotate-right", {"bits":3})))
        mismatch = assess(ledger, 0b001, "clear-lowest-set-bit", {"bits":3})
        self.assertEqual(0b001 & (0b001 - 1), 0b000)
        self.assertFalse(any(r["eligible"] for r in mismatch))
        self.assertTrue(all("operation_mismatch" in r["reasons"] for r in mismatch))
        self.assertEqual(ledger.posterior(a.skill_id,a.conditions)[1:],(0,0))
        ledger.close()

    def test_changed_conditions_and_unverified_attribution_reject(self):
        ledger = SkillLedger()
        insert(ledger,"rotate-right",4,2,conditions={"bits":3,"mode":"rotate"})
        insert(ledger,"rotate-right",4,2,evidence={"attributable":None})
        mismatch = assess(ledger,4,"rotate-right",{"bits":3,"mode":"clear"})
        self.assertFalse(any(r["eligible"] for r in mismatch))
        self.assertTrue(any("conflict:mode" in r["reasons"] for r in mismatch))
        ledger.close()

    def test_conflicting_same_starting_state(self):
        ledger = SkillLedger()
        insert(ledger,"rotate-right",4,2)
        insert(ledger,"rotate-right",4,7)
        cases=assess(ledger,4,"rotate-right",{"bits":3})
        self.assertFalse(any(r["eligible"] for r in cases))
        self.assertTrue(any("contradictory_evidence" in r["reasons"] for r in cases))
        ledger.close()

    def test_delayed_recall(self):
        ledger=SkillLedger()
        trace=insert(ledger,"rotate-right",1,4)
        for i in range(180):
            ledger.event("intervening_inference",{"ordinal":i})
        self.assertTrue(any(r["trace_id"]==trace.trace_id and r["eligible"]
                            for r in assess(ledger,1,"rotate-right",{"bits":3})))
        ledger.close()

    def test_attribution_and_snapshot_restoration(self):
        with TemporaryDirectory() as root:
            ledger=SkillLedger(Path(root)/"state.db")
            trace=insert(ledger,"rotate-right",1,4)
            def record(prediction, baseline=None):
                return ledger.event("inference",{"prediction":prediction,
                    "seed":213,"architecture":"fixture-probe",
                    "counterfactual_prediction":baseline,
                    "counterfactual_seed":213 if baseline is not None else None,
                    "counterfactual_protocol":"auxiliary-native-paired-v1" if baseline is not None else None,
                    "counterfactual_model_architecture":"fixture-probe" if baseline is not None else None,
                    "counterfactual_before_outcome":baseline is not None,
                    "used":[{"skill_id":trace.skill_id,"trace_ids":[trace.trace_id],"fit":.75}]})
            a=ledger.observe(record("100","000"),observed="100",baseline="000",evidence_source="independent",outcome_verified=True)
            self.assertEqual(a["attribution"],"positive")
            b=ledger.observe(record("100","100"),observed="100",baseline="100",evidence_source="independent",outcome_verified=True)
            self.assertEqual(b["attribution"],"provisional")
            c=ledger.observe(record("000","100"),observed="100",baseline="100",evidence_source="independent",outcome_verified=True)
            self.assertEqual(c["attribution"],"negative")
            d=ledger.observe(record("100"),observed="100",baseline=None,evidence_source="unknown")
            self.assertEqual(d["attribution"],"provisional")
            self.assertEqual(ledger.posterior(trace.skill_id,trace.conditions)[1:],(1,1))
            original=ledger.events()
            ledger.snapshot(Path(root)/"copy.db")
            ledger.close()
            restored=SkillLedger(Path(root)/"copy.db")
            self.assertEqual(original,restored.events())
            self.assertEqual(trace.trace_id,restored.get_traces()[0].trace_id)
            self.assertEqual(restored.posterior(trace.skill_id,trace.conditions)[1:],(1,1))
            restored.close()

    def test_deterministic_replay(self):
        def replay():
            ledger=SkillLedger()
            insert(ledger,"rotate-right",4,2)
            insert(ledger,"rotate-right",2,1)
            ledger.event("inspection",{"seed":213,
                "candidates":assess(ledger,1,"rotate-right",{"bits":3})})
            serialized=json.dumps(ledger.events(),sort_keys=True)
            ledger.close()
            return serialized
        self.assertEqual(replay(),replay())


try:
    import torch
    from ntsa_x.auxiliary import AuxiliaryConfig, AuxiliaryModel
    HAS_AUXILIARY=True
except ImportError:
    HAS_AUXILIARY=False


@skipUnless(HAS_AUXILIARY,"Requires PyTorch and authoritative Auxiliary source")
class NativePathTests(TestCase):
    def test_shared_projection_inference_and_structured_order(self):
        torch.set_num_threads(1)
        torch.manual_seed(21)
        model=AuxiliaryModel(AuxiliaryConfig(
            width=16,layers=1,max_context=128,max_answer=24,
            diffusion_layers=1,denoise_steps=2,pdc_max_cycles=2,pdc_min_cycles=1))
        prefix,_=model.codec.conversation(
            [{"role":"user","content":"Rotate a 3-bit register"}],
            generation=True,max_tokens=model.config.max_context)
        with torch.inference_mode():
            projection,_=model.resolve_projections(prefix,seed=213)
        state=projection.flatten().tolist()
        ledger=SkillLedger()
        trace=ledger.add_trace(architecture=model.architecture,
             state=state,state_transition=[.05]*len(state),shape=(3,16),
             operation="rotate-right",conditions={"bits":3},
             action="100",outcome={"next":"010"},
             evidence={"attributable":True,"source":"fixture"},
             provenance="fixture:known-outcome")
        engine=RecallBeforeInference(model,ledger,fit_threshold=.65)
        output,receipt=engine.infer(prefix,operation="rotate-right",
                conditions={"bits":3},seed=213,max_new_tokens=4)
        self.assertIsInstance(output,list)
        self.assertEqual(len(receipt["used"]),1)
        self.assertEqual(receipt["used"][0]["trace_ids"],[trace.trace_id])
        self.assertEqual([e["kind"] for e in ledger.events()][-6:],
           ["situation_encoded","candidate_recall","applicability_assessed",
            "guidance_applied","prediction_generated","inference"])
        _,mismatch=engine.infer(prefix,operation="clear-lowest-set-bit",
                               conditions={"bits":3},seed=213,max_new_tokens=4)
        self.assertFalse(mismatch["used"])
        ledger.close()
