"""Reproducible standard-library-only tests of the SCASI III NTSA-X adapter."""
from __future__ import annotations
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
import sys
import time
import unittest
from ntsa_x.scasi_iii import Objective, PredictiveSuperCycle, native_scasi_extension


@dataclass(frozen=True)
class Record:
    content: str
    source: str = "synthetic-fixture"
    created: float = 0.0
    verified: bool = False


class FakeMemory:
    def __init__(self, records=()):
        self.records = list(records)
        self.recall_calls = 0

    def recall(self, query, *, session, limit=5):
        self.recall_calls += 1
        words = set(query.casefold().split())
        return [r for r in self.records if words & set(r.content.casefold().split())][:limit]


GOALS = (Objective("deploy", ("deploy service", "production release"),
                   ("release manifest", "rollback plan")),
         Objective("training", ("train model", "learning upgrade"),
                   ("training data rights", "held out evaluation")))


class SCASIIIRegression(unittest.TestCase):
    def test_warm_resolves_before_matching_input(self):
        m = FakeMemory([Record("release manifest signed", created=time.time())])
        c = PredictiveSuperCycle(GOALS, memory=m)
        c.warm(("deploy",))
        calls = m.recall_calls
        result = c.resolve("deploy service now")
        self.assertTrue(result["forecasts"][0]["prepared_before_request"])
        self.assertEqual(calls, m.recall_calls)
        gaps = {g["prerequisite"]:g["state"] for g in result["forecasts"][0]["knowledge_gaps"]}
        self.assertEqual(gaps, {"release manifest":"unverified", "rollback plan":"missing"})
        self.assertEqual(c.snapshot()["metrics"]["cache_hits"], 1)

    def test_no_fabricated_knowledge(self):
        c = PredictiveSuperCycle(GOALS, memory=FakeMemory())
        out = c.resolve("train model")
        self.assertEqual(len(out["forecasts"]), 1)
        self.assertTrue(all(g["state"]=="missing" for g in out["forecasts"][0]["knowledge_gaps"]))
        self.assertFalse(out["candidate_facts_are_verified"])
        self.assertFalse(out["physical_negative_latency_claimed"])

    def test_stale_is_not_current(self):
        c = PredictiveSuperCycle(GOALS, memory=FakeMemory([Record("rollback plan", created=time.time()-1000)]), horizon_seconds=15)
        f = c.resolve("deploy service")["forecasts"][0]
        self.assertEqual(f["knowledge_gaps"][1]["state"], "stale")

    def test_forward_and_reverse_and_adaptation(self):
        c = PredictiveSuperCycle(GOALS, memory=FakeMemory())
        out = c.resolve("train model please")["forecasts"][0]
        self.assertEqual(out["objective_id"], "training")
        self.assertEqual(out["reverse_requirements"], ("training data rights", "held out evaluation"))
        self.assertEqual(c.record_outcome("training", success=True)["parameter_updates"], 0)
        self.assertEqual(c.snapshot()["outcome_samples"]["training"], 1)

    def test_bounded_and_no_unrelated_prefetch(self):
        c = PredictiveSuperCycle(GOALS, memory=FakeMemory())
        self.assertEqual(c.resolve("tell me a joke")["forecasts"], [])
        with self.assertRaises(ValueError):
            c.warm(("nonexistent",))
        with self.assertRaises(ValueError):
            c.resolve("x" * 4097)

    def test_native_extension_adapter(self):
        c = PredictiveSuperCycle(GOALS, memory=FakeMemory())
        c.warm(("deploy",))
        stub = ModuleType("ntsa_x.unified_cognition")
        stub.NativeExtension = lambda name, supports, contribute: SimpleNamespace(
            name=name, supports=supports, contribute=contribute)
        with patch.dict(sys.modules, {"ntsa_x.unified_cognition": stub}):
            ext = native_scasi_extension(c)
        self.assertEqual(ext.name, "scasi3.predictive")
        self.assertFalse(ext.supports("unrelated subject"))
        self.assertTrue(ext.supports("deploy service"))
        observation = ext.contribute(SimpleNamespace(request="deploy service"))
        self.assertIn("prepared=True", observation)
        self.assertIn("rollback plan:missing", observation)

    def test_digest_stable_for_identical_evidence(self):
        m = FakeMemory([Record("release manifest", created=time.time())])
        c = PredictiveSuperCycle(GOALS, memory=m)
        a = c.resolve("deploy service")["forecasts"][0]
        b = c.resolve("deploy service")["forecasts"][0]
        self.assertEqual(a["provenance_digest"], b["provenance_digest"])


if __name__ == "__main__":
    unittest.main()
