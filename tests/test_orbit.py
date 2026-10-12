import unittest
from ntsa_x import (
    Check, ConvergenceEngine, EventStream, MemoryChain, Operator, OrbitEngine,
)


CATEGORIES = ("state", "semantic", "causal", "constraint", "evidence", "delta")


def verify_target(state):
    ok = state.get("value") == 3
    return [Check(category, ok, "test-oracle", external=category == "evidence") for category in CATEGORIES]


class TestConvergence(unittest.TestCase):
    def test_missing_evidence_blocks_closure(self):
        checks = [Check(c, True, "self") for c in CATEGORIES]
        report = ConvergenceEngine().evaluate(checks)
        self.assertFalse(report.converged)
        self.assertEqual(report.score, 1.0)
        self.assertFalse(report.external_evidence)

    def test_contradiction_and_counter_orbit_mismatch_block(self):
        checks = [Check(c, True, "oracle", external=c == "evidence") for c in CATEGORIES]
        self.assertFalse(ConvergenceEngine().evaluate(checks, contradictions=["unresolved"]).converged)
        self.assertFalse(ConvergenceEngine().evaluate(checks, forward={"x": 1}, reverse={"x": 2}).converged)
        self.assertTrue(ConvergenceEngine().evaluate(checks, forward={"x": 1}, reverse={"x": 1}).converged)
        with self.assertRaises(ValueError):
            ConvergenceEngine().evaluate(checks, forward={"x": 1})

    def test_failed_and_missing_checks(self):
        checks = [Check("evidence", True, "oracle", external=True), Check("state", False, "oracle")]
        report = ConvergenceEngine().evaluate(checks)
        self.assertFalse(report.converged)
        self.assertIn("state", report.failed)
        self.assertIn("semantic", report.missing)


class TestOrbit(unittest.TestCase):
    def test_executes_to_verified_commit_and_recall_revalidation(self):
        chain = MemoryChain()
        events = EventStream()
        calls = {"n": 0}
        def increment(s):
            calls["n"] += 1
            return {"value": s["value"] + 1}
        engine = OrbitEngine(verifier=verify_target, chain=chain, stream=events, max_steps=5, max_cost=5)
        op = Operator("increment", increment, axis=(1, 1, -1))
        result = engine.run({"value": 0}, [op], context_id="case-a")
        self.assertTrue(result.verified)
        self.assertEqual(result.steps, 3)
        self.assertEqual(result.position, (3, 3, -3))
        self.assertTrue(chain.verify(result.committed_digest))
        self.assertEqual(len(chain.blocks), 1)
        self.assertIn("commit.append", [e.type for e in events.events])
        recalled = engine.run({"value": 0}, [op], context_id="case-a", recall_digest=result.committed_digest)
        self.assertEqual(recalled.steps, 0)
        self.assertEqual(calls["n"], 3)
        self.assertEqual(recalled.reason, "verified_recall")
        mismatch = engine.run({"value": 0}, [op], context_id="other", recall_digest=result.committed_digest)
        self.assertEqual(mismatch.steps, 3)
        self.assertIn("recall.miss", [e.type for e in events.events])

    def test_unverified_never_committed(self):
        engine = OrbitEngine(verifier=lambda state: [], max_steps=2)
        result = engine.run({"value": 0}, [Operator("go", lambda s: {"value": 1})], context_id="case")
        self.assertFalse(result.verified)
        self.assertEqual(result.reason, "step_limit")
        self.assertEqual(len(engine.chain.blocks), 0)

    def test_predicates_budget_and_deny_external(self):
        events = EventStream()
        engine = OrbitEngine(verifier=verify_target, stream=events, max_steps=4, max_cost=2)
        operators = [
            Operator("forbidden", lambda s: {"value": 3}, external_effects=True, priority=100),
            Operator("ineligible", lambda s: {"value": 3}, eligible=lambda s: False, priority=90),
            Operator("too_expensive", lambda s: {"value": 3}, estimated_cost=3, priority=80),
            Operator("increment", lambda s: {"value": s["value"] + 1}, priority=1),
        ]
        result = engine.run({"value": 0}, operators, context_id="case")
        self.assertFalse(result.verified)
        self.assertEqual(result.steps, 2)
        self.assertEqual(result.reason, "no_operator")
        self.assertIn("plugin.denied", [e.type for e in events.events])
        self.assertEqual(len(engine.chain.blocks), 0)

    def test_recall_tampering_rejected(self):
        chain = MemoryChain()
        block = chain.append({"context_id": "case", "state": {"value": 3}}, verified=True)
        chain.blocks[block.digest].payload["state"]["value"] = 5
        self.assertFalse(chain.verify(block.digest))
        with self.assertRaisesRegex(ValueError, "Hash collision or inconsistent content"):
            OrbitEngine(verifier=verify_target, chain=chain, max_steps=1).run(
                {"value": 0}, [Operator("set", lambda s: {"value": 3})],
                context_id="case", recall_digest=block.digest,
            )

    def test_materializer_is_metadata_only_non_authoritative(self):
        def broken_sink(event):
            raise RuntimeError("renderer unavailable")
        stream = EventStream(sink=broken_sink, max_events=2)
        stream.emit("observe", step=0)
        stream.emit("orbit.step", step=1)
        stream.emit("orbit.close", closure=True)
        self.assertEqual(len(stream.events), 2)
        self.assertEqual(stream.dropped_events, 1)
        envelope = stream.events[-1].to_dict()
        self.assertEqual(envelope["redaction_level"], "metadata")
        self.assertEqual(envelope["schema_version"], "1.0")
        with self.assertRaises(ValueError):
            stream.emit("observe", password="secret")
        with self.assertRaises(ValueError):
            stream.emit("observe", reason="x" * 257)


if __name__ == "__main__":
    unittest.main()
