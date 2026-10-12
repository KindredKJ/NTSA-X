import json
import sqlite3
from tempfile import TemporaryDirectory
from pathlib import Path
from urllib.request import urlopen
import unittest

from ntsa_x.convergence import Check, ConvergenceEngine
from ntsa_x.counter_orbit import CounterOrbitEngine, PairedOperator
from ntsa_x.events import EventStream
from ntsa_x.materializer import LocalMaterializer
from ntsa_x.nexus import NexusStore

CATEGORIES = ("state", "semantic", "causal", "constraint", "evidence", "delta")


def checks(state):
    ok = state.get("value") == 3
    return [Check(c, ok, "independent-test-oracle", external=(c == "evidence")) for c in CATEGORIES]


def report():
    return ConvergenceEngine().evaluate([
        Check(c, True, "test-oracle", external=(c == "evidence")) for c in CATEGORIES
    ])


class TestNexus(unittest.TestCase):
    def test_durable_multi_parent_graph_and_reopen(self):
        with TemporaryDirectory() as folder:
            file = Path(folder) / "nexus.sqlite3"
            with NexusStore(file) as graph:
                a = graph.append({"name": "A"}, report=report())
                b = graph.append({"name": "B"}, report=report())
                child = graph.append({"name": "C"}, report=report(), parents=(a.digest, b.digest))
                self.assertTrue(graph.verify(child.digest))
                self.assertEqual(graph.stats(), {"nodes": 3, "edges": 2})
                self.assertEqual(graph.append({"name": "C"}, report=report(), parents=(a.digest, b.digest)).digest, child.digest)
                self.assertEqual(len(graph.lineage(child.digest)), 3)
                with self.assertRaises(ValueError):
                    graph.append({"bad": 1}, report=report(), parents=("missing",))
                self.assertEqual(graph.stats(), {"nodes": 3, "edges": 2})
            with NexusStore(file) as reopened:
                self.assertTrue(reopened.verify(child.digest))
                self.assertEqual(reopened.read(child.digest).parents, (a.digest, b.digest))
                self.assertEqual(reopened.stats(), {"nodes": 3, "edges": 2})

    def test_fail_closed_without_proof(self):
        with NexusStore(":memory:") as graph:
            bad = ConvergenceEngine().evaluate([Check("state", True, "self")])
            with self.assertRaises(ValueError):
                graph.append({"state": "unsafe"}, report=bad)
            with self.assertRaises(ValueError):
                graph.append({"state": "unsafe"}, report=None)
            self.assertEqual(graph.stats(), {"nodes": 0, "edges": 0})

    def test_tamper_detection_and_rollback(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "store.sqlite3"
            with NexusStore(path) as graph:
                parent = graph.append({"value": 1}, report=report())
                child = graph.append({"value": 2}, report=report(), parents=(parent.digest,))
                with sqlite3.connect(path) as raw:
                    raw.execute("UPDATE nodes SET payload=? WHERE digest=?", ('{"value":999}', parent.digest))
                self.assertFalse(graph.verify(child.digest))
                with self.assertRaises(ValueError):
                    graph.read(child.digest)
                with self.assertRaises(ValueError):
                    graph.append({"value": 4}, report=report(), parents=(child.digest,))
                self.assertEqual(graph.stats()["nodes"], 2)

    def test_reject_duplicate_parents_and_invalid_values(self):
        with NexusStore(":memory:") as graph:
            parent = graph.append({"valid": 1}, report=report())
            with self.assertRaises(ValueError):
                graph.append({"bad": 1}, report=report(), parents=(parent.digest, parent.digest))
            with self.assertRaises(ValueError):
                graph.append({"nan": float("nan")}, report=report())


class TestCounterOrbit(unittest.TestCase):
    def make_engine(self, store, **kwargs):
        return CounterOrbitEngine(
            verifier=checks,
            project_forward=lambda s: {"value": s["value"]},
            project_reverse=lambda s: {"value": s["value"]},
            store=store, **kwargs,
        )

    def test_both_orbits_converge_and_recall_after_reopen(self):
        with TemporaryDirectory() as folder:
            db = Path(folder) / "nexus.sqlite3"
            telemetry = EventStream()
            with NexusStore(db) as store:
                engine = self.make_engine(store, stream=telemetry, max_steps=5, max_cost=10)
                pair = PairedOperator("advance", lambda s: {"value": s["value"] + 1},
                                      lambda s: {"value": s["value"] + 1}, axis=(1, 0, -1))
                result = engine.run({"value": 0}, {"value": 0}, [pair], context_id="test")
                self.assertTrue(result.verified)
                self.assertEqual(result.steps, 3)
                self.assertEqual(result.resolved, {"value": 3})
                self.assertEqual(result.position, (3, 0, -3))
                self.assertEqual(result.spent, 6)
                self.assertTrue(store.verify(result.committed_digest))
                self.assertEqual(store.stats()["nodes"], 1)
                self.assertIn("counter_orbit.step", [e.type for e in telemetry.events])
                self.assertIn("counter_orbit.compare", [e.type for e in telemetry.events])
                digest = result.committed_digest
            with NexusStore(db) as reopened:
                engine = self.make_engine(reopened)
                remembered = engine.run({"value": 0}, {"value": 0}, [], context_id="test", recall_digest=digest)
                self.assertTrue(remembered.verified)
                self.assertEqual(remembered.steps, 0)
                self.assertEqual(remembered.reason, "verified_recall")

    def test_agreement_and_evidence_are_both_required(self):
        with NexusStore(":memory:") as store:
            engine = self.make_engine(store, max_steps=3)
            diverging = PairedOperator("diverge", lambda s: {"value": s["value"] + 1},
                                      lambda s: {"value": s["value"] + 2})
            result = engine.run({"value": 0}, {"value": 0}, [diverging], context_id="different")
            self.assertFalse(result.verified)
            self.assertEqual(store.stats()["nodes"], 0)
            no_evidence = CounterOrbitEngine(
                verifier=lambda s: [Check(c, True, "self") for c in CATEGORIES],
                project_forward=lambda s: s,
                project_reverse=lambda s: s,
                store=store, max_steps=1,
            )
            same = PairedOperator("same", lambda s: s, lambda s: s)
            self.assertFalse(no_evidence.run({"v": 1}, {"v": 1}, [same], context_id="self").verified)
            self.assertEqual(store.stats()["nodes"], 0)

    def test_contradiction_blocks_commit(self):
        with NexusStore(":memory:") as store:
            engine = self.make_engine(store, max_steps=3, contradictions=lambda f, r: ["unresolved"])
            pair = PairedOperator("advance", lambda s: {"value": s["value"] + 1},
                                  lambda s: {"value": s["value"] + 1})
            self.assertFalse(engine.run({"value": 0}, {"value": 0}, [pair], context_id="contradicted").verified)
            self.assertEqual(store.stats()["nodes"], 0)

    def test_budget_and_declared_external_effects(self):
        with NexusStore(":memory:") as store:
            telemetry = EventStream()
            engine = self.make_engine(store, max_steps=3, max_cost=2, stream=telemetry)
            forbidden = PairedOperator("external", lambda s: {"value": 3}, lambda s: {"value": 3},
                                       external_effects=True, priority=100)
            advance = PairedOperator("advance", lambda s: {"value": s["value"] + 1},
                                     lambda s: {"value": s["value"] + 1})
            result = engine.run({"value": 0}, {"value": 0}, [forbidden, advance], context_id="budget")
            self.assertFalse(result.verified)
            self.assertEqual(result.steps, 1)
            self.assertEqual(result.reason, "no_operator")
            self.assertIn("plugin.denied", [e.type for e in telemetry.events])
            self.assertEqual(store.stats()["nodes"], 0)

    def test_context_mismatch_does_not_recall(self):
        with NexusStore(":memory:") as store:
            engine = self.make_engine(store, max_steps=3)
            pair = PairedOperator("advance", lambda s: {"value": s["value"] + 1},
                                  lambda s: {"value": s["value"] + 1})
            result = engine.run({"value": 0}, {"value": 0}, [pair], context_id="old")
            rerun = engine.run({"value": 0}, {"value": 0}, [pair], context_id="new",
                               recall_digest=result.committed_digest)
            self.assertEqual(rerun.steps, 3)
            self.assertEqual(rerun.reason, "verified_convergence")


class TestMaterializer(unittest.TestCase):
    def test_http_dashboard_snapshot_and_live_sse(self):
        with LocalMaterializer() as viewer:
            telemetry = EventStream(sink=viewer.publish)
            with urlopen(viewer.url + "health", timeout=5) as response:
                self.assertEqual(json.load(response)["status"], "ok")
            with urlopen(viewer.url, timeout=5) as response:
                self.assertIn(b"Materializer", response.read())
            with urlopen(viewer.url + "app.js", timeout=5) as response:
                self.assertIn(b"EventSource", response.read())
            with urlopen(viewer.url + "events", timeout=5) as sse:
                self.assertTrue(sse.readline().startswith(b": connected"))
                self.assertEqual(sse.readline(), b"\n")
                telemetry.emit("counter_orbit.step", step=1, axis_x=1, axis_y=0, axis_z=-1)
                line = sse.readline()
                self.assertTrue(line.startswith(b"data: "))
                data = json.loads(line[6:])
                self.assertEqual(data["type"], "counter_orbit.step")
                self.assertNotIn("state", data["payload"])
            with urlopen(viewer.url + "api/events", timeout=5) as response:
                snapshot = json.load(response)
                self.assertEqual(len(snapshot["events"]), 1)
                self.assertEqual(snapshot["events"][0]["payload"]["step"], 1)

    def test_payload_filter_and_loopback_only(self):
        with self.assertRaises(ValueError):
            LocalMaterializer(host="0.0.0.0")
        with LocalMaterializer() as viewer:
            with self.assertRaises(ValueError):
                viewer.publish({"schema_version": "1.0", "redaction_level": "metadata",
                                "payload": {"password": "secret"}})


if __name__ == "__main__":
    unittest.main()
