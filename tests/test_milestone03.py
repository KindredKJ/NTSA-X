"""Milestone 03: evidence-scoped reusable convergences and fixed worker tests."""
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, mock
from urllib.request import urlopen
import sqlite3
import subprocess

from ntsa_x.convergence import Check, ConvergenceEngine
from ntsa_x.counter_orbit import PairedOperator
from ntsa_x.events import EventStream
from ntsa_x.indexed_orbit import IndexedCounterOrbitEngine
from ntsa_x.kindred_contracts import KindredContracts
from ntsa_x.materializer import LocalMaterializer
from ntsa_x.nexus import NexusStore
from ntsa_x.plugins import PluginFailure, PluginManifest, PluginRegistry, PluginWorker
from ntsa_x.reuse import ConvergenceLibrary, fingerprint


CATEGORIES = ("state", "semantic", "causal", "constraint", "evidence", "delta")


def oracle(state):
    good = state.get("value") == 3
    return [Check(category, good, "independent-toy-verifier", external=(category == "evidence"))
            for category in CATEGORIES]


def agreed_report():
    checks = [Check(category, True, "toy-verifier", external=(category == "evidence"))
              for category in CATEGORIES]
    return ConvergenceEngine().evaluate(checks, forward={"value": 3}, reverse={"value": 3})


class ReuseTests(TestCase):
    def engine(self, store, stream=None, verifier=oracle):
        return IndexedCounterOrbitEngine(
            library=ConvergenceLibrary(store),
            verifier=verifier, store=store, stream=stream,
            project_forward=lambda value: value,
            project_reverse=lambda value: value,
            max_steps=4, max_cost=8,
        )

    def test_scoped_exact_input_reuse_persists_and_revalidates(self):
        with TemporaryDirectory() as tmp:
            db = Path(tmp) / "reused.db"
            called = {"n": 0}
            def up(s):
                called["n"] += 1
                return {"value": s["value"] + 1}
            op = PairedOperator("increase", up, lambda s: {"value": s["value"] + 1})
            with NexusStore(db) as store:
                engine = self.engine(store)
                first = engine.run({"value": 0}, {"value": 0}, [op],
                                   context_id="job", reuse_scope="sum/v1+goal3",
                                   verifier_revision="oracle/v1")
                self.assertTrue(first.verified)
                self.assertEqual(first.steps, 3)
                self.assertTrue(store.verify(first.committed_digest))
                # one result + one SHA-linked index attestation
                self.assertEqual(store.stats()["nodes"], 2)
                self.assertEqual(called["n"], 3)
            with NexusStore(db) as store:
                engine = self.engine(store)
                again = engine.run({"value": 0}, {"value": 0}, [],
                                   context_id="job", reuse_scope="sum/v1+goal3",
                                   verifier_revision="oracle/v1")
                self.assertEqual(again.reason, "indexed_convergence")
                self.assertEqual(again.steps, 0)
                self.assertEqual(again.committed_digest, first.committed_digest)
                missed = engine.run({"value": 0}, {"value": 0}, [],
                                    context_id="job", reuse_scope="sum/v1+goal3",
                                    verifier_revision="oracle/v2")
                self.assertFalse(missed.verified)
                self.assertEqual(missed.reason, "no_operator")
                self.assertEqual(store.stats()["nodes"], 2)

    def test_new_inputs_and_scope_do_not_reuse(self):
        with NexusStore(":memory:") as store:
            engine = self.engine(store)
            op = PairedOperator("increase", lambda s: {"value": s["value"] + 1},
                                lambda s: {"value": s["value"] + 1})
            first = engine.run({"value": 0}, {"value": 0}, [op], context_id="job",
                               reuse_scope="alg/v1", verifier_revision="verify/v1")
            self.assertTrue(first.verified)
            for forward, scope, rev in [
                ({"value": 1}, "alg/v1", "verify/v1"),
                ({"value": 0}, "alg/v2", "verify/v1"),
                ({"value": 0}, "alg/v1", "verify/v2"),
            ]:
                trial = engine.run(forward, {"value": 0}, [], context_id="job",
                                   reuse_scope=scope, verifier_revision=rev)
                self.assertFalse(trial.verified)

    def test_stale_evidence_refuses_reuse(self):
        with NexusStore(":memory:") as store:
            engine = self.engine(store)
            op = PairedOperator("up", lambda s: {"value": s["value"] + 1},
                                lambda s: {"value": s["value"] + 1})
            result = engine.run({"value": 0}, {"value": 0}, [op], context_id="job",
                                reuse_scope="v1", verifier_revision="v1")
            self.assertTrue(result.verified)
            engine.verifier = lambda state: [Check(c, False, "current-oracle") for c in CATEGORIES]
            again = engine.run({"value": 0}, {"value": 0}, [], context_id="job",
                               reuse_scope="v1", verifier_revision="v1")
            self.assertFalse(again.verified)
            self.assertNotEqual(again.reason, "indexed_convergence")

    def test_index_redirection_fails_closed(self):
        with NexusStore(":memory:") as store:
            lib = ConvergenceLibrary(store)
            engine = self.engine(store)
            op = PairedOperator("up", lambda s: {"value": s["value"] + 1},
                                lambda s: {"value": s["value"] + 1})
            result = engine.run({"value": 0}, {"value": 0}, [op], context_id="job",
                                reuse_scope="alg", verifier_revision="v1")
            self.assertTrue(result.verified)
            with store._lock:
                # Redirect index to the verified result directly, rather than
                # the immutable binding that commits to fingerprint and scope.
                store._db.execute("UPDATE convergence_index SET node_digest=?",
                                  (result.committed_digest,))
            self.assertIsNone(lib.get(context_id="job", scope="alg", revision="v1",
                                      inputs=({"value": 0}, {"value": 0})))

    def test_composition_preserves_proof_chain_and_rejects_bad_report(self):
        with NexusStore(":memory:") as store:
            lib = ConvergenceLibrary(store)
            report = agreed_report()
            a = store.append({"step": "a"}, report=report)
            b = store.append({"step": "b"}, report=report)
            c = lib.compose(
                context_id="assembled", scope="macro/v1", revision="verifier/v1",
                inputs=({"seed": 9}, {"seed": 9}), members=(a.digest, b.digest),
                forward={"value": 3}, reverse={"value": 3},
                resolved={"value": 3}, report=report,
            )
            self.assertEqual(c.payload["kind"], "higher_order_convergence/v1")
            self.assertEqual(c.parents, (a.digest, b.digest))
            self.assertTrue(store.verify(c.digest))
            looked_up = lib.get(context_id="assembled", scope="macro/v1",
                                revision="verifier/v1",
                                inputs=({"seed": 9}, {"seed": 9}))
            self.assertEqual(looked_up.digest, c.digest)
            with self.assertRaises(ValueError):
                lib.compose(context_id="bad", scope="macro", revision="v1",
                            inputs=({}, {}), members=(a.digest,),
                            forward={}, reverse={}, resolved={},
                            report=ConvergenceEngine().evaluate([]))


class WorkerTests(TestCase):
    def make_worker(self, stream=None):
        registry = PluginRegistry()
        registry.register(PluginManifest("scalar.demo", "1.0.0",
                                         ("increment", "decrement", "identity", "negate")))
        return PluginWorker(registry, timeout=5, stream=stream)

    def test_versioned_worker_in_separate_process(self):
        worker = self.make_worker()
        self.assertEqual(worker.invoke("scalar.demo", "1.0.0", "increment",
                                       {"value": 2, "amount": 3}), {"value": 5})
        self.assertEqual(worker.invoke("scalar.demo", "1.0.0", "decrement",
                                       {"value": 5, "amount": 2}), {"value": 3})

    def test_unregistered_or_bad_input_is_denied(self):
        stream = EventStream()
        worker = self.make_worker(stream)
        with self.assertRaises(PermissionError):
            worker.invoke("scalar.demo", "0.0.9", "increment",
                          {"value": 1, "amount": 1})
        self.assertIn("plugin.denied", [event.type for event in stream.events])
        with self.assertRaises(PermissionError):
            worker.invoke("scalar.demo", "1.0.0", "network",
                          {"value": 1, "amount": 1})
        with self.assertRaises(ValueError):
            worker.invoke("scalar.demo", "1.0.0", "increment",
                          {"value": 1, "amount": 1, "secret": "private"})
        with self.assertRaises(ValueError):
            PluginManifest("danger", "1", ("shell",))

    def test_worker_timeout_fails_closed(self):
        worker = self.make_worker()
        with mock.patch("ntsa_x.plugins.subprocess.run",
                        side_effect=subprocess.TimeoutExpired("python", 1)):
            with self.assertRaises(PluginFailure):
                worker.invoke("scalar.demo", "1.0.0", "increment",
                              {"value": 1, "amount": 1})

    def test_registry_immutable(self):
        registry = PluginRegistry()
        registry.register(PluginManifest("a", "1", ("increment",)))
        with self.assertRaises(ValueError):
            registry.register(PluginManifest("a", "2", ("increment",)))

    def test_worker_operator_in_real_paired_orbit(self):
        worker = self.make_worker()
        pair = worker.paired_operator("scalar.demo", "1.0.0",
                                      forward_op="increment", reverse_op="increment")
        with NexusStore(":memory:") as store:
            engine = self.engine_for_pair(store)
            result = engine.run({"value": 0}, {"value": 0}, [pair], context_id="worker")
            self.assertTrue(result.verified)
            self.assertEqual(result.steps, 3)
            self.assertEqual(result.resolved, {"value": 3})

    @staticmethod
    def engine_for_pair(store):
        from ntsa_x.counter_orbit import CounterOrbitEngine
        return CounterOrbitEngine(
            store=store, verifier=oracle,
            project_forward=lambda s: s, project_reverse=lambda s: s,
            max_steps=3, max_cost=6,
        )


class ContractTests(TestCase):
    def test_ports_generate_metadata_only_contract_envelopes(self):
        registry = KindredContracts()
        e = EventStream().emit("orbit.step", step=1, axis_x=1).to_dict()
        wrapped = registry.wrap_event("kindred.strata", e)
        self.assertEqual(wrapped["schema_version"], "kindred.ntsa/1")
        self.assertEqual(wrapped["event"]["payload"], {"step": 1, "axis_x": 1})
        self.assertEqual(wrapped["mode"], "observe")
        self.assertNotIn("state", wrapped["event"])
        with self.assertRaises(PermissionError):
            registry.wrap_event("kindred.zero-point", e)
        with self.assertRaises(PermissionError):
            registry.wrap_event("outside", e)

    def test_recall_digest_only(self):
        packet = KindredContracts().wrap_recall_ref("0" * 64)
        self.assertEqual(packet["mode"], "read")
        self.assertEqual(packet["reference"]["digest"], "0" * 64)
        with self.assertRaises(ValueError):
            KindredContracts().wrap_recall_ref("0xabc")


if __name__ == "__main__":
    import unittest
    unittest.main()
