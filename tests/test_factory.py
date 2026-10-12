"""Factory throughput, DAG consistency, fail-closed commits, and capacity tests."""
from tempfile import TemporaryDirectory
from urllib.request import urlopen
import json
from pathlib import Path
from unittest import TestCase

from ntsa_x.events import EventStream
from ntsa_x.factory import FactoryTask, ProductionFactory, MAX_SCALAR
from ntsa_x.nexus import NexusStore
from ntsa_x.materializer import LocalMaterializer
from ntsa_x.plugins import PluginManifest, PluginRegistry, PluginWorker


class FactoryTests(TestCase):
    def test_100_independent_verified_items_with_bounded_workers(self):
        with NexusStore(":memory:") as store:
            stream = EventStream()
            factory = ProductionFactory(store, workers=8, stream=stream)
            tasks = [
                FactoryTask(f"unit-{i:03}", "increment", amount=2, seed=i)
                for i in range(100)
            ]
            run = factory.run(tasks, batch_id="batch-100")
            self.assertEqual(run.completed, 100)
            self.assertEqual(run.failed, 0)
            self.assertEqual(run.blocked, 0)
            self.assertLessEqual(run.peak_in_flight, 8)
            self.assertGreater(run.peak_in_flight, 0)
            self.assertGreater(run.elapsed_seconds, 0)
            self.assertEqual(store.stats(), {"nodes": 100, "edges": 0})
            self.assertEqual({r.value for r in run.receipts}, set(range(2, 102)))
            self.assertTrue(all(store.verify(r.digest) for r in run.receipts))
            self.assertEqual(run.receipts[0].status, "committed")
            self.assertIn("factory.start", [e.type for e in stream.events])
            self.assertIn("factory.close", [e.type for e in stream.events])
            self.assertIn("committed_tasks_per_second", run.to_dict())

    def test_dependency_sum_and_ancestry(self):
        with NexusStore(":memory:") as store:
            tasks = [
                FactoryTask("a", amount=2),
                FactoryTask("b", amount=3),
                FactoryTask("joined", "increment", amount=4,
                            dependencies=("a", "b"), use_dependency_sum=True),
                FactoryTask("last", "negate", dependencies=("joined",),
                            use_dependency_sum=True),
            ]
            run = ProductionFactory(store, workers=4).run(tasks, batch_id="dag")
            self.assertEqual(run.completed, 4)
            indexed = {r.task_id: r for r in run.receipts}
            self.assertEqual(indexed["joined"].value, 9)
            self.assertEqual(indexed["last"].value, -9)
            join = store.read(indexed["joined"].digest)
            self.assertEqual(join.parents, (indexed["a"].digest, indexed["b"].digest))
            self.assertEqual(store.stats(), {"nodes": 4, "edges": 3})
            self.assertEqual(len(store.lineage(indexed["last"].digest)), 4)

    def test_invalid_graph_is_rejected_before_writes(self):
        with NexusStore(":memory:") as store:
            factory = ProductionFactory(store)
            for tasks in (
                (FactoryTask("x"), FactoryTask("x")),
                (FactoryTask("x", dependencies=("missing",)),),
                (FactoryTask("x", dependencies=("y",)),
                 FactoryTask("y", dependencies=("x",))),
                (FactoryTask("x", dependencies=("x",)),),
            ):
                with self.subTest(tasks=tasks), self.assertRaises(ValueError):
                    factory.run(tasks, batch_id="bad")
            self.assertEqual(store.stats()["nodes"], 0)

    def test_wrong_worker_output_blocks_dependents_but_other_work_proceeds(self):
        registry = PluginRegistry()
        registry.register(PluginManifest("ntsa.factory.scalar", "1.0.0",
                                         ("increment", "decrement", "identity", "negate")))
        worker = PluginWorker(registry)
        original = worker.invoke
        def corrupted(plugin_id, version, operation, state):
            if state["value"] == 99:
                return {"value": 9999}
            return original(plugin_id, version, operation, state)
        worker.invoke = corrupted
        with NexusStore(":memory:") as store:
            run = ProductionFactory(store, workers=3, plugin_worker=worker).run(
                [FactoryTask("bad", seed=99),
                 FactoryTask("child", dependencies=("bad",), use_dependency_sum=True),
                 FactoryTask("good", seed=4)],
                batch_id="partial",
            )
            statuses = {r.task_id: r.status for r in run.receipts}
            self.assertEqual(statuses, {"bad": "failed", "child": "blocked",
                                        "good": "committed"})
            self.assertEqual(store.stats()["nodes"], 1)
            self.assertIsNone(run.receipts[0].digest)
            self.assertIsNone(run.receipts[1].digest)

    def test_persisted_batches_reverify_after_reopening(self):
        with TemporaryDirectory() as d:
            file = Path(d) / "factory.sqlite3"
            tasks = [FactoryTask("one", seed=10), FactoryTask("two", seed=20)]
            with NexusStore(file) as store:
                first = ProductionFactory(store, workers=2).run(tasks, batch_id="same")
                self.assertEqual(first.completed, 2)
            with NexusStore(file) as store:
                self.assertTrue(all(store.verify(r.digest) for r in first.receipts))
                second = ProductionFactory(store, workers=2).run(tasks, batch_id="same")
                self.assertEqual(second.completed, 2)
                self.assertEqual(store.stats()["nodes"], 2)
                self.assertEqual([r.digest for r in first.receipts],
                                 [r.digest for r in second.receipts])

    def test_factory_telemetry_reaches_local_materializer(self):
        with NexusStore(":memory:") as store, LocalMaterializer() as viewer:
            stream = EventStream(sink=viewer.publish)
            run = ProductionFactory(store, workers=2, stream=stream).run(
                [FactoryTask("a"), FactoryTask("b")], batch_id="observed"
            )
            self.assertEqual(run.completed, 2)
            with urlopen(viewer.url + "api/events", timeout=5) as response:
                snapshot = json.load(response)
            types = [event["type"] for event in snapshot["events"]]
            self.assertIn("factory.start", types)
            self.assertIn("factory.commit", types)
            self.assertIn("factory.close", types)
            self.assertEqual(snapshot["events"][-1]["payload"]["completed"], 2)
            with urlopen(viewer.url, timeout=5) as response:
                self.assertIn(b'id="factory-rate"', response.read())
            with urlopen(viewer.url + "app.js", timeout=5) as response:
                self.assertIn(b'factory.close', response.read())

    def test_capacity_enforced_and_unsafe_inputs_denied(self):
        with NexusStore(":memory:") as store:
            with self.assertRaises(ValueError):
                ProductionFactory(store, workers=101)
            with self.assertRaises(ValueError):
                ProductionFactory(store, workers=0)
            with self.assertRaises(ValueError):
                ProductionFactory(store, max_batch=0)
            with self.assertRaises(ValueError):
                ProductionFactory(store, max_batch=1).run(
                    [FactoryTask("a"), FactoryTask("b")], batch_id="too-many"
                )
            with self.assertRaises(ValueError):
                FactoryTask("a", amount=-1)
            with self.assertRaises(ValueError):
                FactoryTask("a", seed=MAX_SCALAR + 1)
            with self.assertRaises(ValueError):
                FactoryTask("a", operation="shell")
            with self.assertRaises(ValueError):
                FactoryTask("a", dependencies=("x", "x"))
            with self.assertRaises(ValueError):
                FactoryTask("a", use_dependency_sum=True)
            self.assertEqual(store.stats()["nodes"], 0)

    def test_dependency_aggregation_range_failure_blocks_followers(self):
        with NexusStore(":memory:") as store:
            tasks = [
                FactoryTask("a", "identity", seed=MAX_SCALAR),
                FactoryTask("b", "identity", seed=MAX_SCALAR),
                FactoryTask("overflow", dependencies=("a","b"), use_dependency_sum=True),
                FactoryTask("follower", dependencies=("overflow",), use_dependency_sum=True),
            ]
            run = ProductionFactory(store, workers=2).run(tasks, batch_id="overflow")
            self.assertEqual(run.completed, 2)
            self.assertEqual(run.failed, 1)
            self.assertEqual(run.blocked, 1)
            self.assertEqual(store.stats()["nodes"], 2)


if __name__ == "__main__":
    import unittest
    unittest.main()
