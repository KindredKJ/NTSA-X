"""Persistent Scalar: cross-platform process, concurrency and failure tests."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from ntsa_x.factory import FactoryTask, ProductionFactory
from ntsa_x.nexus import NexusStore
from ntsa_x.plugins import PluginFailure, PluginManifest, PluginRegistry
from ntsa_x.persistent_workers import PersistentPluginWorker


def registry():
    instance = PluginRegistry()
    instance.register(PluginManifest(
        "ntsa.factory.scalar", "1.0.0",
        ("increment", "decrement", "identity", "negate"),
    ))
    return instance


class PersistentScalarTests(TestCase):
    def test_one_process_serves_100_commands(self):
        with PersistentPluginWorker(registry(), workers=1, timeout=5) as pool:
            first_pid = pool._all[0].process.pid
            for i in range(100):
                self.assertEqual(pool.invoke(
                    "ntsa.factory.scalar", "1.0.0", "increment",
                    {"value": i, "amount": 2}), {"value": i + 2})
            self.assertEqual(pool._all[0].process.pid, first_pid)
            self.assertEqual(pool.stats["starts"], 1)
            self.assertEqual(pool.stats["requests"], 100)
            self.assertEqual(pool.stats["restarts"], 0)

    def test_concurrent_dispatch_no_response_cross_talk(self):
        with PersistentPluginWorker(registry(), workers=4, timeout=5) as pool:
            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = [
                    executor.submit(pool.invoke, "ntsa.factory.scalar",
                                    "1.0.0", "increment",
                                    {"value": i, "amount": 3})
                    for i in range(100)
                ]
                results = [f.result() for f in futures]
            self.assertEqual([r["value"] for r in results], list(range(3, 103)))
            self.assertEqual(pool.stats["requests"], 100)

    def test_invalid_inputs_and_versions_denied(self):
        with PersistentPluginWorker(registry(), workers=1, timeout=5) as pool:
            with self.assertRaises(PermissionError):
                pool.invoke("ntsa.factory.scalar", "v0", "increment",
                            {"value": 0, "amount": 1})
            with self.assertRaises(ValueError):
                pool.invoke("ntsa.factory.scalar", "1.0.0", "increment",
                            {"value": 0, "amount": 1, "secret": "no"})
            with self.assertRaises(ValueError):
                pool.invoke("ntsa.factory.scalar", "1.0.0", "increment",
                            {"value": True, "amount": 1})
            self.assertEqual(pool.stats["requests"], 0)

    def test_killed_worker_fails_current_call_then_restarts(self):
        with PersistentPluginWorker(registry(), workers=1, timeout=5) as pool:
            old_pid = pool._all[0].process.pid
            pool._all[0].process.kill()
            pool._all[0].process.wait(timeout=5)
            with self.assertRaises(PluginFailure):
                pool.invoke("ntsa.factory.scalar", "1.0.0", "increment",
                            {"value": 1, "amount": 1})
            self.assertNotEqual(pool._all[0].process.pid, old_pid)
            self.assertEqual(pool.invoke("ntsa.factory.scalar", "1.0.0",
                                         "increment", {"value": 2, "amount": 1}),
                             {"value": 3})
            self.assertEqual(pool.stats["restarts"], 1)

    def test_wrong_reply_id_rejected(self):
        with PersistentPluginWorker(registry(), workers=1, timeout=5) as pool:
            slot = pool._all[0]
            real = slot.inbox
            class WrongReply:
                def get(self, timeout=None):
                    return b'{"id":"wrong","ok":true,"state":{"value":3}}\n'
            slot.inbox = WrongReply()
            try:
                with self.assertRaises(PluginFailure):
                    pool.invoke("ntsa.factory.scalar", "1.0.0",
                                "increment", {"value": 2, "amount": 1})
            finally:
                slot.inbox = real
            self.assertEqual(pool.stats["restarts"], 1)
            self.assertEqual(pool.invoke("ntsa.factory.scalar", "1.0.0",
                                         "increment", {"value": 7, "amount": 1}),
                             {"value": 8})

    def test_worker_recycling(self):
        with PersistentPluginWorker(registry(), workers=1, timeout=5,
                                    max_tasks_per_worker=2) as pool:
            for _ in range(2):
                self.assertEqual(pool.invoke(
                    "ntsa.factory.scalar", "1.0.0", "identity",
                    {"value": 5, "amount": 0}), {"value": 5})
            self.assertEqual(pool.stats["restarts"], 1)
            self.assertEqual(pool.stats["starts"], 2)

    def test_factory_dag_and_reopen_nexus(self):
        with TemporaryDirectory() as folder:
            db = Path(folder) / "nexus.db"
            with PersistentPluginWorker(registry(), workers=4, timeout=5) as pool:
                with NexusStore(db) as store:
                    tasks = [
                        FactoryTask("first", seed=1, amount=2),
                        FactoryTask("second", seed=4, amount=3),
                        FactoryTask("joined", dependencies=("first", "second"),
                                    use_dependency_sum=True, amount=10),
                    ]
                    result = ProductionFactory(store, workers=4, plugin_worker=pool).run(
                        tasks, batch_id="persistent")
                    self.assertEqual(result.completed, 3)
                    self.assertEqual([r.value for r in result.receipts], [3, 7, 20])
                    self.assertTrue(all(store.verify(r.digest) for r in result.receipts))
                    self.assertEqual(store.stats()["edges"], 2)
                    digests = [r.digest for r in result.receipts]
            with NexusStore(db) as reopened:
                self.assertTrue(all(reopened.verify(d) for d in digests))

    def test_factory_isolates_dead_process_and_blocks_dependents(self):
        with PersistentPluginWorker(registry(), workers=1, timeout=5) as pool:
            pool._all[0].process.kill()
            pool._all[0].process.wait(timeout=5)
            with NexusStore(":memory:") as store:
                run = ProductionFactory(store, workers=1, plugin_worker=pool).run(
                    [FactoryTask("broken", seed=1),
                     FactoryTask("child", dependencies=("broken",),
                                 use_dependency_sum=True),
                     FactoryTask("healthy", seed=4)], batch_id="recovery")
                self.assertEqual({r.task_id: r.status for r in run.receipts},
                                 {"broken": "failed", "child": "blocked",
                                  "healthy": "committed"})
                self.assertEqual(store.stats()["nodes"], 1)

    def test_close_terminates_all_processes(self):
        pool = PersistentPluginWorker(registry(), workers=2, timeout=5)
        pool.start()
        children = [slot.process for slot in pool._all]
        pool.close()
        self.assertTrue(all(child.poll() is not None for child in children))
        with self.assertRaises(PluginFailure):
            pool.invoke("ntsa.factory.scalar", "1.0.0", "identity",
                        {"value": 1, "amount": 0})


if __name__ == "__main__":
    import unittest
    unittest.main()
