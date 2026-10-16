"""Unified cognition: memory multi-types, graph, native extensions and framing."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from ntsa_x.memory_fabric import MemoryFabric
from ntsa_x.unified_cognition import (
    UnifiedCognitiveMechanism, NativeExtension, scalar_extension,
    workspace_extension, BOS, USER, EOS, ASSISTANT,
)


class DummyConfig:
    max_context = 1024
    max_answer = 32


class DummyModel:
    config = DummyConfig()
    architecture = "ntsa-x-cog-fu-discrete-v1"

    def __init__(self):
        self.prefix = []

    def unified_draft(self, prefix, **_kwargs):
        self.prefix = prefix
        return list(b"Okay.")


def prefix(query: str) -> list[int]:
    return [BOS, USER, *query.encode("utf-8"), EOS, ASSISTANT]


class MemoryFabricTests(unittest.TestCase):
    def test_read_write_guards_and_typed_recall(self):
        with MemoryFabric(":memory:") as store:
            with self.assertRaises(PermissionError):
                store.remember("semantic", "A penguin is a bird",
                               source="operator")
            self.assertEqual(store.recall("penguin"), [])
        with MemoryFabric(":memory:", writable=True) as store:
            ids = [
                store.remember("semantic", "A penguin is a bird", source="user"),
                store.remember("procedural", "To build: first run tests",
                               source="user"),
                store.remember("episodic", "Yesterday tested the plugin",
                               source="session"),
                store.remember("working", "Context: Aurora project",
                               source="session", session="first"),
                store.remember("relational", "Penguins are linked to birds",
                               source="user"),
            ]
            self.assertEqual(set(store.summary()["counts"]),
                             {"semantic", "procedural", "episodic",
                              "working", "relational"})
            self.assertEqual(store.get(ids[0]).content, "A penguin is a bird")
            with self.assertRaises(ValueError):
                store.remember("semantic", "False claim",
                               source="user", verified=True)
            self.assertEqual(store.recall("Aurora", session="second"), [])
            self.assertEqual(len(store.recall("Aurora", session="first")), 1)
            store.relate(ids[0], ids[4], "related_to")
            found = store.recall("penguin", limit=2)
            self.assertEqual({x.id for x in found}, {ids[0], ids[4]})
            store.relate(ids[0], ids[1], "depends_on")
            self.assertIn(ids[1], {x.id for x in
                                   store.recall("penguin", limit=4)})
            with self.assertRaises(ValueError):
                store.relate(ids[0], "../bad", "related_to")

    def test_sqlite_persists_and_expires(self):
        with TemporaryDirectory() as tmp:
            file = Path(tmp) / "memory.sqlite"
            with MemoryFabric(file, writable=True) as mem:
                identifier = mem.remember(
                    "procedural", "Test the package", source="user")
            with MemoryFabric(file) as readonly:
                self.assertEqual(readonly.get(identifier).kind, "procedural")
                with self.assertRaises(PermissionError):
                    readonly.relate(identifier, "none", "related_to")


class UnifiedRuntimeTests(unittest.TestCase):
    def test_extension_contributes_to_one_unified_prefix(self):
        model = DummyModel()
        runtime = UnifiedCognitiveMechanism(
            model, extensions=(scalar_extension(),))
        emitted = bytes(runtime.generate(prefix("scalar increment 7 by 2")))
        self.assertEqual(emitted, b"Okay.")
        encoded = bytes(x for x in model.prefix if 0 <= x < 256)
        self.assertIn(b"Native scalar computation", encoded)
        self.assertIn(b"=> 9", encoded)
        self.assertEqual(runtime.last_receipt["native_extensions"],
                         ["scalar.native"])
        self.assertFalse(runtime.last_receipt["verified_external_truth"])

    def test_memory_recall_and_opt_in_two_way_write(self):
        model = DummyModel()
        with MemoryFabric(":memory:", writable=True) as memory:
            memory.remember("semantic", "Cats are mammals", source="reviewed-user")
            runtime = UnifiedCognitiveMechanism(
                model, memory=memory, persist_episodes=True)
            result = bytes(runtime.generate(prefix("Tell me about cats")))
            self.assertEqual(result, b"Okay.")
            text = bytes(x for x in model.prefix if 0 <= x < 256)
            self.assertIn(b"Cats are mammals", text)
            self.assertTrue(runtime.last_receipt["episodic_memory_committed"])
            self.assertEqual(memory.summary()["counts"]["episodic"], 1)
            key = runtime.learn("procedural", "Check evidence before writing",
                                source="operator")
            self.assertEqual(memory.get(key).kind, "procedural")

    def test_no_unreviewed_memory_writes_or_unsafe_extensions(self):
        with MemoryFabric(":memory:") as memory:
            runtime = UnifiedCognitiveMechanism(
                DummyModel(), memory=memory, persist_episodes=True)
            with self.assertRaises(PermissionError):
                list(runtime.generate(prefix("hello")))
        with self.assertRaises(ValueError):
            UnifiedCognitiveMechanism(
                DummyModel(), extensions=(
                    NativeExtension("duplicate.ext", lambda _: True,
                                    lambda _f: "a"),
                    NativeExtension("duplicate.ext", lambda _: True,
                                    lambda _f: "b")))
        runtime = UnifiedCognitiveMechanism(DummyModel())
        with self.assertRaises(ValueError):
            list(runtime.generate([BOS, *b"not chat", ASSISTANT]))

    def test_context_limit_discards_optional_notes_not_user_content(self):
        model = DummyModel()
        model.config = type("SmallConfig", (), {
            "max_context": 40, "max_answer": 32})()
        runtime = UnifiedCognitiveMechanism(
            model, extensions=(NativeExtension(
                "long.ext", lambda _: True, lambda _: "extra" * 50),))
        list(runtime.generate(prefix("hello")))
        self.assertLessEqual(len(model.prefix), 40)
        self.assertEqual(runtime.last_receipt["notes_accepted"], 0)
        with self.assertRaisesRegex(ValueError, "context"):
            list(runtime.generate(prefix("x" * 100)))

    def test_background_l2_extensions_run_concurrently_with_ordered_results(self):
        from threading import Barrier
        rendezvous = Barrier(2, timeout=3)
        def first(_frame):
            rendezvous.wait()
            return "first native result"
        def second(_frame):
            rendezvous.wait()
            return "second native result"
        model = DummyModel()
        runtime = UnifiedCognitiveMechanism(
            model,
            extensions=(
                NativeExtension("first.ext", lambda _: True, first),
                NativeExtension("second.ext", lambda _: True, second),
            ))
        self.assertEqual(bytes(runtime.generate(prefix("check"))), b"Okay.")
        self.assertEqual(runtime.last_receipt["background_levels"]
                         ["L2_direct_native_extensions"],
                         ["first.ext", "second.ext"])
        self.assertFalse(runtime.last_receipt["persistent_background_daemon"])
        encoded = bytes(x for x in model.prefix if 0 <= x < 256)
        self.assertLess(encoded.index(b"first native"), encoded.index(b"second native"))

    def test_readonly_workspace_adapter_respects_workbench_guards(self):
        from ntsa_x.agent_workbench import Workbench
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "README.md").write_text("unique valid architecture", encoding="utf-8")
            wb = Workbench(root)
            runtime = UnifiedCognitiveMechanism(
                DummyModel(), extensions=(workspace_extension(wb),))
            self.assertEqual(
                bytes(runtime.generate(prefix("search workspace for architecture"))),
                b"Okay.")
            model_text = bytes(x for x in runtime.model.prefix if 0 <= x < 256)
            self.assertIn(b"README.md", model_text)
            with self.assertRaises(ValueError):
                list(runtime.generate(prefix("inspect file ../secret.txt")))


if __name__ == "__main__":
    unittest.main()
