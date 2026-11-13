import math
import tempfile
import unittest
from pathlib import Path

from ntsa_x.semantic_memory import SemanticMemory, FusedRecallMemory, _unit_vector


class ToyEmbedder:
    """Synthetic vector fixture; NOT a real text-semantic quality evaluation."""
    model_id = "toy-encoder-test-v1"
    def encode(self, text):
        words = text.casefold()
        if any(s in words for s in ("deploy", "launch", "publish", "website", "site")):
            return (1, 0, 0)
        if any(s in words for s in ("money", "ledger", "account", "payment", "billing")):
            return (0, 1, 0)
        return (0, 0, 1)
    def embed_document(self, text):
        return self.encode(text)
    def embed_query(self, text):
        return self.encode(text)


class Legacy:
    writable = True
    def __init__(self):
        self.values = []
    def remember(self, kind, content, *, source, session="default", ttl_seconds=None):
        self.values.append((kind, content, source, session))
        return "legacy-id"
    def recall(self, query, *, session="default", limit=5, include_associations=True):
        from ntsa_x.semantic_memory import MemoryRecord
        return [MemoryRecord("legacy", "episodic", "User approved an earlier feature",
                             "legacy", session, False, 1.0)]


class SemanticMemoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "mem.sqlite3"
        self.mem = SemanticMemory(self.path, embedder=ToyEmbedder(), writable=True)

    def tearDown(self):
        self.mem.close()
        self.temp.cleanup()

    def test_dense_retrieval_of_different_wording(self):
        website = self.mem.remember("semantic", "The project website can be deployed using its release workflow",
                                    source="user", session="kindred")
        self.mem.remember("semantic", "Finance account ledger can reconcile payments",
                          source="user", session="kindred")
        hits = self.mem.recall_hits("How do I publish the site?", session="kindred")
        self.assertEqual(hits[0].record.id, website)
        self.assertAlmostEqual(hits[0].similarity, 1.0)

    def test_session_scope_isolation(self):
        self.mem.remember("semantic", "Website deployment", source="user", session="A")
        self.assertEqual(self.mem.recall("publish the site", session="B"), [])
        self.assertIsNone(self.mem.get("no-such-id", session="A"))

    def test_persistence_and_dedupe(self):
        ident = self.mem.remember("semantic", "Ledger payment receipts", source="user", session="A")
        self.assertEqual(ident, self.mem.remember("semantic", "Ledger payment receipts", source="user", session="A"))
        self.mem.close()
        self.mem = SemanticMemory(self.path, embedder=ToyEmbedder(), writable=True)
        self.assertEqual(self.mem.get(ident, session="A").content, "Ledger payment receipts")
        self.assertEqual(self.mem.summary(session="A")["count"], 1)

    def test_recall_threshold(self):
        self.mem.remember("semantic", "Deploy website", source="user")
        self.assertEqual(self.mem.recall_hits("Account ledger", min_similarity=0.3), [])

    def test_writing_requires_capability(self):
        other = SemanticMemory(":memory:", embedder=ToyEmbedder(), writable=False)
        try:
            with self.assertRaises(PermissionError):
                other.remember("semantic", "Website release", source="user")
        finally:
            other.close()

    def test_capture_salience(self):
        rejected = self.mem.capture("semantic", "Trivial note", source="user",
                                    importance=.2, durability=.1, confidence=.9)
        self.assertFalse(rejected.stored)
        accepted = self.mem.capture("semantic", "Public site release workflow",
                                    source="user", importance=.9, durability=.8, confidence=.9)
        self.assertTrue(accepted.stored)
        self.assertEqual(self.mem.get(accepted.memory_id).content, "Public site release workflow")

    def test_embedding_validation(self):
        with self.assertRaises(ValueError):
            _unit_vector([0, 0, 0])
        with self.assertRaises(ValueError):
            _unit_vector([math.nan, 2])

    def test_fusion_compatible_and_scoped(self):
        self.mem.remember("semantic", "Genesis website release", source="user", session="kindred")
        fusion = FusedRecallMemory(Legacy(), self.mem)
        recalled = fusion.recall("Launch site", session="kindred")
        self.assertEqual(len(recalled), 2)
        self.assertIn("Genesis website release", recalled[0].content)
        self.assertEqual(fusion.remember("episodic", "test", source="user", session="kindred"), "legacy-id")
        self.assertEqual(fusion.remember("semantic", "Ledger payment", source="user", session="kindred"),
                         self.mem.recall("Money account", session="kindred")[0].id)

    def test_corrupt_vectors_not_silently_accepted(self):
        key = self.mem.remember("semantic", "Website deployment", source="user")
        self.mem._db.execute("UPDATE semantic_memories SET embedding=? WHERE id=?", ("[1.0,2.0]", key))
        with self.assertRaises(ValueError):
            self.mem.recall("Publish site")

    def test_ttl_expiration(self):
        key = self.mem.remember("semantic", "Launch website", source="user", ttl_seconds=1)
        self.mem._db.execute("UPDATE semantic_memories SET expires=? WHERE id=?", (1, key))
        self.assertEqual(self.mem.recall("Publish site"), [])
        self.assertEqual(self.mem.summary()["count"], 0)

    def test_model_specific_index_isolation(self):
        self.mem.remember("semantic", "Launch website", source="user")
        class New(ToyEmbedder):
            model_id = "new-encoder-v2"
        other = SemanticMemory(self.path, embedder=New(), writable=True)
        try:
            self.assertEqual(other.recall("Publish site"), [])
        finally:
            other.close()



class UnifiedIntegrationTests(unittest.TestCase):
    def test_recalled_dense_context_precedes_native_generation(self):
        from ntsa_x.memory_fabric import MemoryFabric
        from ntsa_x.unified_cognition import (
            UnifiedCognitiveMechanism, BOS, USER, EOS, ASSISTANT
        )

        class Config:
            max_context = 2048

        class Model:
            config = Config()
            architecture = "test-model"
            def __init__(self):
                self.last_prefix = None
            def unified_draft(self, prefix, **kwargs):
                self.last_prefix = prefix
                return list(b"Found context")

        primary = MemoryFabric(":memory:", writable=True)
        dense = SemanticMemory(":memory:", embedder=ToyEmbedder(), writable=True)
        try:
            dense.remember(
                "semantic", "Kindred Genesis website release instructions",
                source="user", session="founder"
            )
            fusion = FusedRecallMemory(primary, dense)
            model = Model()
            runtime = UnifiedCognitiveMechanism(
                model, memory=fusion, session="founder", persist_episodes=True
            )
            query = "Please publish the site"
            prompt = [BOS, USER, *query.encode("utf-8"), EOS, ASSISTANT]
            self.assertEqual(bytes(runtime.generate(prompt)), b"Found context")
            serialized = bytes(x for x in model.last_prefix if 0 <= x < 256)
            self.assertIn(b"Kindred Genesis website release instructions", serialized)
            self.assertEqual(runtime.last_receipt["memory_recalled"], 1)
            self.assertTrue(runtime.last_receipt["episodic_memory_committed"])
            self.assertEqual(primary.summary()["counts"]["episodic"], 1)
        finally:
            dense.close()
            primary.close()

if __name__ == "__main__":
    unittest.main()
