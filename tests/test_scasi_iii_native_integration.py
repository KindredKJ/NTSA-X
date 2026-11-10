"""Real NTSA-X SQLite memory + UnifiedCognitiveMechanism smoke integration."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from ntsa_x.memory_fabric import MemoryFabric
from ntsa_x.unified_cognition import BOS, USER, EOS, ASSISTANT
from ntsa_x.scasi_iii import Objective
from ntsa_x.scasi_iii_evolution import PredictiveKnowledgeCoordinator

class DeterministicModel:
    config = SimpleNamespace(max_context=8192)
    architecture = "test-model"
    def unified_draft(self, augmented, *, max_new_tokens, temperature):
        return list(b"SCASI model hook reached")[:max_new_tokens]

class NativeCognitionIntegration(unittest.TestCase):
    def test_native_memory_and_cognitive_hook(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "test.sqlite"
            with MemoryFabric(path, writable=True) as memory:
                memory.remember("semantic", "release manifest reviewed",
                                source="fixture", session="scasi")
                memory.remember("procedural", "rollback plan is drafted",
                                source="fixture", session="scasi")
                coordinator = PredictiveKnowledgeCoordinator((
                    Objective("deploy", ("deploy service",),
                              ("release manifest", "rollback plan")),),
                    memory=memory, session="scasi")
                self.assertEqual(coordinator.anticipate(max_objectives=1)["warmed"], ("deploy",))
                runtime = coordinator.attach_to_model(DeterministicModel())
                prefix = [BOS, USER, *b"deploy service", EOS, ASSISTANT]
                self.assertIn("SCASI model hook reached",
                              bytes(runtime.generate(prefix, max_new_tokens=32)).decode())
                self.assertIn("scasi3.predictive", runtime.last_receipt["native_extensions"])
                self.assertFalse(runtime.last_receipt["episodic_memory_committed"])
                result = coordinator.resolve("deploy service")
                self.assertTrue(result["forecasts"][0]["prepared_before_request"])
                self.assertTrue(all(g["state"] == "unverified"
                                    for g in result["forecasts"][0]["knowledge_gaps"]))
            with MemoryFabric(path, writable=False) as persisted:
                self.assertTrue(persisted.recall("rollback plan", session="scasi"))

if __name__ == "__main__":
    unittest.main()
