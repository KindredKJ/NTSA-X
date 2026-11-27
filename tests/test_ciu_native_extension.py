"""Exercise the real native-extension host with a deterministic test model."""
import unittest
from types import SimpleNamespace

from ntsa_x.capability_ciu import BinaryTask, CapabilityRuntime, ciu_extension
from ntsa_x.convergence import Check, ConvergenceEngine, REQUIRED_CHECKS
from ntsa_x.nexus import NexusStore
from ntsa_x.unified_cognition import UnifiedCognitiveMechanism, BOS, USER, EOS, ASSISTANT


class TestModel:
    config = SimpleNamespace(max_context=2048)

    def unified_draft(self, prefix, **kwargs):
        self.prefix = prefix
        return list(b"test model output")


class NativeCIUTests(unittest.TestCase):
    def test_cognition_contributes_acquired_then_recalled_rule(self):
        report = ConvergenceEngine().evaluate([
            Check(c, True, "synthetic fixture", external=c == "evidence")
            for c in REQUIRED_CHECKS
        ])
        task = BinaryTask("cubic", "v1", 8, lambda s: s ^ (((s & 7) == 7) << 7), report)
        with NexusStore(":memory:") as store:
            runtime = CapabilityRuntime(store)
            extension = ciu_extension(runtime, (task,), acquire=True)
            self.assertFalse(extension.supports("ciu unknown 7"))
            self.assertFalse(extension.supports("ciu cubic 256"))
            self.assertFalse(extension.supports("please run ciu cubic 7"))
            model = TestModel()
            host = UnifiedCognitiveMechanism(model, extensions=(extension,))
            for query, result in [
                ("ciu cubic 7", b"CIU cubic: 135; source=acquired; oracle_calls=512"),
                ("ciu cubic 0x0f", b"CIU cubic: 143; source=recalled; oracle_calls=0"),
            ]:
                prompt = [BOS, USER, *query.encode(), EOS, ASSISTANT]
                self.assertEqual(bytes(host.generate(prompt)), b"test model output")
                context = bytes(x for x in model.prefix if 0 <= x <= 255)
                self.assertIn(result, context)
                self.assertEqual(host.last_receipt["native_extensions"], ["ciu.native"])
                self.assertFalse(host.last_receipt["verified_external_truth"])
