import random
import tempfile
import unittest
from pathlib import Path

from ntsa_x.convergence import Check, ConvergenceEngine, REQUIRED_CHECKS
from ntsa_x.nexus import NexusStore
from ntsa_x.xor_ciu import NexusXorCIU, XorTransition


def fixture_report(verified=True):
    # Synthetic unit-test evidence only, never a real-world attestation.
    return ConvergenceEngine().evaluate([
        Check(category, True, "unit-test fixture",
              external=verified and category == "evidence")
        for category in REQUIRED_CHECKS
    ])


class XorCIUTests(unittest.TestCase):
    def test_exhaustive_reversal_and_composition(self):
        a = XorTransition.compose(8, [0xA6, 0x11])
        b = XorTransition.compose(8, [0x3D, 0x69])
        composite = a.then(b)
        self.assertEqual(composite.count, 4)
        for state in range(256):
            self.assertEqual(composite.apply(state), b.apply(a.apply(state)))
            self.assertEqual(composite.apply(composite.apply(state)), state)

    def test_million_operations(self):
        rng = random.Random(92)
        sequential = 0xA6
        def masks():
            nonlocal sequential
            for _ in range(1_000_000):
                mask = rng.getrandbits(64)
                sequential ^= mask
                yield mask
        transition = XorTransition.compose(64, masks())
        self.assertEqual(transition.count, 1_000_000)
        self.assertEqual(transition.apply(0xA6), sequential)

    def test_identity_and_invalid_widths(self):
        self.assertEqual(XorTransition.compose(8, []).apply(42), 42)
        self.assertEqual(XorTransition.compose(8, [42, 42]).mask, 0)
        for width in [0, -1, True, 1_048_577]:
            with self.assertRaises(ValueError):
                XorTransition.compose(width, [])
        for invalid in [256, -1, True]:
            with self.assertRaises(ValueError):
                XorTransition.compose(8, [invalid])
        with self.assertRaises(ValueError):
            XorTransition.compose(8, []).then(XorTransition.compose(16, []))

    def test_persistent_scoped_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nexus.db"
            with NexusStore(path) as store:
                adapter = NexusXorCIU(store)
                receipt = adapter.record(
                    width=8, initial=1, terminal=6, masks=[2, 5],
                    context_id="task-a", revision="v1",
                    report=fixture_report(), verify=lambda s: s == 6)
                self.assertTrue(store.verify(receipt))
            with NexusStore(path) as store:
                adapter = NexusXorCIU(store)
                args = dict(initial=1, context_id="task-a", revision="v1",
                            verify=lambda s: s == 6)
                self.assertEqual(adapter.replay(receipt, **args), 6)
                for changes in [dict(initial=2), dict(context_id="task-b"),
                                dict(revision="v2"), dict(verify=lambda s: False)]:
                    self.assertIsNone(adapter.replay(receipt, **(args | changes)))
                self.assertIsNone(adapter.replay("missing", **args))

    def test_unverified_or_wrong_terminal_not_persisted(self):
        with NexusStore(":memory:") as store:
            adapter = NexusXorCIU(store)
            args = dict(width=8, initial=1, terminal=6, masks=[2, 5],
                        context_id="task-a", revision="v1",
                        report=fixture_report(), verify=lambda s: s == 6)
            for changes in [dict(report=fixture_report(False)),
                            dict(terminal=7), dict(verify=lambda s: False)]:
                with self.assertRaises(ValueError):
                    adapter.record(**(args | changes))
            self.assertEqual(store.stats()["nodes"], 0)


    def test_recursive_receipts(self):
        paired = ConvergenceEngine().evaluate([
            Check(category, True, "unit-test fixture",
                  external=category == "evidence")
            for category in REQUIRED_CHECKS
        ], forward={"state": 15}, reverse={"state": 15})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "recursive.db"
            with NexusStore(path) as store:
                ciu = NexusXorCIU(store)
                common = dict(context_id="chain", revision="v1", report=paired)
                a = ciu.record(width=8, initial=0, terminal=3, masks=[1, 2],
                               verify=lambda s: s == 3, **common)
                b = ciu.record(width=8, initial=3, terminal=7, masks=[4],
                               verify=lambda s: s == 7, **common)
                c = ciu.record(width=8, initial=7, terminal=15, masks=[8],
                               verify=lambda s: s == 15, **common)
                ab = ciu.compose_receipts([a, b], initial=0,
                                         verify=lambda s: s == 7, **common)
                abc = ciu.compose_receipts([ab, c], initial=0,
                                          verify=lambda s: s == 15, **common)
                self.assertEqual(store.read(abc).payload["count"], 4)
                self.assertEqual(store.read(abc).parents, (ab, c))
                for members in [[], [b, a], [a, c], ["0" * 64]]:
                    with self.assertRaises(ValueError):
                        ciu.compose_receipts(members, initial=0,
                                             verify=lambda s: True, **common)
                with self.assertRaises(ValueError):
                    ciu.compose_receipts([a, b], initial=0,
                                         verify=lambda s: False, **common)
                with self.assertRaises(ValueError):
                    ciu.compose_receipts([a, b], initial=0,
                                         verify=lambda s: True,
                                         **(common | {"revision": "v2"}))
                with self.assertRaises(ValueError):
                    ciu.compose_receipts([a, b], initial=0,
                                         verify=lambda s: True,
                                         **(common | {"report": fixture_report()}))
            with NexusStore(path) as store:
                self.assertEqual(NexusXorCIU(store).replay(
                    abc, initial=0, context_id="chain", revision="v1",
                    verify=lambda s: s == 15), 15)
                self.assertTrue(store.verify(abc))


if __name__ == "__main__":
    unittest.main()
