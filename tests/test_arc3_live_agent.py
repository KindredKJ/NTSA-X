"""Unit tests for ARC-AGI-3 policy bookkeeping. They are not external game scores."""
import random
import unittest
from types import SimpleNamespace
from ntsa_x.arc3_live_agent import NoveltyPolicy


class FakeAction:
    def __init__(self, name):
        self.name = name

    def is_complex(self):
        return False


class ARC3PolicyTests(unittest.TestCase):
    def test_memory_updates_on_observed_frame_changes(self):
        p = NoveltyPolicy(99)
        actions = [FakeAction("ACTION1"), FakeAction("ACTION2")]
        first = p.pick(actions)
        self.assertEqual(first.name, "ACTION1")
        self.assertTrue(p.record(first, SimpleNamespace(frame=[[[0, 0], [0, 1]]])))
        second = p.pick(actions)
        self.assertEqual(second.name, "ACTION2")
        self.assertFalse(p.record(second, SimpleNamespace(frame=[[[0, 0], [0, 1]]])))
        self.assertEqual(p.steps, 2)
        self.assertEqual(len(p.visited), 1)
        self.assertEqual(p.gain["ACTION1"], 1)
        self.assertEqual(p.gain["ACTION2"], 0)
        self.assertEqual(len(p.chain), 64)

    def test_same_seed_evidence_produces_same_sequence(self):
        actions = [FakeAction("ACTION1"), FakeAction("ACTION2"), FakeAction("ACTION3")]
        def experiment():
            p = NoveltyPolicy(17)
            sequence = []
            for i in range(15):
                a = p.pick(actions)
                sequence.append(a.name)
                p.record(a, SimpleNamespace(frame=[[[i % 4]]]))
            return sequence, p.chain
        self.assertEqual(experiment(), experiment())

    def test_invalid_or_missing_input(self):
        p = NoveltyPolicy(1)
        with self.assertRaises(ValueError):
            p.pick([])
        self.assertFalse(p.record(FakeAction("ACTION1"), None))
        self.assertEqual(p.steps, 1)
        self.assertEqual(len(p.visited), 0)


if __name__ == "__main__":
    unittest.main()
