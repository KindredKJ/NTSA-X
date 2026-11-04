"""Pure-Python research checks; synthetic rewards never prove real AGI."""
import json
import random
import unittest

from ntsa_x.interventional_memory import (
    InterventionalBeliefMemory, SyntheticContextualWorld, benchmark,
)


class InterventionalMemoryTests(unittest.TestCase):
    def test_observations_update_posterior_and_hash_link_trace(self):
        model = InterventionalBeliefMemory(2, 3)
        self.assertAlmostEqual(model.predict(1, 0).posterior_mean, 0.5)
        first = model.observe(1, 0, 1)
        self.assertEqual(len(first), 64)
        self.assertAlmostEqual(model.predict(1, 0).posterior_mean, 2 / 3)
        second = model.observe(1, 0, 0)
        self.assertNotEqual(first, second)
        self.assertAlmostEqual(model.predict(1, 0).posterior_mean, 0.5)
        self.assertEqual(model.predict(1, 0).observations, 2)
        self.assertEqual(model.events, 2)
        self.assertEqual(model.trace_sha256, second)

    def test_distinct_actions_have_distinct_beliefs(self):
        model = InterventionalBeliefMemory(2, 2)
        for _ in range(12):
            model.observe(0, 1, 1)
        self.assertGreater(model.predict(0, 1).posterior_mean, 0.9)
        self.assertEqual(model.predict(0, 0).posterior_mean, 0.5)
        self.assertEqual(model.predict(1, 1).posterior_mean, 0.5)

    def test_rng_choices_repeat_exactly_for_equal_seed_and_evidence(self):
        a, b = InterventionalBeliefMemory(3, 3), InterventionalBeliefMemory(3, 3)
        for model in (a, b):
            model.observe(1, 1, 1)
            model.observe(1, 0, 0)
        rng1, rng2 = random.Random(88), random.Random(88)
        self.assertEqual([a.choose(1, rng1) for _ in range(100)],
                         [b.choose(1, rng2) for _ in range(100)])

    def test_hidden_reward_stream_is_reproducible_and_action_specific(self):
        a = SyntheticContextualWorld(6, 4, 3)
        b = SyntheticContextualWorld(6, 4, 3)
        self.assertEqual(a.hidden_task_hash(), b.hidden_task_hash())
        self.assertEqual(
            [a.intervention(1, 2, i) for i in range(50)],
            [b.intervention(1, 2, i) for i in range(50)])
        self.assertNotEqual(
            a.hidden_task_hash(), SyntheticContextualWorld(7, 4, 3).hidden_task_hash())

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            InterventionalBeliefMemory(0, 2)
        with self.assertRaises(ValueError):
            InterventionalBeliefMemory(2, 1)
        memory = InterventionalBeliefMemory(2, 2)
        for context, action, reward in [(-1, 0, 1), (0, 3, 1),
                                         (0, 0, -1), (0, 1, True)]:
            with self.assertRaises(ValueError):
                memory.observe(context, action, reward)
        with self.assertRaises(TypeError):
            memory.choose(0, "unseeded")
        with self.assertRaises(ValueError):
            benchmark(seeds=[0, 0])
        with self.assertRaises(ValueError):
            benchmark(steps=10)
        with self.assertRaises(ValueError):
            benchmark(warmup=800)

    def test_benchmark_is_deterministic_and_labels_synthetic_truthfully(self):
        x = benchmark(range(4), steps=300, warmup=100)
        y = benchmark(range(4), steps=300, warmup=100)
        self.assertEqual(json.dumps(x, sort_keys=True), json.dumps(y, sort_keys=True))
        self.assertFalse(x["is_external_benchmark"])
        self.assertFalse(x["is_agi_proof"])
        self.assertFalse(x["is_frontier_rank_one_proof"])
        self.assertEqual(x["verification_status"], "INTERNAL_SYNTHETIC_RESEARCH_ONLY")
        self.assertEqual(len(x["runs"]), 4)
        self.assertEqual(x["protocol"]["steps"], 300)

    def test_adaptive_memory_competes_against_random_on_multiple_seeds(self):
        report = benchmark(range(12), steps=800, warmup=300)
        self.assertGreater(report["learned_mean_success_rate"],
                           report["random_mean_success_rate"])
        self.assertLess(report["learned_mean_success_rate"], 1)
        self.assertEqual(report["protocol"]["identical_context_schedule"], True)


if __name__ == "__main__":
    unittest.main()
