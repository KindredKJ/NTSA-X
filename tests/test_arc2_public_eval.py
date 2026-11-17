"""Behavior and blinded-evaluation checks on synthetic fixtures only."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from ntsa_x.arc2_public_eval import transform, learn_mapping, predict, grid_ok, evaluate


class ARC2PolicyTests(unittest.TestCase):
    def test_geometry(self):
        inp = [[1,2,3],[4,5,6]]
        self.assertEqual(transform(inp, "rotate90"), [[4,1],[5,2],[6,3]])
        self.assertEqual(transform(inp, "rotate270"), [[3,6],[2,5],[1,4]])
        self.assertEqual(transform(inp, "flip_h"), [[3,2,1],[6,5,4]])
        self.assertEqual(transform(inp, "rot180"), [[6,5,4],[3,2,1]])

    def test_recolor_from_multiple_examples_without_test_outputs(self):
        train = [
            {"input": [[1,2]], "output": [[3,4]]},
            {"input": [[2,1]], "output": [[4,3]]},
        ]
        got = predict(train, [[1,2],[2,1]])
        self.assertEqual(got[0], [[3,4],[4,3]])
        self.assertEqual(len(got), 2)
        self.assertNotEqual(predict([], [[1,2],[2,1]], baseline=True)[0], got[0])

    def test_conflicting_mapping_rejected(self):
        train = [
            {"input": [[1]], "output": [[2]]},
            {"input": [[1]], "output": [[3]]},
        ]
        self.assertIsNone(learn_mapping(train, "identity"))
        self.assertTrue(grid_ok([[0,1],[2,9]]))
        self.assertFalse(grid_ok([[10]]))

    def test_exactly_two_guesses_without_mutating_input(self):
        original = [[1,2],[3,4]]
        pair = {"input": [[1,2]], "output": [[2,1]]}
        result = predict([pair], original)
        self.assertEqual(len(result), 2)
        self.assertEqual(original, [[1,2],[3,4]])

    def test_full_evaluation_refuses_non_official_size(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            sample = {"train": [{"input": [[1]], "output": [[2]]}],
                      "test": [{"input": [[1]], "output": [[2]]}]}
            (folder/"sample.json").write_text(json.dumps(sample))
            with self.assertRaisesRegex(ValueError, "120-task"):
                evaluate(folder)

    def test_test_answer_never_reaches_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            for i in range(120):
                sample = {"train": [{"input": [[1,2]], "output": [[2,1]]}],
                          "test": [{"input": [[1,2]], "output": [[2,1]]}]}
                (folder/f"{i:08x}.json").write_text(json.dumps(sample))
            original = predict
            calls = []
            def wrapped(train, inp, baseline=False):
                calls.append((train, inp, baseline))
                self.assertNotIn("output", inp if isinstance(inp, dict) else {})
                return original(train, inp, baseline=baseline)
            with patch("ntsa_x.arc2_public_eval.predict", wrapped):
                result = evaluate(folder)
            self.assertEqual(len(calls), 240)
            self.assertEqual(result["task_count"], 120)
            self.assertTrue(result["no_hidden_answer_use"])
            self.assertFalse(result["official_private_score"])
            self.assertEqual(result["primary_mean_fractional_task_score"], 1)


if __name__ == "__main__":
    unittest.main()
