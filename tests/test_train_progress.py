"""Measured training progress remains optional, finite and non-destructive."""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from ntsa_x.train_progress import open_progress, emit_progress


class TestNativeProgress(unittest.TestCase):
    def test_exact_completed_step_and_epoch_events(self):
        with TemporaryDirectory() as root:
            target = Path(root) / "events.jsonl"
            path = open_progress(target, total_steps=10)
            for step in range(1, 11):
                emit_progress(path, "optimizer_step", step=step,
                              total_steps=10, epoch=1, every=3)
            emit_progress(path, "epoch_complete", step=10,
                          total_steps=10, epoch=1, validation_metric=0.42)
            messages = [json.loads(s) for s in target.read_text().splitlines()]
            self.assertEqual([m["step"] for m in messages], [0, 3, 6, 9, 10, 10])
            self.assertEqual(messages[-1]["validation_metric"], 0.42)
            self.assertEqual(messages[0]["type"], "train_started")
            self.assertEqual(messages[-1]["type"], "epoch_complete")

    def test_nonfinite_epoch_metric_fails(self):
        with TemporaryDirectory() as root:
            path = open_progress(Path(root) / "events.jsonl", total_steps=2)
            with self.assertRaises(ValueError):
                emit_progress(path, "epoch_complete", step=2, total_steps=2,
                              epoch=1, validation_metric=float("inf"))

    def test_exclusive_artifact_and_disabled_telemetry(self):
        with TemporaryDirectory() as root:
            target = Path(root) / "events.jsonl"
            open_progress(target, total_steps=1)
            with self.assertRaises(FileExistsError):
                open_progress(target, total_steps=1)
            emit_progress(None, "optimizer_step", step=1, total_steps=1, epoch=1)
            self.assertEqual(len(target.read_text().splitlines()), 1)


if __name__ == "__main__":
    unittest.main()
