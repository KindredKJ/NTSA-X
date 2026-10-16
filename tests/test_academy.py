"""Milestone 13: no-teacher original curriculum, gated mastery and safe export."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
import json

from ntsa_x.academy import (
    LESSONS, audit_lessons, curriculum_report, export,
    lesson_passed, next_lessons,
)


class AcademyTests(TestCase):
    def test_curriculum_complete_and_prerequisites(self):
        audit_lessons()
        report = curriculum_report()
        self.assertEqual(report["lessons"], 12)
        self.assertEqual(report["training_examples"], 24)
        self.assertEqual(report["reserved_exam_examples"], 12)
        self.assertEqual(set(report["ready_first"]),
                         {"lang.words", "math.operators", "cs.values"})
        self.assertEqual(next_lessons({"lang.words"}),
                         ["lang.sentences", "math.operators", "cs.values"])

    def test_mastery_requires_correct_answers(self):
        self.assertTrue(lesson_passed("math.operators", ["22"])["passed"])
        self.assertFalse(lesson_passed("math.operators", ["23"])["passed"])
        self.assertTrue(lesson_passed("lang.words", ["NOUN"])["passed"])
        with self.assertRaisesRegex(ValueError, "count"):
            lesson_passed("lang.words", [])
        with self.assertRaisesRegex(ValueError, "Unknown"):
            next_lessons({"not-a-real-lesson"})

    def test_export_separates_exercise_and_reserved_exam(self):
        with TemporaryDirectory() as tmp:
            target = Path(tmp) / "academy-v1"
            report = export(target)
            train = [json.loads(line) for line in
                     (target / "train-chat.jsonl").read_text().splitlines()]
            test = [json.loads(line) for line in
                    (target / "exam-reserved.jsonl").read_text().splitlines()]
            self.assertEqual(len(train), 24)
            self.assertEqual(len(test), 12)
            self.assertEqual(len(report["files_sha256"]), 3)
            self.assertEqual(set(train[0]), {"messages"})
            self.assertEqual([r["messages"][-1]["role"] for r in train],
                             ["assistant"] * 24)
            self.assertEqual(len(set(x["question"] for x in test)), 12)
            self.assertTrue(
                all("gold" not in json.dumps(x) for x in train))
            with self.assertRaises(FileExistsError):
                export(target)

    def test_no_training_or_network_import(self):
        self.assertTrue(all(l.explanation and l.worked for l in LESSONS))
