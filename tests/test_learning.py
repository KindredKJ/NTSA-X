"""Teacher-to-student approval and recall regression tests."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
import json

from ntsa_x.learning import LearningCorpus, digest
from ntsa_x.live_gateway import GatewayError, LiveService, TurnStore


class InstrumentedTeacher:
    model = "test-local-teacher-v1"

    def __init__(self):
        self.calls = 0

    def stream(self, messages):
        self.calls += 1
        yield "A reviewed example"


class LearningTests(TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        root = Path(self.tmp.name)
        self.corpus = LearningCorpus(root / "learning.db")
        self.store = TurnStore(root / "sessions.db")

    def tearDown(self):
        self.tmp.cleanup()

    def test_requires_explicit_review_and_teacher_rights(self):
        sample = self.corpus.capture("session_0001", "What is 2+2?",
                                     "Four.", [], "teacher-v1")
        self.assertIsNone(self.corpus.recall("session_0001", "What is 2+2?", []))
        with self.assertRaises(ValueError):
            self.corpus.approve("session_0001", sample, rights=False)
        self.assertFalse(self.corpus.approve("session_0002", sample, rights=True))
        self.assertTrue(self.corpus.approve("session_0001", sample, rights=True))
        self.assertEqual(self.corpus.recall("session_0001", "What is 2+2?", []),
                         "Four.")
        self.assertIsNone(self.corpus.recall("session_0002", "What is 2+2?", []))
        self.assertIsNone(self.corpus.recall(
            "session_0001", "What is 2+2?", [{"role": "user", "content": "context"}]))
        self.assertIsNone(self.corpus.recall("session_0001", "What is 3+2?", []))

    def test_gateway_uses_shared_approved_recall_before_model(self):
        teacher = InstrumentedTeacher()
        service = LiveService(self.store, teacher, self.corpus)
        self.assertEqual("".join(service.stream(
            "session_0001", "test prompt", capture_for_learning=True)),
            "A reviewed example")
        sample = service.last_result("session_0001")["sample_id"]
        self.assertEqual(teacher.calls, 1)
        self.assertTrue(self.corpus.approve(
            "session_0001", sample, rights=True, share=True))
        self.assertEqual("".join(service.stream(
            "session_0002", "test prompt")), "A reviewed example")
        self.assertEqual(teacher.calls, 1)
        self.assertEqual(service.last_result("session_0002")["source"],
                         "approved-recall")
        self.assertEqual(service.last_result("session_0002")["sample_id"], None)
        self.assertEqual([t["role"] for t in self.store.history("session_0002")],
                         ["user", "assistant"])
        self.corpus.clear("session_0001")
        self.assertIsNone(self.corpus.recall("session_0002", "test prompt", []))

    def test_declined_learning_captures_nothing(self):
        teacher = InstrumentedTeacher()
        service = LiveService(self.store, teacher, self.corpus)
        list(service.stream("session_0001", "test prompt"))
        self.assertIsNone(service.last_result("session_0001")["sample_id"])
        with self.corpus.connect() as conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0], 0)
        with self.assertRaises(GatewayError):
            list(service.stream("session_0001", "test prompt",
                                capture_for_learning="yes"))

    def test_student_mode_makes_zero_teacher_calls_and_never_captures_student(self):
        teacher = InstrumentedTeacher()
        student = InstrumentedTeacher()
        student.model = "independent-student"
        service = LiveService(self.store, teacher, self.corpus,
                              student=student, use_student=True)
        self.assertEqual("".join(service.stream(
            "session_0001", "independent", capture_for_learning=True)),
            "A reviewed example")
        self.assertEqual(teacher.calls, 0)
        self.assertEqual(student.calls, 1)
        self.assertEqual(service.last_result("session_0001")["source"], "student")
        self.assertIsNone(service.last_result("session_0001")["sample_id"])
        with self.corpus.connect() as conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0], 0)
        with self.assertRaises(ValueError):
            LiveService(self.store, None, self.corpus, use_student=True)

    def test_human_review_cannot_change_or_overwrite_sample(self):
        sample = self.corpus.capture("session_0001", "prompt", "answer", [], "teacher")
        self.assertTrue(self.corpus.approve("session_0001", sample, True))
        self.assertFalse(self.corpus.approve("session_0001", sample, True))
        with self.corpus.connect() as conn:
            answer = conn.execute("SELECT answer FROM samples WHERE id=?", (sample,)).fetchone()[0]
        self.assertEqual(answer, "answer")

    def test_train_eval_split_prompt_disjoint_and_rights_checked(self):
        for i in range(35):
            prompt = "prompt-" + str(i)
            sample = self.corpus.capture(
                "session_0001", prompt, "response-" + str(i), [],
                "teacher-v1")
            self.assertTrue(self.corpus.approve("session_0001", sample, True))
        # This entry has NOT been approved, so it must never enter either split.
        self.corpus.capture("session_0001", "unapproved", "not included", [], "teacher-v1")
        root = Path(self.tmp.name)
        report = self.corpus.export(root / "train.jsonl", root / "eval.jsonl")
        self.assertEqual(report["training_examples"] + report["held_out_examples"], 35)
        train = [json.loads(line) for line in (root / "train.jsonl").read_text().splitlines()]
        evaluation = [json.loads(line) for line in (root / "eval.jsonl").read_text().splitlines()]
        training_prompts = {item["messages"][-2]["content"] for item in train}
        heldout_prompts = {item["messages"][-2]["content"] for item in evaluation}
        self.assertFalse(training_prompts & heldout_prompts)
        self.assertNotIn("unapproved", training_prompts | heldout_prompts)
        self.assertTrue(all(int(digest(p)[:8], 16) % 5 == 0 for p in heldout_prompts))
        with self.assertRaises(ValueError):
            self.corpus.export(root / "train.jsonl", root / "eval.jsonl")


if __name__ == "__main__":
    import unittest
    unittest.main()
