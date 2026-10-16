"""Milestone 13: compact, auditable original lessons and deterministic exams.

Curriculum is NOT a pretrained model, a proof of generalization, or a
license to copy textbooks. It creates reviewed original micro-lessons,
train-only supervised conversations and a concept-disjoint exam file.
No dependencies, network, or model calls; no checkpoint writes.
"""
from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import json
import re


@dataclass(frozen=True)
class Lesson:
    id: str
    domain: str
    prerequisite: tuple[str, ...]
    concept: str
    explanation: str
    worked: tuple[str, ...]
    exercises: tuple[tuple[str, str], ...]
    exam: tuple[tuple[str, str], ...]


LESSONS = (
    Lesson("lang.words", "language", (), "Words and meaning",
           "A noun names a person, place, thing, or idea. A verb expresses an action or state. Name the function of a word in its sentence.",
           ("In 'Birds fly', 'Birds' is a noun; 'fly' is a verb.",),
           (("In 'Dogs run', what part of speech is 'Dogs'?", "noun"),
            ("In 'Cats sleep', what part of speech is 'sleep'?", "verb")),
           (("In 'Trees grow', what part of speech is 'Trees'?", "noun"),)),
    Lesson("lang.sentences", "language", ("lang.words",), "Sentence meaning",
           "The subject identifies whom or what a statement concerns. A predicate describes what the subject does or is. Read each clause before answering.",
           ("In 'The engine starts', subject = 'The engine'; predicate = 'starts'.",),
           (("Identify the subject in 'The robot moves'.", "The robot"),
            ("Identify the predicate in 'The server responds'.", "responds")),
           (("Identify the subject in 'The computer works'.", "The computer"),)),
    Lesson("lang.inference", "language", ("lang.sentences",), "Entailment",
           "Only conclude what the stated premises support. A conclusion requiring unstated facts is not entailed.",
           ("All sparrows are birds. Pip is a sparrow. Therefore Pip is a bird.",),
           (("All roses are plants. Rose A is a rose. Is Rose A a plant? Answer yes or no.", "yes"),
            ("All squares are shapes. Shape B is a shape. Must B be a square? Answer yes or no.", "no")),
           (("All whales are mammals. W is a whale. Is W a mammal? Answer yes or no.", "yes"),)),
    Lesson("lang.instructions", "language", ("lang.inference",), "Follow explicit instructions",
           "Identify the exact requested output format and content. Do not add facts not supported by the instruction.",
           ("Instruction: 'Write only BLUE in uppercase.' Correct output: BLUE.",),
           (("Write only GREEN in uppercase.", "GREEN"),
            ("Write only YES in uppercase.", "YES")),
           (("Write only TRUE in uppercase.", "TRUE"),)),
    Lesson("math.operators", "math", (), "Arithmetic operations",
           "Addition combines quantities; subtraction finds a difference. Solve by checking the inverse operation.",
           ("7 + 5 = 12; check 12 - 5 = 7.",),
           (("Calculate 8 + 6. Respond with the number only.", "14"),
            ("Calculate 17 - 9. Respond with the number only.", "8")),
           (("Calculate 13 + 9. Respond with the number only.", "22"),)),
    Lesson("math.fractions", "math", ("math.operators",), "Fractions",
           "To add fractions with equal denominators, add numerators and retain the denominator; simplify the result.",
           ("1/4 + 2/4 = 3/4.",),
           (("Calculate 1/5 + 2/5. Respond as a simplified fraction.", "3/5"),
            ("Calculate 3/8 + 1/8. Respond as a simplified fraction.", "1/2")),
           (("Calculate 2/7 + 3/7. Respond as a simplified fraction.", "5/7"),)),
    Lesson("math.algebra", "math", ("math.fractions",), "One-step equations",
           "An equation remains balanced when the same operation is applied to both sides. Undo an addition using subtraction.",
           ("x + 4 = 11; subtract 4 from both sides, x = 7.",),
           (("Solve x + 6 = 15. Respond with x only.", "9"),
            ("Solve x + 3 = 12. Respond with x only.", "9")),
           (("Solve x + 8 = 21. Respond with x only.", "13"),)),
    Lesson("math.logic", "math", ("math.algebra",), "Logical implication",
           "For implication P implies Q, the only false case is P true and Q false. This is a rule for evaluating truth assignments.",
           ("If P is true and Q is false, P implies Q is false.",),
           (("P=true and Q=true. Is 'P implies Q' true or false?", "true"),
            ("P=false and Q=true. Is 'P implies Q' true or false?", "true")),
           (("P=true and Q=false. Is 'P implies Q' true or false?", "false"),)),
    Lesson("cs.values", "computer_science", (), "Variables and values",
           "An assignment binds a name to a value. After a later assignment to the same name, its value changes in the current environment.",
           ("Python: x = 2; x = 5. The final value of x is 5.",),
           (("Python: a = 3; a = 7. What is a? Number only.", "7"),
            ("Python: y = 9; y = 4. What is y? Number only.", "4")),
           (("Python: z = 12; z = 6. What is z? Number only.", "6"),)),
    Lesson("cs.branching", "computer_science", ("cs.values",), "Branching",
           "An if branch runs only when its condition is true. Else runs when the condition is false.",
           ("Python: if 2 > 1: x = 4 else: x = 9. The resulting x is 4.",),
           (("Python logic: if 3 > 8 then output A else output B. Which letter?", "B"),
            ("Python logic: if 10 > 2 then output X else output Y. Which letter?", "X")),
           (("Python logic: if 5 > 11 then output M else output N. Which letter?", "N"),)),
    Lesson("cs.functions", "computer_science", ("cs.branching",), "Functions",
           "A function receives inputs and returns a value according to its body. Evaluate the body with the supplied arguments.",
           ("For def double(x): return x * 2, double(3) returns 6.",),
           (("Given def triple(x): return x * 3, what is triple(4)? Number only.", "12"),
            ("Given def inc(x): return x + 1, what is inc(8)? Number only.", "9")),
           (("Given def double(x): return x * 2, what is double(7)? Number only.", "14"),)),
    Lesson("cs.testing", "computer_science", ("cs.functions",), "Software tests",
           "A test compares observed behavior against an expected result. Passing one test cannot establish correctness for every possible input.",
           ("assert double(3) == 6 tests one input, not every input.",),
           (("Does passing a single unit test prove a program correct for all inputs? Answer yes or no.", "no"),
            ("If a test expects 3 but gets 4, did that test pass? Answer yes or no.", "no")),
           (("Does a passing test guarantee a program has zero bugs? Answer yes or no.", "no"),)),
)


def normalize_answer(text: str) -> str:
    if not isinstance(text, str):
        raise ValueError("Answer must be text")
    return " ".join(text.strip().casefold().split())


def audit_lessons() -> None:
    ids = set()
    seen_prompts = set()
    for l in LESSONS:
        if l.id in ids:
            raise ValueError("Duplicate lesson identifier")
        if any(p not in ids for p in l.prerequisite):
            raise ValueError("Unmet or forward prerequisite")
        if not l.explanation or not l.worked or not l.exercises or not l.exam:
            raise ValueError("Lesson lacks instruction or examination")
        ids.add(l.id)
        for q, a in (*l.exercises, *l.exam):
            key = normalize_answer(q)
            if key in seen_prompts or not a.strip():
                raise ValueError("Duplicate question or empty gold answer")
            seen_prompts.add(key)


def lesson_passed(lesson_id: str, answers: list[str], threshold: float = 1.0) -> dict:
    audit_lessons()
    lesson = next((l for l in LESSONS if l.id == lesson_id), None)
    if lesson is None or not isinstance(answers, list) or not 0 < threshold <= 1:
        raise ValueError("Invalid lesson, answers or threshold")
    if len(answers) != len(lesson.exam):
        raise ValueError("Exact exam response count required")
    matches = [normalize_answer(gold) == normalize_answer(answer)
               for (_, gold), answer in zip(lesson.exam, answers)]
    score = sum(matches) / len(matches)
    return {"lesson": lesson_id, "score": score,
            "passed": score >= threshold, "individual_correct": matches,
            "grading": "exact_normalized_match_only",
            "limitation": "Exact match cannot reliably assess free-form reasoning"}


def next_lessons(passed: set[str]) -> list[str]:
    audit_lessons()
    valid = {l.id for l in LESSONS}
    if not isinstance(passed, set) or not passed <= valid:
        raise ValueError("Unknown mastered lesson")
    return [l.id for l in LESSONS
            if l.id not in passed and set(l.prerequisite) <= passed]


def curriculum_report() -> dict:
    audit_lessons()
    return {
        "curriculum": "kindred-academy-m13-v1",
        "domains": sorted({l.domain for l in LESSONS}),
        "lessons": len(LESSONS),
        "training_examples": sum(len(l.exercises) for l in LESSONS),
        "reserved_exam_examples": sum(len(l.exam) for l in LESSONS),
        "ready_first": next_lessons(set()),
        "progression": "all_prerequisites_mastered_and_exam_passed",
        "source": "Original illustrative micro-lessons authored for this repository",
        "rights": "Original authored text; review before commercial training",
        "limitations": "Tiny demonstration curriculum. No textbook ingestion, model training, "
                       "external evaluation, or generalization proven.",
    }


def export(outdir: str | Path) -> dict:
    audit_lessons()
    destination = Path(outdir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError("Refuse to overwrite existing curriculum export")
    if not destination.parent.is_dir():
        raise ValueError("Create parent directory first")
    # Full validation occurs before creating files.
    training, exam = [], []
    for l in LESSONS:
        for question, answer in l.exercises:
            instruction = (f"Subject: {l.domain}. Concept: {l.concept}. "
                           f"Explanation: {l.explanation} "
                           f"Worked example: {l.worked[0]} "
                           f"Exercise: {question}")
            training.append({
                "messages": [
                    {"role": "user", "content": instruction},
                    {"role": "assistant", "content": answer}],
                "lesson_id": l.id,
            })
        for question, gold in l.exam:
            exam.append({
                "lesson_id": l.id, "domain": l.domain,
                "question": question, "gold": gold,
            })
    # Prevent exact question overlap and inadvertent exam leakage.
    train_questions = {
        normalize_answer(q)
        for l in LESSONS for q, _ in l.exercises
    }
    if any(normalize_answer(x["question"]) in train_questions for x in exam):
        raise ValueError("Exam question seen during practice")
    destination.mkdir()
    manifest = curriculum_report()
    (destination / "train-chat.jsonl").write_text(
        "".join(json.dumps({"messages": x["messages"]}, ensure_ascii=False)
                + "\n" for x in training), encoding="utf-8")
    (destination / "exam-reserved.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in exam),
        encoding="utf-8")
    (destination / "curriculum.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    hashes = {p.name: sha256(p.read_bytes()).hexdigest()
              for p in destination.iterdir() if p.is_file()}
    return {"output": str(destination), "files_sha256": hashes,
            **manifest}


def main() -> None:
    parser = ArgumentParser(description="Kindred Academy accelerated curriculum; no model training")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--export", metavar="NEW_DIRECTORY")
    parser.add_argument("--grade-lesson")
    parser.add_argument("--answers-json", help="JSON array of answers for the specified lesson")
    args = parser.parse_args()
    try:
        if args.export:
            result = export(args.export)
        elif args.grade_lesson:
            if not args.answers_json:
                raise ValueError("Grading requires --answers-json")
            result = lesson_passed(args.grade_lesson, json.loads(args.answers_json))
        elif args.report:
            result = curriculum_report()
        else:
            parser.error("Choose --report, --export or --grade-lesson")
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
