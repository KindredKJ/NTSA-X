# Milestone 13 — Kindred Academy

Accelerated, compact, mastery-based curriculum research prototype.

## Subjects
Language: word classes, sentences, entailment and instruction following.
Mathematics: arithmetic, fractions, equations and logic.
Computer science: values, branching, functions and testing.

12 original micro-lessons each include a prerequisite, an explanation, a worked example, two practice questions and one separate reserved exam question. This is a tiny proof of pipeline, not a full curriculum, English fluency, or reasoning competence.

## Local PowerShell workflow

    Set-Location -LiteralPath (Join-Path $HOME "KindredLabs\NTSA-X-GitHub")
    git pull --ff-only origin main
    py -3.12 -m unittest discover -s tests -p test_academy.py -v
    py -3.12 -m ntsa_x.academy --report
    New-Item -ItemType Directory -Force ".local\academy" | Out-Null
    py -3.12 -m ntsa_x.academy --export ".local\academy\m13-v1"

The exporter writes train-chat.jsonl (24 original practice conversations), exam-reserved.jsonl (12 disjoint questions), and curriculum.json. It rejects overwrites. Never include exam-reserved.jsonl in model training. Do not mistake question-disjoint exams for concept-disjoint or contamination-proof testing.

Grade an answer:

    py -3.12 -m ntsa_x.academy --grade-lesson "math.operators" --answers-json '["22"]'

Scores use normalized exact string matches: this can incorrectly reject equivalent correct mathematical expressions or paraphrases. The readiness gate is deterministic, not a learned tutor.

## Integration and scientific limits

The train-chat.jsonl records use the supervised COG-FU trainer's messages-only format; they do NOT use the rights-attested raw-text manifest used by cog_fu_pretrain. Build an independent held-out supervised conversation set before fitting any new model. The 24 exercises are vastly too few to yield coherent language by themselves. The currently observed native checkpoints collapse into repeated characters.

Use original workbooks or confirm actual licenses for any third-party material. Future implementation should add curriculum selection from independent exam performance, human-approved rights metadata, near-duplicate auditing, diverse instruction demonstrations, semantic or programmatic graders, held-out application tests, and measured tokens-per-mastery / compute cost. Preserve old checkpoints; explicitly version any future model training. This release **does not train a model**.
