"""Explicitly approved teacher examples; never verified Nexus facts."""
from __future__ import annotations
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
from hashlib import sha256
from pathlib import Path
from uuid import uuid4
import json
import sqlite3


def encoded(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def digest(value):
    return sha256(encoded(value).encode("utf-8")).hexdigest()


def history_ok(history):
    if not isinstance(history, list) or len(history) > 24:
        raise ValueError("Invalid context")
    for turn in history:
        if (not isinstance(turn, dict) or set(turn) != {"role", "content"}
                or turn["role"] not in ("user", "assistant")
                or not isinstance(turn["content"], str)
                or len(turn["content"]) > 16000):
            raise ValueError("Invalid conversation history")
    return history


class LearningCorpus:
    """Local dataset with human review, provenance and exact-context recall."""

    def __init__(self, path):
        self.path = str(path)
        if self.path == ":memory:":
            raise ValueError("Corpus requires a durable path")
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS samples(
                id TEXT PRIMARY KEY, session TEXT NOT NULL,
                prompt TEXT NOT NULL, answer TEXT NOT NULL,
                context TEXT NOT NULL, prompt_sha TEXT NOT NULL,
                context_sha TEXT NOT NULL, teacher TEXT NOT NULL,
                expires TEXT NOT NULL, approval TEXT,
                shared INTEGER NOT NULL DEFAULT 0)""")
            db.execute("CREATE INDEX IF NOT EXISTS approved_lookup "
                       "ON samples(prompt_sha,context_sha,approval,expires)")

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        try:
            with db:
                yield db
        finally:
            db.close()

    def capture(self, session, prompt, answer, context, teacher, days=30):
        from .live_gateway import valid_session
        valid_session(session)
        history_ok(context)
        if (not isinstance(prompt, str) or not 1 <= len(prompt.strip()) <= 4000
                or not isinstance(answer, str) or not 1 <= len(answer) <= 16000
                or not isinstance(teacher, str) or not 1 <= len(teacher) <= 128
                or not isinstance(days, int) or not 1 <= days <= 365):
            raise ValueError("Invalid learning sample")
        sample = uuid4().hex
        expires = (datetime.now(timezone.utc) + timedelta(days=days)).isoformat()
        with self.connect() as db:
            db.execute("INSERT INTO samples VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                       (sample, session, prompt, answer, encoded(context),
                        digest(prompt), digest(context), teacher, expires, None, 0))
        return sample

    def approve(self, session, sample, rights, share=False):
        from .live_gateway import valid_session
        valid_session(session)
        if (rights is not True or type(share) is not bool
                or not isinstance(sample, str) or len(sample) != 32):
            raise ValueError("Explicit approval, permission and sample ID required")
        with self.connect() as db:
            result = db.execute("""UPDATE samples SET approval=?, shared=?
                WHERE session=? AND id=? AND approval IS NULL AND expires>?""",
                (datetime.now(timezone.utc).isoformat(), int(share), session,
                 sample, datetime.now(timezone.utc).isoformat()))
        return result.rowcount == 1

    def recall(self, session, prompt, context):
        from .live_gateway import valid_session
        valid_session(session)
        history_ok(context)
        with self.connect() as db:
            result = db.execute("""SELECT answer FROM samples
                WHERE prompt_sha=? AND context_sha=? AND approval IS NOT NULL
                  AND expires>? AND
                  (session=? OR (shared=1 AND context='[]'))
                ORDER BY approval DESC LIMIT 1""",
                (digest(prompt), digest(context),
                 datetime.now(timezone.utc).isoformat(), session)).fetchone()
        return result[0] if result else None

    def clear(self, session):
        from .live_gateway import valid_session
        valid_session(session)
        with self.connect() as db:
            db.execute("DELETE FROM samples WHERE session=?", (session,))

    def export(self, train_file, eval_file):
        """Export only approved, unexpired samples with prompt-disjoint split."""
        with self.connect() as db:
            rows = db.execute("""SELECT prompt,answer,context FROM samples
                WHERE approval IS NOT NULL AND expires>? ORDER BY id""",
                (datetime.now(timezone.utc).isoformat(),)).fetchall()
        train, evaluation = [], []
        for prompt, answer, context in rows:
            item = {"messages": json.loads(context) +
                    [{"role": "user", "content": prompt},
                     {"role": "assistant", "content": answer}]}
            (evaluation if int(digest(prompt)[:8], 16) % 5 == 0
             else train).append(item)
        if not train or not evaluation:
            raise ValueError("More approved, diverse examples required for holdout")
        paths = (Path(train_file), Path(eval_file))
        if paths[0] == paths[1] or any(path.exists() for path in paths):
            raise ValueError("Refusing to overwrite existing dataset")
        for path, rows in zip(paths, (train, evaluation)):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("".join(encoded(row) + "\n" for row in rows),
                            encoding="utf-8")
        return {"training_examples": len(train),
                "held_out_examples": len(evaluation)}


if __name__ == "__main__":
    from argparse import ArgumentParser
    p = ArgumentParser()
    p.add_argument("--db", default=str(Path.home() / ".kindred" / "ntsa-learning.sqlite3"))
    p.add_argument("--train", required=True)
    p.add_argument("--eval", required=True)
    a = p.parse_args()
    print(encoded(LearningCorpus(a.db).export(a.train, a.eval)))
