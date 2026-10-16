"""Typed bidirectional memory fabric for NTSA-X unified cognition.

Working, episodic, semantic, procedural, and relational/associative records
share one local interface. A search score is lexical; do NOT mistake this for
trained semantic embeddings, verified knowledge or autonomous learning.
Writes require an explicit caller capability. The Nexus verified ledger stays
separate. Database is local and NOT an encryption or adversarial sandbox.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
import json
import re
import sqlite3
import time
from uuid import uuid4


KINDS = frozenset({"working", "episodic", "semantic", "procedural", "relational"})
WORDS = re.compile(r"[^\W_]+", flags=re.UNICODE)


def _terms(value: str) -> set[str]:
    return {word.casefold() for word in WORDS.findall(value) if len(word) > 1}


@dataclass(frozen=True)
class MemoryRecord:
    id: str
    kind: str
    content: str
    source: str
    session: str
    verified: bool
    created: float


class MemoryFabric:
    """Local recall/write/link API with explicit mutation permission."""
    def __init__(self, path: str | Path = ":memory:", *, writable: bool = False):
        self.path = str(path)
        self.writable = writable
        self._lock = RLock()
        self._db = sqlite3.connect(
            self.path, timeout=10, isolation_level=None,
            check_same_thread=False)
        self._db.execute("PRAGMA foreign_keys=ON")
        if self.path != ":memory:":
            self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript("""
            CREATE TABLE IF NOT EXISTS memories(
              id TEXT PRIMARY KEY, kind TEXT NOT NULL, content TEXT NOT NULL,
              source TEXT NOT NULL, session TEXT NOT NULL,
              verified INTEGER NOT NULL CHECK(verified IN (0,1)),
              created REAL NOT NULL, expires REAL);
            CREATE TABLE IF NOT EXISTS associations(
              a TEXT NOT NULL REFERENCES memories(id),
              b TEXT NOT NULL REFERENCES memories(id),
              relation TEXT NOT NULL,
              PRIMARY KEY(a,b,relation));
            CREATE INDEX IF NOT EXISTS idx_memory_session ON memories(session);
        """)

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def remember(self, kind: str, content: str, *, source: str,
                 session: str = "default", verified: bool = False,
                 ttl_seconds: int | None = None) -> str:
        if not self.writable:
            raise PermissionError("Memory mutation capability disabled")
        if (kind not in KINDS or type(verified) is not bool
                or not isinstance(content, str)
                or not 1 <= len(content.encode("utf-8")) <= 4096
                or not isinstance(source, str) or not 1 <= len(source) <= 180
                or not isinstance(session, str) or not 1 <= len(session) <= 100
                or (ttl_seconds is not None and
                    (type(ttl_seconds) is not int or
                     not 1 <= ttl_seconds <= 31_536_000))):
            raise ValueError("Invalid bounded memory record")
        if verified:
            # This API does not independently verify external claims.
            raise ValueError("Only an external verifier may attest facts")
        now = time.time()
        ident = uuid4().hex
        with self._lock:
            self._db.execute(
                "INSERT INTO memories VALUES(?,?,?,?,?,?,?,?)",
                (ident, kind, content, source, session, 0, now,
                 now + ttl_seconds if ttl_seconds is not None else None))
        return ident

    def relate(self, first: str, second: str, relation: str) -> None:
        if not self.writable:
            raise PermissionError("Memory mutation capability disabled")
        if (not isinstance(first, str) or not isinstance(second, str)
                or first == second or not isinstance(relation, str)
                or not re.fullmatch(r"[a-z_]{2,32}", relation)):
            raise ValueError("Invalid association")
        with self._lock:
            for ident in (first, second):
                if not self._db.execute(
                    "SELECT 1 FROM memories WHERE id=?", (ident,)
                ).fetchone():
                    raise ValueError("Unknown memory in relation")
            self._db.execute(
                "INSERT OR IGNORE INTO associations(a,b,relation) VALUES(?,?,?)",
                (first, second, relation))

    def get(self, ident: str) -> MemoryRecord | None:
        with self._lock:
            item = self._db.execute(
                """SELECT id,kind,content,source,session,verified,created
                   FROM memories
                   WHERE id=? AND (expires IS NULL OR expires>?)""",
                (ident, time.time())).fetchone()
        return MemoryRecord(*item) if item else None

    def recall(self, query: str, *, session: str = "default",
               limit: int = 5, include_associations: bool = True) -> list[MemoryRecord]:
        if (not isinstance(query, str) or len(query) > 4096
                or not isinstance(session, str) or len(session) > 100
                or type(limit) is not int or not 1 <= limit <= 20):
            raise ValueError("Invalid recall query")
        words = _terms(query)
        if not words:
            return []
        with self._lock:
            rows = self._db.execute(
                """SELECT id,kind,content,source,session,verified,created
                   FROM memories WHERE (expires IS NULL OR expires>?)
                   AND (kind!='working' OR session=?)
                   ORDER BY created DESC LIMIT 1000""",
                (time.time(), session)).fetchall()
            scores = []
            for data in rows:
                overlap = len(words & _terms(data[2]))
                if overlap:
                    # direct lexical matches first; working same-session
                    # context gets a small tie-break, never authoritative.
                    scores.append((overlap, int(data[1] == "working"), data[6],
                                   MemoryRecord(*data)))
            scores.sort(reverse=True, key=lambda item: item[:3])
            direct = [value[-1] for value in scores[:limit]]
            if not include_associations or len(direct) >= limit:
                return direct
            # One graph hop surfaces related context WITHOUT claiming that
            # an association establishes a true logical implication.
            found = {x.id for x in direct}
            for item in list(direct):
                linked = self._db.execute(
                    """SELECT CASE WHEN a=? THEN b ELSE a END
                       FROM associations WHERE a=? OR b=? LIMIT 20""",
                    (item.id, item.id, item.id)).fetchall()
                for (neighbor,) in linked:
                    if neighbor in found:
                        continue
                    record = self.get(neighbor)
                    if record and (record.kind != "working"
                                   or record.session == session):
                        direct.append(record)
                        found.add(neighbor)
                    if len(direct) >= limit:
                        return direct
        return direct

    def summary(self) -> dict:
        with self._lock:
            rows = self._db.execute(
                """SELECT kind,COUNT(*) FROM memories
                   WHERE expires IS NULL OR expires>? GROUP BY kind""",
                (time.time(),)).fetchall()
        return {"counts": dict(rows), "writable": self.writable,
                "verification": "operator/unverified, never external truth"}
