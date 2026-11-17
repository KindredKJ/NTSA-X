"""Kindred secondary semantic memory: persistent, scoped and recall-before-compute.

This is an optional adapter for NTSA-X UnifiedCognitiveMechanism. Dense semantic
quality depends on the supplied embedding model; similarity is NOT a proof of
truth or a safe substitute for exact verified convergence in ConvergenceLibrary.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from math import isfinite, sqrt
from pathlib import Path
from threading import RLock
from typing import Protocol, Sequence
import json
import sqlite3
import time
import uuid

KINDS = frozenset({"working", "episodic", "semantic", "procedural", "relational"})


class Embedder(Protocol):
    model_id: str

    def embed_document(self, text: str) -> Sequence[float]: ...

    def embed_query(self, text: str) -> Sequence[float]: ...


class SentenceTransformersEmbedder:
    """Local embedding adapter; requires optional sentence-transformers package.

    Model files must be available to the runtime. The selected model determines
    language coverage, semantic recall quality, latency, and embedding width.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", *, device: str = "cpu"):
        if not model_name or not isinstance(model_name, str):
            raise ValueError("Model name required")
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("Install optional dependency: pip install sentence-transformers") from exc
        self.model_id = model_name
        self._model = SentenceTransformer(model_name, device=device, trust_remote_code=False)

    def embed_document(self, text: str) -> Sequence[float]:
        return self._model.encode_document(text, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> Sequence[float]:
        return self._model.encode_query(text, normalize_embeddings=True).tolist()


@dataclass(frozen=True)
class MemoryRecord:
    id: str
    kind: str
    content: str
    source: str
    session: str
    verified: bool
    created: float


@dataclass(frozen=True)
class SemanticHit:
    record: MemoryRecord
    similarity: float


@dataclass(frozen=True)
class CaptureDecision:
    stored: bool
    reason: str
    memory_id: str | None


def _unit_vector(vector: Sequence[float]) -> tuple[float, ...]:
    try:
        values = tuple(float(x) for x in vector)
    except (TypeError, ValueError) as exc:
        raise ValueError("Embedding must be a numeric sequence") from exc
    if not 2 <= len(values) <= 4096 or any(not isfinite(x) for x in values):
        raise ValueError("Embedding dimension or numeric value invalid")
    norm = sqrt(sum(x*x for x in values))
    if norm < 1e-12:
        raise ValueError("Embedding norm must be nonzero")
    return tuple(x/norm for x in values)


def _bounded_text(value: str, *, limit: int, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value.encode("utf-8")) > limit:
        raise ValueError(f"{name} must contain 1..{limit} UTF-8 bytes")
    return value


class SemanticMemory:
    """Secondary persistence engine backed by SQLite and dense cosine retrieval.

    Records remain scoped to a single tenant and session. This deliberately
    differs from NTSA-X's original MemoryFabric cross-session semantic behavior.
    You can use a stable session name for long-lived memory of one user/project.
    """

    def __init__(self, path: str | Path, *, embedder: Embedder,
                 tenant: str = "founder", writable: bool = False,
                 max_scan: int = 20000):
        self.path = str(path)
        self.embedder = embedder
        self.tenant = _bounded_text(tenant, limit=100, name="tenant")
        self.writable = writable
        if not isinstance(writable, bool) or type(max_scan) is not int or not 1 <= max_scan <= 100000:
            raise ValueError("Invalid memory options")
        self.max_scan = max_scan
        self.model_id = _bounded_text(getattr(embedder, "model_id", None), limit=200, name="model_id")
        self._lock = RLock()
        self._db = sqlite3.connect(self.path, timeout=10, isolation_level=None, check_same_thread=False)
        self._db.execute("PRAGMA foreign_keys=ON")
        if self.path != ":memory:":
            self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript("""
            CREATE TABLE IF NOT EXISTS semantic_memories (
              id TEXT PRIMARY KEY,
              tenant TEXT NOT NULL,
              session TEXT NOT NULL,
              kind TEXT NOT NULL,
              content TEXT NOT NULL,
              source TEXT NOT NULL,
              created REAL NOT NULL,
              expires REAL,
              digest TEXT NOT NULL,
              model_id TEXT NOT NULL,
              embedding TEXT NOT NULL,
              UNIQUE(tenant,session,kind,digest,model_id)
            );
            CREATE INDEX IF NOT EXISTS idx_semantic_scope
              ON semantic_memories(tenant,session,model_id,expires);
        """)

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _embedding(self, text: str, *, query: bool) -> tuple[float, ...]:
        method = self.embedder.embed_query if query else self.embedder.embed_document
        return _unit_vector(method(text))

    def remember(self, kind: str, content: str, *, source: str,
                 session: str = "default", verified: bool = False,
                 ttl_seconds: int | None = None) -> str:
        if not self.writable:
            raise PermissionError("Semantic memory write capability disabled")
        if kind not in KINDS or type(verified) is not bool or verified:
            raise ValueError("Invalid memory kind or unsupported independent verification")
        _bounded_text(content, limit=4096, name="content")
        _bounded_text(source, limit=180, name="source")
        _bounded_text(session, limit=100, name="session")
        if ttl_seconds is not None and (type(ttl_seconds) is not int or not 1 <= ttl_seconds <= 31536000):
            raise ValueError("TTL must be 1..31536000 seconds")
        digest = sha256(content.encode("utf-8")).hexdigest()
        vector = json.dumps(self._embedding(content, query=False), separators=(",", ":"))
        now = time.time()
        expiry = now + ttl_seconds if ttl_seconds is not None else None
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                self._db.execute("DELETE FROM semantic_memories WHERE expires IS NOT NULL AND expires<=?", (now,))
                existing = self._db.execute("""
                    SELECT id FROM semantic_memories
                    WHERE tenant=? AND session=? AND kind=? AND digest=? AND model_id=?
                """, (self.tenant, session, kind, digest, self.model_id)).fetchone()
                if existing:
                    ident = existing[0]
                else:
                    ident = uuid.uuid4().hex
                    self._db.execute("""
                        INSERT INTO semantic_memories
                          (id,tenant,session,kind,content,source,created,expires,digest,model_id,embedding)
                        VALUES(?,?,?,?,?,?,?,?,?,?,?)
                    """, (ident, self.tenant, session, kind, content, source, now,
                          expiry, digest, self.model_id, vector))
                self._db.execute("COMMIT")
            except BaseException:
                self._db.execute("ROLLBACK")
                raise
        return ident

    def capture(self, kind: str, content: str, *, source: str,
                session: str = "default", importance: float,
                durability: float, confidence: float,
                ttl_seconds: int | None = None) -> CaptureDecision:
        """Deterministic retention policy for caller-assessed memory candidates.

        Confidence/importance are caller assertions, not model-based guarantees.
        Never ingest unsolicited model output as verified external truth.
        """
        factors = (importance, durability, confidence)
        if any(type(v) not in (float, int) or not isfinite(v) or not 0 <= v <= 1 for v in factors):
            raise ValueError("Retention factors must be finite numbers between 0 and 1")
        if confidence < 0.6:
            return CaptureDecision(False, "insufficient_source_confidence", None)
        if importance * durability < 0.45:
            return CaptureDecision(False, "low_expected_future_utility", None)
        memory_id = self.remember(kind, content, source=source, session=session, ttl_seconds=ttl_seconds)
        return CaptureDecision(True, "retained_unverified", memory_id)

    def get(self, ident: str, *, session: str = "default") -> MemoryRecord | None:
        _bounded_text(session, limit=100, name="session")
        with self._lock:
            row = self._db.execute("""
                SELECT id,kind,content,source,session,created FROM semantic_memories
                WHERE tenant=? AND session=? AND id=? AND model_id=?
                  AND (expires IS NULL OR expires>?)
            """, (self.tenant, session, ident, self.model_id, time.time())).fetchone()
        return MemoryRecord(row[0],row[1],row[2],row[3],row[4],False,row[5]) if row else None

    def recall_hits(self, query: str, *, session: str = "default",
                    limit: int = 5, min_similarity: float = 0.3) -> list[SemanticHit]:
        _bounded_text(query, limit=4096, name="query")
        _bounded_text(session, limit=100, name="session")
        if type(limit) is not int or not 1 <= limit <= 20 or not isfinite(min_similarity) or not -1 <= min_similarity <= 1:
            raise ValueError("Invalid recall limit or threshold")
        q = self._embedding(query, query=True)
        with self._lock:
            rows = self._db.execute("""
                SELECT id,kind,content,source,session,created,embedding
                FROM semantic_memories
                WHERE tenant=? AND session=? AND model_id=?
                  AND (expires IS NULL OR expires>?)
                ORDER BY created DESC LIMIT ?
            """, (self.tenant, session, self.model_id, time.time(), self.max_scan + 1)).fetchall()
        if len(rows) > self.max_scan:
            raise RuntimeError("Semantic memory exceeds exact scan budget; configure a vector index")
        hits = []
        for ident,kind,content,source,sess,created,embedding in rows:
            values = _unit_vector(json.loads(embedding))
            if len(q) != len(values):
                raise ValueError("Embedding dimension mismatch: rebuild index with same model")
            score = sum(a*b for a,b in zip(q,values))
            if score >= min_similarity:
                hits.append(SemanticHit(MemoryRecord(ident,kind,content,source,sess,False,created),score))
        hits.sort(key=lambda h: (-h.similarity,-h.record.created,h.record.id))
        return hits[:limit]

    def recall(self, query: str, *, session: str = "default",
               limit: int = 5, include_associations: bool = True) -> list[MemoryRecord]:
        return [hit.record for hit in self.recall_hits(query, session=session, limit=limit)]

    def summary(self, *, session: str = "default") -> dict:
        _bounded_text(session, limit=100, name="session")
        with self._lock:
            count = self._db.execute("""SELECT COUNT(*) FROM semantic_memories WHERE
                tenant=? AND session=? AND model_id=? AND (expires IS NULL OR expires>?)""",
                (self.tenant,session,self.model_id,time.time())).fetchone()[0]
        return {"count":count,"tenant":self.tenant,"session":session,
                "model_id":self.model_id,"writable":self.writable,"verified_external_truth":False}


class FusedRecallMemory:
    """Drop-in memory port for NTSA-X UnifiedCognitiveMechanism.

    Existing MemoryFabric remains primary for lexical/episodic/procedural data;
    the new SemanticMemory provides dense recall. This class has the same
    recall and remember methods used by UnifiedCognitiveMechanism.
    """

    def __init__(self, primary, semantic: SemanticMemory):
        self.primary = primary
        self.semantic = semantic
        self.writable = bool(primary.writable and semantic.writable)

    def remember(self, kind: str, content: str, *, source: str,
                 session: str = "default", ttl_seconds: int | None = None) -> str:
        if kind == "semantic":
            return self.semantic.remember(kind,content,source=source,session=session,ttl_seconds=ttl_seconds)
        return self.primary.remember(kind,content,source=source,session=session,ttl_seconds=ttl_seconds)

    def recall(self, query: str, *, session: str = "default",limit: int = 5,
               include_associations: bool = True) -> list[MemoryRecord]:
        if type(limit) is not int or not 1 <= limit <= 20:
            raise ValueError("Invalid fusion recall limit")
        dense = self.semantic.recall(query,session=session,limit=limit)
        lexical = self.primary.recall(query,session=session,limit=limit,
                                      include_associations=include_associations)
        found = set()
        combined = []
        for entry in [*dense,*lexical]:
            if entry.session != session:
                continue
            key = (entry.kind,entry.content)
            if key not in found:
                combined.append(entry)
                found.add(key)
            if len(combined) == limit:
                break
        return combined
