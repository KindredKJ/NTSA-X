"""SQLite-backed, append-only Nexus DAG for verified NTSA-X research state.

This is local storage with hash integrity, NOT a trust authority, encryption layer,
blockchain, or tamper-proof log against an attacker able to rewrite the database.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
from threading import RLock
from typing import Mapping

from .convergence import ConvergenceReport


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


@dataclass(frozen=True)
class NexusNode:
    digest: str
    payload: dict
    parents: tuple[str, ...]
    created_at: str


class NexusStore:
    """Transactionally append and validate multi-parent state records."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        self._lock = RLock()
        self._db = sqlite3.connect(self.path, timeout=10.0, isolation_level=None, check_same_thread=False)
        self._db.execute("PRAGMA foreign_keys=ON")
        if self.path != ":memory:":
            self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript("""
            CREATE TABLE IF NOT EXISTS nodes (
              digest TEXT PRIMARY KEY,
              payload TEXT NOT NULL,
              parents TEXT NOT NULL,
              created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            );
            CREATE TABLE IF NOT EXISTS edges (
              child TEXT NOT NULL REFERENCES nodes(digest),
              parent TEXT NOT NULL REFERENCES nodes(digest),
              PRIMARY KEY(child, parent)
            );
            CREATE INDEX IF NOT EXISTS idx_edges_parent ON edges(parent);
        """)

    def __enter__(self):
        return self

    def __exit__(self, _type, _value, _traceback):
        self.close()

    def close(self):
        with self._lock:
            self._db.close()

    def _row(self, digest):
        return self._db.execute(
            "SELECT digest, payload, parents, created_at FROM nodes WHERE digest=?", (digest,)
        ).fetchone()

    def _verify(self, digest: str, seen: set[str], visiting: set[str]) -> bool:
        if digest in visiting:
            return False
        if digest in seen:
            return True
        row = self._row(digest)
        if row is None:
            return False
        _digest, payload_text, parents_text, _created = row
        try:
            payload = json.loads(payload_text)
            parents = json.loads(parents_text)
            if not isinstance(payload, dict) or not isinstance(parents, list) or not all(isinstance(p, str) for p in parents):
                return False
            record = {"payload": payload, "parents": parents, "verified": True}
            if sha256(_canonical(record).encode("utf-8")).hexdigest() != digest:
                return False
            if payload_text != _canonical(payload) or parents_text != _canonical(parents):
                return False
            rows = self._db.execute("SELECT parent FROM edges WHERE child=?", (digest,)).fetchall()
            if {p for (p,) in rows} != set(parents) or len(rows) != len(parents):
                return False
            visiting.add(digest)
            good = all(self._verify(parent, seen, visiting) for parent in parents)
            visiting.remove(digest)
            if good:
                seen.add(digest)
            return good
        except (TypeError, ValueError, OverflowError, RecursionError, sqlite3.DatabaseError):
            return False

    def verify(self, digest: str) -> bool:
        with self._lock:
            return self._verify(digest, set(), set())

    def append(self, payload: Mapping, *, report: ConvergenceReport, parents=()) -> NexusNode:
        if not isinstance(report, ConvergenceReport) or not report.converged or not report.external_evidence:
            raise ValueError("A converged report with attributed external-evidence check is required")
        if not isinstance(payload, Mapping):
            raise TypeError("Payload must be a mapping")
        parent_list = tuple(parents)
        if len(set(parent_list)) != len(parent_list) or not all(isinstance(p, str) for p in parent_list):
            raise ValueError("Parent digests must be unique strings")
        # Roundtrip to JSON to guarantee a detached, portable value.
        payload_obj = json.loads(_canonical(dict(payload)))
        payload_json = _canonical(payload_obj)
        parents_json = _canonical(list(parent_list))
        record = {"payload": payload_obj, "parents": list(parent_list), "verified": True}
        digest = sha256(_canonical(record).encode("utf-8")).hexdigest()
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                if any(not self._verify(parent, set(), set()) for parent in parent_list):
                    raise ValueError("Unknown or invalid parent lineage")
                existing = self._row(digest)
                if existing:
                    if not self._verify(digest, set(), set()):
                        raise ValueError("Existing record has corrupted lineage")
                    self._db.execute("COMMIT")
                    return NexusNode(digest, json.loads(existing[1]), tuple(json.loads(existing[2])), existing[3])
                self._db.execute(
                    "INSERT INTO nodes(digest,payload,parents) VALUES(?,?,?)",
                    (digest, payload_json, parents_json),
                )
                for parent in parent_list:
                    self._db.execute("INSERT INTO edges(child,parent) VALUES(?,?)", (digest, parent))
                row = self._row(digest)
                self._db.execute("COMMIT")
                return NexusNode(digest, payload_obj, parent_list, row[3])
            except BaseException:
                self._db.execute("ROLLBACK")
                raise

    def read(self, digest: str) -> NexusNode | None:
        with self._lock:
            row = self._row(digest)
            if row is None:
                return None
            if not self._verify(digest, set(), set()):
                raise ValueError("Nexus node failed lineage integrity verification")
            return NexusNode(digest, json.loads(row[1]), tuple(json.loads(row[2])), row[3])

    def lineage(self, digest: str, *, max_depth: int = 32) -> list[NexusNode]:
        if type(max_depth) is not int or max_depth < 0:
            raise ValueError("max_depth must be nonnegative")
        with self._lock:
            origin = self.read(digest)
            if origin is None:
                return []
            output, visited, pending = [], set(), [(origin, 0)]
            while pending:
                node, depth = pending.pop(0)
                if node.digest in visited:
                    continue
                visited.add(node.digest)
                output.append(node)
                if depth < max_depth:
                    for parent in node.parents:
                        pending.append((self.read(parent), depth + 1))
            return output

    def stats(self) -> dict:
        with self._lock:
            nodes = self._db.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
            edges = self._db.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
            return {"nodes": nodes, "edges": edges}
