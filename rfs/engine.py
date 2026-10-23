"""RFS-001: exact, provenance-scoped recall before pure computation.

Standalone; does not import, alter, or write into NTSA-X storage.
SHA-256 detects accidental cache corruption, NOT malicious database tampering.
Verifier is supplied by the caller and must check domain correctness.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from threading import RLock
from typing import Any, Callable
import json
import sqlite3
import time


class VerificationError(RuntimeError):
    """A candidate value failed the current verifier."""


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False)


def _snapshot(value: Any) -> Any:
    """Reject unsupported or non-portable objects and detach caller state."""
    return json.loads(_canonical(value))


def _digest(encoded: str) -> str:
    return sha256(encoded.encode("utf-8")).hexdigest()


def _identifier(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise ValueError(f"{name} must be a nonempty string of <=256 characters")


@dataclass(frozen=True)
class WorkSpec:
    """Complete identity of a deterministic, side-effect-free calculation.

    Caller must include all state that influences the result in arguments,
    context or dependencies. Revisions must change when code/policy changes.
    """
    namespace: str
    tenant: str
    operation: str
    arguments: dict[str, Any]
    context: dict[str, Any]
    implementation_revision: str
    verifier_revision: str
    policy_revision: str
    dependencies: dict[str, str] = field(default_factory=dict)
    ttl_seconds: int = 300
    pure: bool = True

    def __post_init__(self) -> None:
        for name in ("namespace", "tenant", "operation", "implementation_revision",
                     "verifier_revision", "policy_revision"):
            _identifier(getattr(self, name), name)
        if not isinstance(self.pure, bool) or not self.pure:
            raise ValueError("Only explicitly pure, deterministic work is cacheable")
        if isinstance(self.ttl_seconds, bool) or not isinstance(self.ttl_seconds, int) or self.ttl_seconds < 1:
            raise ValueError("ttl_seconds must be a positive integer")
        for name in ("arguments", "context", "dependencies"):
            value = getattr(self, name)
            if not isinstance(value, dict):
                raise TypeError(f"{name} must be a JSON-compatible mapping")
            _snapshot(value)
        if any(not isinstance(k, str) or not isinstance(v, str)
               for k, v in self.dependencies.items()):
            raise TypeError("dependencies must map version keys to string digests")


@dataclass(frozen=True)
class RunResult:
    value: Any
    source: str
    key: str
    value_digest: str


class RecallFirstEngine:
    """SQLite-backed bounded-freshness reuse, isolated from NTSA-X.

    The cache is an optimization, not an authority to issue external effects.
    A cache hit is accepted only after fresh verifier execution.
    """

    def __init__(self, db_path: str | Path, *, clock: Callable[[], float] = time.time):
        location = str(db_path)
        if location != ":memory:" and not location.endswith(".rfs.sqlite3"):
            raise ValueError("RFS database path must end in .rfs.sqlite3")
        if location != ":memory:":
            Path(location).parent.mkdir(parents=True, exist_ok=True)
        self._clock = clock
        self._lock = RLock()
        self._metrics = {"hits": 0, "misses": 0, "invalidations": 0,
                         "worker_invocations": 0, "verifier_invocations": 0}
        self._db = sqlite3.connect(location, timeout=20, isolation_level=None,
                                   check_same_thread=False)
        self._db.execute("PRAGMA busy_timeout=20000")
        if location != ":memory:":
            self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS rfs_exact_results (
                work_key TEXT PRIMARY KEY,
                result_json TEXT NOT NULL,
                result_sha TEXT NOT NULL,
                created_at REAL NOT NULL,
                expires_at REAL NOT NULL
            )
        """)

    def __enter__(self) -> "RecallFirstEngine":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def metrics(self) -> dict[str, int]:
        with self._lock:
            return dict(self._metrics)

    def execute(
        self,
        spec: WorkSpec,
        worker: Callable[[dict[str, Any]], Any],
        verifier: Callable[[dict[str, Any], Any, dict[str, Any]], bool],
    ) -> RunResult:
        if not isinstance(spec, WorkSpec):
            raise TypeError("spec must be a WorkSpec")
        if not callable(worker) or not callable(verifier):
            raise TypeError("worker and verifier must be callable")
        arguments = _snapshot(spec.arguments)
        context = _snapshot(spec.context)
        identity = {
            "schema": "rfs-exact-v1", "namespace": spec.namespace,
            "tenant": spec.tenant, "operation": spec.operation,
            "arguments": arguments, "context": context,
            "implementation_revision": spec.implementation_revision,
            "verifier_revision": spec.verifier_revision,
            "policy_revision": spec.policy_revision,
            "dependencies": _snapshot(spec.dependencies),
            "ttl_seconds": spec.ttl_seconds,
        }
        key = _digest(_canonical(identity))
        with self._lock:
            row = self._db.execute(
                "SELECT result_json, result_sha, expires_at "
                "FROM rfs_exact_results WHERE work_key=?", (key,)
            ).fetchone()
            if row is not None:
                raw, saved_hash, expiry = row
                valid = expiry > self._clock() and _digest(raw) == saved_hash
                if valid:
                    try:
                        recovered = json.loads(raw)
                        self._metrics["verifier_invocations"] += 1
                        valid = verifier(_snapshot(arguments),
                                         _snapshot(recovered), _snapshot(context)) is True
                    except Exception:
                        # Recalculation is safer than accepting a failed verifier.
                        valid = False
                if valid:
                    self._metrics["hits"] += 1
                    return RunResult(recovered, "recalled", key, saved_hash)
                self._db.execute("DELETE FROM rfs_exact_results WHERE work_key=?", (key,))
                self._metrics["invalidations"] += 1

            self._metrics["misses"] += 1
            self._metrics["worker_invocations"] += 1
            computed = _snapshot(worker(_snapshot(arguments)))
            self._metrics["verifier_invocations"] += 1
            if verifier(_snapshot(arguments), _snapshot(computed),
                        _snapshot(context)) is not True:
                raise VerificationError("Fresh computation failed its verifier")
            payload = _canonical(computed)
            result_hash = _digest(payload)
            now = self._clock()
            self._db.execute(
                "INSERT OR REPLACE INTO rfs_exact_results "
                "(work_key,result_json,result_sha,created_at,expires_at) "
                "VALUES (?,?,?,?,?)",
                (key, payload, result_hash, now, now + spec.ttl_seconds),
            )
            return RunResult(computed, "computed", key, result_hash)
