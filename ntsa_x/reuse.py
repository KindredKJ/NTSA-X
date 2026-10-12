"""Indexed, context-scoped reuse of verified Nexus convergence records.

Indexes contain SHA-256 fingerprints, never raw initial inputs. The Nexus
hash chain proves internal data integrity, not whether external facts are true.
"""
from __future__ import annotations

from hashlib import sha256
import json
from typing import Mapping, Sequence

from .convergence import ConvergenceReport
from .nexus import NexusNode, NexusStore


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":"), sort_keys=True)


def fingerprint(forward_initial: Mapping, reverse_initial: Mapping) -> str:
    if not isinstance(forward_initial, Mapping) or not isinstance(reverse_initial, Mapping):
        raise TypeError("Input trajectories must be mappings")
    blob = _canonical({"forward": dict(forward_initial), "reverse": dict(reverse_initial)})
    return sha256(blob.encode("utf-8")).hexdigest()


def _identifier(label, value):
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise ValueError(f"{label} must be a nonempty string (<= 256 characters)")


class ConvergenceLibrary:
    """Append-only lookup keys backed by the SAME database as NexusStore.

    The scope must cover operators, goal, constraints, and relevant tool versions.
    Revision must identify the current independent verifier/evidence policy.
    """

    def __init__(self, store: NexusStore):
        if not isinstance(store, NexusStore):
            raise TypeError("ConvergenceLibrary requires NexusStore")
        self.store = store
        with store._lock:
            store._db.executescript("""
                CREATE TABLE IF NOT EXISTS convergence_index (
                    context_id TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    revision TEXT NOT NULL,
                    input_digest TEXT NOT NULL,
                    node_digest TEXT NOT NULL REFERENCES nodes(digest),
                    PRIMARY KEY(context_id, scope, revision, input_digest)
                );
                CREATE INDEX IF NOT EXISTS idx_convergence_node
                    ON convergence_index(node_digest);
            """)

    def _key(self, context_id, scope, revision, inputs):
        for label, value in (("context_id", context_id), ("scope", scope), ("revision", revision)):
            _identifier(label, value)
        if not isinstance(inputs, tuple) or len(inputs) != 2:
            raise TypeError("inputs must be (forward_initial, reverse_initial)")
        return context_id, scope, revision, fingerprint(*inputs)

    def get(self, *, context_id: str, scope: str, revision: str, inputs: tuple) -> NexusNode | None:
        """Integrity-check the index and every ancestor. Caller MUST reverify."""
        key = self._key(context_id, scope, revision, inputs)
        with self.store._lock:
            row = self.store._db.execute(
                "SELECT node_digest FROM convergence_index "
                "WHERE context_id=? AND scope=? AND revision=? AND input_digest=?", key
            ).fetchone()
            if row is None:
                return None
            try:
                binding = self.store.read(row[0])
                if binding is None:
                    return None
                meta = binding.payload
                if (meta.get("kind") != "convergence_index/v1"
                    or meta.get("context_id") != context_id
                    or meta.get("scope") != scope or meta.get("revision") != revision
                    or meta.get("input_digest") != key[3]
                    or len(binding.parents) != 1
                    or meta.get("result_digest") != binding.parents[0]):
                    return None
                node = self.store.read(binding.parents[0])
            except ValueError:
                return None  # Fail closed on corruption or redirected entries.
            if node is None:
                return None
            p = node.payload
            if (p.get("context_id") != context_id
                or not isinstance(p.get("forward"), dict)
                or not isinstance(p.get("reverse"), dict)
                or not isinstance(p.get("resolved"), dict)):
                return None
            return node

    def register(
        self, *, context_id: str, scope: str, revision: str,
        inputs: tuple, node_digest: str, report: ConvergenceReport,
    ) -> str:
        if not isinstance(report, ConvergenceReport) or not report.converged or not report.external_evidence:
            raise ValueError("Cannot index unverified convergence")
        key = self._key(context_id, scope, revision, inputs)
        with self.store._lock:
            node = self.store.read(node_digest)
            if node is None or node.payload.get("context_id") != context_id:
                raise ValueError("Missing or mismatched Nexus convergence")
            p = node.payload
            if not all(isinstance(p.get(k), dict) for k in ("forward", "reverse", "resolved")):
                raise ValueError("Index only accepts complete paired convergence records")
            binding = self.store.append({
                "kind": "convergence_index/v1", "context_id": context_id,
                "scope": scope, "revision": revision,
                "input_digest": key[3], "result_digest": node_digest,
            }, report=report, parents=(node_digest,))
            self.store._db.execute("BEGIN IMMEDIATE")
            try:
                existing = self.store._db.execute(
                    "SELECT node_digest FROM convergence_index WHERE "
                    "context_id=? AND scope=? AND revision=? AND input_digest=?", key
                ).fetchone()
                if existing is not None and existing[0] != binding.digest:
                    raise ValueError("Immutable convergence index conflict: bump revision")
                self.store._db.execute(
                    "INSERT OR IGNORE INTO convergence_index"
                    "(context_id,scope,revision,input_digest,node_digest) VALUES(?,?,?,?,?)",
                    (*key, binding.digest),
                )
                self.store._db.execute("COMMIT")
            except BaseException:
                self.store._db.execute("ROLLBACK")
                raise
        return node_digest

    def compose(
        self, *, context_id: str, scope: str, revision: str, inputs: tuple,
        members: Sequence[str], forward: Mapping, reverse: Mapping,
        resolved: Mapping, report: ConvergenceReport,
    ) -> NexusNode:
        """Make a higher-order convergence with verified ancestry.

        The independent verifier must have already evaluated this COMPOSITION,
        not merely each member. Caller ensures the two projections agree.
        """
        self._key(context_id, scope, revision, inputs)
        if not isinstance(report, ConvergenceReport) or not report.converged or not report.external_evidence:
            raise ValueError("Unverified composite cannot be committed")
        if report.counter_orbit_agreement is not True:
            raise ValueError("Composite requires checked counter-orbit agreement")
        members = tuple(members)
        if not members or len(set(members)) != len(members):
            raise ValueError("Composite must reference unique verified members")
        for key in ("forward", "reverse", "resolved"):
            value = {"forward":forward, "reverse":reverse, "resolved":resolved}[key]
            if not isinstance(value, Mapping):
                raise TypeError("Composite state must be a mapping")
        payload = {
            "kind": "higher_order_convergence/v1",
            "context_id": context_id,
            "forward": dict(forward), "reverse": dict(reverse),
            "resolved": dict(resolved),
            "members": list(members),
            "scope": scope, "revision": revision,
        }
        node = self.store.append(payload, report=report, parents=members)
        self.register(context_id=context_id, scope=scope, revision=revision,
                      inputs=inputs, node_digest=node.digest, report=report)
        return node
