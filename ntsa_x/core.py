"""Minimal deterministic lineage and operator gating for NTSA-X research."""
from dataclasses import dataclass
from copy import deepcopy
from math import isfinite
import hashlib
import json

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

@dataclass(frozen=True)
class Block:
    payload: dict
    parents: tuple[str, ...]
    verified: bool
    digest: str

class MemoryChain:
    def __init__(self):
        self.blocks: dict[str, Block] = {}

    def append(self, payload: dict, parents=(), *, verified=False):
        if not verified:
            raise ValueError("Cannot commit unverified state")
        parents = tuple(parents)
        if any(parent not in self.blocks for parent in parents):
            raise ValueError("Unknown parent")
        if any(not self.verify(parent) for parent in parents):
            raise ValueError("Invalid parent lineage")
        payload = deepcopy(payload)
        record = {"payload": payload, "parents": list(parents), "verified": True}
        digest = hashlib.sha256(canonical(record).encode("utf-8")).hexdigest()
        block = Block(dict(payload), parents, True, digest)
        if digest in self.blocks and self.blocks[digest] != block:
            raise ValueError("Hash collision or inconsistent content")
        self.blocks[digest] = block
        return block

    def verify(self, digest: str) -> bool:
        """Validate canonical digests and ancestral presence; not external truth."""
        seen: set[str] = set()
        pending: set[str] = set()

        def visit(current):
            if current in pending:
                return False
            if current in seen:
                return True
            block = self.blocks.get(current)
            if block is None or not block.verified or block.digest != current:
                return False
            pending.add(current)
            record = {"payload": block.payload, "parents": list(block.parents), "verified": True}
            try:
                valid = hashlib.sha256(canonical(record).encode("utf-8")).hexdigest() == current
                valid = valid and all(visit(parent) for parent in block.parents)
            except (TypeError, ValueError, OverflowError):
                valid = False
            pending.remove(current)
            if valid:
                seen.add(current)
            return valid

        return visit(digest)

@dataclass(frozen=True)
class GateDecision:
    enabled: bool
    reason: str

class Scheduler:
    """Budget-aware operator admission; no claim of optimal scheduling."""
    def gate(self, *, verified_recall: bool, estimated_cost: float, remaining_budget: float):
        if verified_recall:
            return GateDecision(False, "verified recall available")
        if not isfinite(estimated_cost) or not isfinite(remaining_budget) or estimated_cost < 0 or remaining_budget < 0:
            raise ValueError("Costs and budgets must be nonnegative")
        if estimated_cost > remaining_budget:
            return GateDecision(False, "budget exceeded")
        return GateDecision(True, "admitted within budget")
