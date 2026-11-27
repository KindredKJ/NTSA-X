"""Exact GF(2) transition composition and Nexus-backed, input-bound replay.

This is a computational primitive, not a neural model or independent verifier.
Only constant masks compose for arbitrary starting states. Recorded adaptive
trajectories may be replayed only for the original input and rule context.
"""
from dataclasses import dataclass
from hashlib import sha256
from typing import Callable, Iterable

from .convergence import ConvergenceReport
from .nexus import NexusStore

KIND = "xor_ciu/trajectory/v1"
MAX_WIDTH = 1_048_576


def _width(width: int) -> int:
    if type(width) is not int or not 1 <= width <= MAX_WIDTH:
        raise ValueError("invalid binary width")
    return width


def _bits(value: int, width: int) -> int:
    if type(value) is not int or value < 0 or value.bit_length() > width:
        raise ValueError("value outside binary width")
    return value


def _label(value: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise ValueError("invalid context or revision")
    return value


@dataclass(frozen=True)
class XorTransition:
    width: int
    mask: int
    count: int

    def __post_init__(self):
        _bits(self.mask, _width(self.width))
        if type(self.count) is not int or self.count < 0:
            raise ValueError("invalid operation count")

    @classmethod
    def compose(cls, width: int, masks: Iterable[int]) -> "XorTransition":
        width = _width(width)
        combined = count = 0
        for mask in masks:
            combined ^= _bits(mask, width)
            count += 1
        return cls(width, combined, count)

    def apply(self, state: int) -> int:
        return _bits(state, self.width) ^ self.mask

    def then(self, other: "XorTransition") -> "XorTransition":
        if not isinstance(other, XorTransition) or other.width != self.width:
            raise ValueError("incompatible transition widths")
        return XorTransition(self.width, self.mask ^ other.mask,
                             self.count + other.count)


class NexusXorCIU:
    """Persist verified trajectory shortcuts in the existing Nexus store.

    Callers provide their existing independent convergence report, a verifier
    bound to the current task, and a rule revision covering all dependencies.
    A receipt is a pointer for exact replay, not evidence of generalization.
    """

    def __init__(self, store: NexusStore):
        if not isinstance(store, NexusStore):
            raise TypeError("NexusStore required")
        self.store = store

    def record(self, *, width: int, initial: int, terminal: int,
               masks: Iterable[int], context_id: str, revision: str,
               report: ConvergenceReport, verify: Callable[[int], bool],
               parents: tuple[str, ...] = ()) -> str:
        width = _width(width)
        _bits(initial, width)
        _bits(terminal, width)
        _label(context_id)
        _label(revision)
        if (not isinstance(report, ConvergenceReport) or not report.converged
                or not report.external_evidence):
            raise ValueError("independently verified convergence required")
        digest = sha256(b"ntsa-x/xor-ciu/trajectory/v1\0" + width.to_bytes(4, "big"))
        size = (width + 7) // 8

        def traced():
            for mask in masks:
                _bits(mask, width)
                digest.update(mask.to_bytes(size, "big"))
                yield mask

        transition = XorTransition.compose(width, traced())
        if transition.apply(initial) != terminal or verify(terminal) is not True:
            raise ValueError("terminal verification failed")
        payload = {
            "kind": KIND, "context_id": context_id, "revision": revision,
            "width": width, "initial": hex(initial), "terminal": hex(terminal),
            "mask": hex(transition.mask), "count": transition.count,
            "trace_digest": digest.hexdigest(),
        }
        return self.store.append(payload, report=report, parents=parents).digest

    def replay(self, receipt: str, *, initial: int, context_id: str,
               revision: str, verify: Callable[[int], bool]) -> int | None:
        _label(context_id)
        _label(revision)
        node = self.store.read(receipt)
        if node is None:
            return None
        p = node.payload
        if (p.get("kind") != KIND or p.get("context_id") != context_id
                or p.get("revision") != revision):
            return None
        width = _width(p["width"])
        _bits(initial, width)
        if hex(initial) != p["initial"]:
            return None
        transition = XorTransition(width, int(p["mask"], 16), p["count"])
        terminal = transition.apply(initial)
        if hex(terminal) != p["terminal"] or verify(terminal) is not True:
            return None
        return terminal

    def compose_receipts(self, receipts: Iterable[str], *, initial: int,
                         context_id: str, revision: str,
                         report: ConvergenceReport,
                         verify: Callable[[int], bool]) -> str:
        """Compose an ordered chain of input-bound, compatible Nexus receipts.

        Original traces are not expanded. Every receipt must match the current
        intermediate state and rule context. Caller independently verifies the
        composite outcome; parent hash integrity is not semantic evidence.
        """
        _label(context_id)
        _label(revision)
        if (not isinstance(report, ConvergenceReport) or not report.converged
                or not report.external_evidence
                or report.counter_orbit_agreement is not True):
            raise ValueError("verified composite with paired agreement required")
        ordered = []
        combined = None
        state = initial
        digest = sha256(b"ntsa-x/xor-ciu/composite/v1\0")
        for receipt in receipts:
            if len(ordered) >= 4096:
                raise ValueError("too many direct members; compose in groups")
            if not isinstance(receipt, str) or len(receipt) != 64:
                raise ValueError("invalid receipt")
            node = self.store.read(receipt)
            if node is None:
                raise ValueError("missing receipt")
            p = node.payload
            if (p.get("kind") != KIND or p.get("context_id") != context_id
                    or p.get("revision") != revision):
                raise ValueError("incompatible receipt context")
            transition = XorTransition(p["width"], int(p["mask"], 16), p["count"])
            _bits(state, transition.width)
            if hex(state) != p["initial"]:
                raise ValueError("discontinuous trajectory")
            state = transition.apply(state)
            if hex(state) != p["terminal"]:
                raise ValueError("invalid member terminal")
            combined = transition if combined is None else combined.then(transition)
            ordered.append(receipt)
            digest.update(bytes.fromhex(receipt))
        if combined is None:
            raise ValueError("empty composite")
        if combined.apply(initial) != state or verify(state) is not True:
            raise ValueError("composite verification failed")
        payload = {
            "kind": KIND, "context_id": context_id, "revision": revision,
            "width": combined.width, "initial": hex(initial),
            "terminal": hex(state), "mask": hex(combined.mask),
            "count": combined.count, "trace_digest": digest.hexdigest(),
            "trace_kind": "ordered_receipts/v1", "members": ordered,
        }
        return self.store.append(payload, report=report,
                                 parents=tuple(dict.fromkeys(ordered))).digest
