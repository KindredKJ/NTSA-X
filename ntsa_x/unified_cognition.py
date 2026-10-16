"""Unified cognitive execution: one decoding/denoising trajectory plus memory
and trusted native extensions. No model-authored tool plans or plugin manifests.

Extensions are Python callbacks injected by an operator, NOT a secure sandbox.
Only vetted local code may register, and untrusted observations are data.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Iterator
import json
import re

from .memory_fabric import MemoryFabric, MemoryRecord

BOS, EOS, USER, ASSISTANT, SYSTEM = 256, 258, 259, 260, 261


@dataclass(frozen=True)
class CognitiveFrame:
    request: str
    recalled: tuple[MemoryRecord, ...]
    session: str


@dataclass(frozen=True)
class ExtensionResult:
    name: str
    context: str
    provenance: str


@dataclass(frozen=True)
class NativeExtension:
    name: str
    supports: Callable[[str], bool]
    contribute: Callable[[CognitiveFrame], str]


def _validate_name(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[a-z][a-z0-9_.-]{1,40}", value):
        raise ValueError("Invalid native extension identifier")
    return value


def scalar_extension() -> NativeExtension:
    """Use the preexisting fixed-function plugin directly as a cognitive lane."""
    from .plugin_worker import dispatch
    rule = re.compile(
        r"^scalar (increment|decrement|negate|identity) "
        r"(-?\d{1,10})(?: by (\d{1,7}))?$", re.I)

    def supports(request: str) -> bool:
        return bool(rule.fullmatch(request.strip()))

    def contribute(frame: CognitiveFrame) -> str:
        found = rule.fullmatch(frame.request.strip())
        if not found:
            return ""
        operation, value, amount = found.groups()
        outcome = dispatch({
            "operation": operation.lower(),
            "state": {"value": int(value), "amount": int(amount or 0)},
        })
        return f"Native scalar computation: {operation} => {outcome['value']}"

    return NativeExtension("scalar.native", supports, contribute)


def workspace_extension(workbench) -> NativeExtension:
    """Directly attach the workbench's SAFE, read-only local operations.

    Workspace path checks remain active. Neither model output nor external
    content may select write/network/browser/shell actions here.
    """
    rule = re.compile(r"^(?:inspect file|search workspace for) (.{1,120})$", re.I)

    def supports(request: str) -> bool:
        return bool(rule.fullmatch(request.strip()))

    def contribute(frame: CognitiveFrame) -> str:
        found = rule.fullmatch(frame.request.strip())
        if not found:
            return ""
        query = found.group(1)
        if frame.request.strip().lower().startswith("inspect file"):
            value = workbench.dispatch("workspace.read", {"path": query})
            return "Read-only workspace source:\n" + value["text"][:1200]
        value = workbench.dispatch(
            "workspace.search", {"pattern": query, "limit": 5})
        return "Read-only workspace matches:\n" + json.dumps(
            value["matches"], ensure_ascii=False)[:1200]

    return NativeExtension("workspace.native", supports, contribute)


class UnifiedCognitiveMechanism:
    """Host model + memory + extensions in one inference path.

    Existing COG-FU checkpoints remain valid. Inference uses the model's
    native fused recurrent/denoising head, never two competing generated texts.
    The same unified prefix incorporates bounded memory and extension results.
    """

    def __init__(self, model, *, memory: MemoryFabric | None = None,
                 extensions: tuple[NativeExtension, ...] = (),
                 session: str = "default", persist_episodes: bool = False):
        if not isinstance(session, str) or not 1 <= len(session) <= 100:
            raise ValueError("Invalid memory session")
        names = set()
        for ext in extensions:
            if not isinstance(ext, NativeExtension):
                raise ValueError("Only explicitly registered native extensions allowed")
            _validate_name(ext.name)
            if ext.name in names:
                raise ValueError("Duplicate extension")
            names.add(ext.name)
        self.model = model
        self.config = model.config
        self.codec = getattr(model, "codec", None)
        self.architecture = getattr(model, "architecture",
                                    "ntsa-x-cog-fu-discrete-v1")
        self.memory = memory
        self.extensions = extensions
        self.session = session
        self.persist_episodes = persist_episodes
        self.last_receipt: dict | None = None

    def _request(self, prefix: list[int]) -> tuple[int, str]:
        if (not isinstance(prefix, list) or not prefix
                or prefix[0] != BOS or prefix[-1] != ASSISTANT):
            raise ValueError("Expected serialized chat prompt ending at assistant")
        try:
            index = len(prefix) - 1 - prefix[::-1].index(USER)
            end = prefix.index(EOS, index + 1)
        except ValueError:
            raise ValueError("No final user message") from None
        if end != len(prefix) - 2:
            raise ValueError("Only final user turn may precede generation")
        body = prefix[index + 1:end]
        if any(not 0 <= t <= 255 for t in body):
            raise ValueError("Invalid user bytes")
        try:
            text = bytes(body).decode("utf-8", errors="strict")
        except UnicodeError:
            raise ValueError("Malformed user UTF-8") from None
        return index, text

    def _augment(self, prefix: list[int], user_start: int,
                 notes: list[str]) -> tuple[list[int], list[str]]:
        # No truncation of user's own message; drop optional notes if needed.
        added: list[str] = []
        current = prefix
        for note in notes:
            # Insert a bounded SYSTEM observation BEFORE the user's final
            # turn, preserving the learned assistant role suffix.
            record = [SYSTEM, *note.encode("utf-8"), EOS]
            if len(current) + len(record) <= self.config.max_context:
                current = current[:user_start] + record + current[user_start:]
                user_start += len(record)
                added.append(note)
        if len(current) > self.config.max_context:
            raise ValueError("Original prompt exceeds model context limit")
        return current, added

    def generate(self, prefix: list[int], *, max_new_tokens: int = 48,
                 temperature: float = 0.0) -> Iterator[int]:
        start, query = self._request(prefix)
        recalled = self.memory.recall(
            query, session=self.session, limit=5) if self.memory else []
        frame = CognitiveFrame(query, tuple(recalled), self.session)
        notes = []
        for record in recalled:
            # No record is treated as externally verified fact.
            notes.append(
                f"[Unverified {record.kind} recall, source={record.source}] "
                + record.content[:400])
        contributions = []
        for ext in self.extensions:
            if ext.supports(query):
                # Operator-approved trusted native code; output bounded and
                # observed, NOT a model-directed arbitrary plugin call.
                raw = ext.contribute(frame)
                if not isinstance(raw, str) or len(raw.encode("utf-8")) > 2048:
                    raise ValueError("Native extension returned invalid output")
                if raw:
                    notes.append(f"[Extension {ext.name}, unverified text] " + raw)
                    contributions.append(ext.name)
        augmented, accepted = self._augment(prefix, start, notes)
        # One neural update loop: native recurrence and masked-byte denoising
        # jointly select each byte and carry the updated state onward.
        produced = self.model.unified_draft(
            augmented, max_new_tokens=max_new_tokens,
            temperature=temperature)
        answer = bytes(produced).decode("utf-8", errors="replace").replace("\ufffd", "?")
        committed = False
        if self.persist_episodes:
            if self.memory is None or not self.memory.writable:
                raise PermissionError("Episodic persistence not authorized")
            self.memory.remember(
                "episodic",
                f"User asked: {query[:500]}\nNative response: {answer[:500]}",
                source="unified-cognition:unverified-output",
                session=self.session, ttl_seconds=30 * 86400)
            committed = True
        self.last_receipt = {
            "architecture": self.architecture,
            "decoder": "unified-recurrent-masked-v1",
            "memory_recalled": len(recalled),
            "memory_accepted_into_context": sum(
                1 for note in accepted if note.startswith("[Unverified")),
            "native_extensions": contributions,
            "notes_accepted": len(accepted),
            "episodic_memory_committed": committed,
            "verified_external_truth": False,
            "cognitive_fluency_proven": False,
        }
        yield from answer.encode("utf-8")

    def learn(self, kind: str, content: str, *, source: str,
              ttl_seconds: int | None = None) -> str:
        """Caller-authorized bidirectional memory submission, not trained weights."""
        if self.memory is None:
            raise PermissionError("Memory fabric not attached")
        return self.memory.remember(
            kind, content, source=source, session=self.session,
            ttl_seconds=ttl_seconds)
