"""Bounded metadata-only event stream for an optional Materializer renderer."""
from dataclasses import dataclass
from datetime import datetime, timezone
from math import isfinite
from typing import Callable
from uuid import uuid4

# Strictly metadata keys. No prompts, memory contents, credential material, or raw states.
SAFE_FIELDS = frozenset({
    "step", "axis_x", "axis_y", "axis_z", "operator", "enabled", "reason",
    "estimated_cost", "remaining_budget", "spent", "score", "verified",
    "digest", "parent_count", "check_count", "missing_count", "failed_count",
    "contradiction_count", "external_evidence", "counter_orbit_agreement",
    "source_count", "closure", "selected", "available", "max_steps",
    "agreement", "node_count", "edge_count", "direction",
    "task_id", "batch_size", "max_parallel", "running", "completed", "failed",
    "blocked", "throughput",
})


@dataclass(frozen=True)
class MaterializerEvent:
    schema_version: str
    event_id: str
    trace_id: str
    timestamp: str
    type: str
    payload: dict
    provenance: str = "ntsa-x"
    redaction_level: str = "metadata"

    def to_dict(self):
        return {
            "schema_version": self.schema_version,
            "event_id": self.event_id, "trace_id": self.trace_id,
            "timestamp": self.timestamp, "type": self.type,
            "payload": dict(self.payload), "provenance": self.provenance,
            "redaction_level": self.redaction_level,
        }


class EventStream:
    """Non-authoritative event collector; sink errors never affect the kernel."""

    def __init__(self, *, sink: Callable[[dict], None] | None = None, max_events: int = 1000, trace_id: str | None = None):
        if type(max_events) is not int or max_events < 1:
            raise ValueError("max_events must be a positive integer")
        self.sink = sink
        self.max_events = max_events
        self.trace_id = trace_id or uuid4().hex
        self.events: list[MaterializerEvent] = []
        self.dropped_events = 0
        self._sequence = 0

    def emit(self, event_type: str, **metadata) -> MaterializerEvent:
        if not event_type or not isinstance(event_type, str):
            raise ValueError("Event type must be a nonempty string")
        unexpected = set(metadata) - SAFE_FIELDS
        if unexpected:
            raise ValueError(f"Disallowed telemetry fields: {sorted(unexpected)}")
        for key, value in metadata.items():
            if value is not None and type(value) not in (str, int, float, bool):
                raise TypeError(f"Telemetry value for {key} must be scalar")
            if isinstance(value, str) and len(value) > 256:
                raise ValueError("Telemetry strings must be <= 256 characters")
            if isinstance(value, float) and not isfinite(value):
                raise ValueError("Telemetry values must be finite")
        self._sequence += 1
        event = MaterializerEvent(
            "1.0", f"{self.trace_id}:{self._sequence}", self.trace_id,
            datetime.now(timezone.utc).isoformat(), event_type, dict(metadata),
        )
        if len(self.events) == self.max_events:
            self.events.pop(0)
            self.dropped_events += 1
        self.events.append(event)
        if self.sink is not None:
            try:
                self.sink(event.to_dict())
            except Exception:
                # Observability must never determine computation or commit results.
                pass
        return event
