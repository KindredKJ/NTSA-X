"""Versioned, local Kindred integration envelopes. No network connections.

Contracts describe integration points for Strata, Zero Point, UMR/uDECA,
Kindred Root, Brainstem, and Materializer. They do not prove those systems
have been deployed or connected to NTSA-X.
"""
from dataclasses import dataclass
from math import isfinite
from typing import Mapping

from .events import SAFE_FIELDS


@dataclass(frozen=True)
class PortContract:
    port_id: str
    schema_version: str
    direction: str
    capabilities: tuple[str, ...]

    def __post_init__(self):
        if self.schema_version != "kindred.ntsa/1" or self.direction not in ("read", "observe"):
            raise ValueError("Unknown or write-capable Kindred contract")
        if not self.port_id or not self.capabilities:
            raise ValueError("Port requires identity and capabilities")


DEFAULT_PORTS = (
    PortContract("kindred.zero-point", "kindred.ntsa/1", "read", ("verified-recall-reference",)),
    PortContract("kindred.umr-udeca", "kindred.ntsa/1", "observe", ("orbit-status",)),
    PortContract("kindred.strata", "kindred.ntsa/1", "observe", ("metadata-transport",)),
    PortContract("kindred.root", "kindred.ntsa/1", "observe", ("commit-digest",)),
    PortContract("kindred.brainstem", "kindred.ntsa/1", "observe", ("plugin-audit-metadata",)),
    PortContract("kindred.materializer", "kindred.ntsa/1", "observe", ("orbit-telemetry",)),
)


class KindredContracts:
    """Build typed envelopes for an explicitly chosen local consumer.

    Packet creation has no side effects. A future remote transport MUST apply
    authenticated authorization, redaction, replay protection and encryption.
    """

    def __init__(self, ports=DEFAULT_PORTS):
        self._ports = {}
        for contract in ports:
            if not isinstance(contract, PortContract) or contract.port_id in self._ports:
                raise ValueError("Duplicate or invalid port contract")
            self._ports[contract.port_id] = contract

    def contract(self, port_id):
        try:
            return self._ports[port_id]
        except KeyError as exc:
            raise PermissionError("Unknown Kindred target port") from exc

    def wrap_event(self, port_id: str, event: Mapping) -> dict:
        contract = self.contract(port_id)
        if contract.direction != "observe":
            raise PermissionError("Read-only recall port cannot receive telemetry")
        if not isinstance(event, Mapping) or event.get("schema_version") != "1.0":
            raise ValueError("Incompatible NTSA-X event schema")
        if event.get("redaction_level") != "metadata":
            raise ValueError("Unredacted events are not permitted")
        payload = event.get("payload")
        if not isinstance(payload, dict) or set(payload) - SAFE_FIELDS:
            raise ValueError("Unsafe event payload")
        for value in payload.values():
            if value is not None and type(value) not in (str, int, float, bool):
                raise ValueError("Event values must be metadata scalars")
            if isinstance(value, str) and len(value) > 256:
                raise ValueError("Event metadata exceeds length limit")
            if type(value) is float and not isfinite(value):
                raise ValueError("Non-finite event metadata")
        for key in ("event_id", "trace_id", "timestamp", "type"):
            if not isinstance(event.get(key), str) or not event[key] or len(event[key]) > 256:
                raise ValueError("Malformed event metadata")
        # Explicitly retain only approved fields; do not forward unknown extras.
        return {
            "schema_version": contract.schema_version,
            "target": port_id,
            "mode": "observe",
            "event": {
                "schema_version": "1.0",
                "event_id": event["event_id"], "trace_id": event["trace_id"],
                "timestamp": event["timestamp"], "type": event["type"],
                "payload": dict(payload),
                "redaction_level": "metadata",
            },
        }

    def wrap_recall_ref(self, digest: str) -> dict:
        contract = self.contract("kindred.zero-point")
        if (not isinstance(digest, str) or len(digest) != 64
            or any(ch not in "0123456789abcdef" for ch in digest)):
            raise ValueError("Recall reference must be a canonical SHA-256 digest")
        return {
            "schema_version": contract.schema_version,
            "target": contract.port_id,
            "mode": "read",
            "reference": {"algorithm": "sha256", "digest": digest},
        }
