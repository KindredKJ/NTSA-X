"""Versioned, fixed-function worker interface for the NTSA-X research prototype.

Workers are separate Python processes. They are NOT hardened sandboxes.
Only the repo's bundled arithmetic worker is called; no arbitrary code loading.
"""
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys
from typing import Mapping

from .counter_orbit import PairedOperator
from .events import EventStream


OPERATIONS = frozenset(("increment", "decrement", "identity", "negate"))


class PluginFailure(RuntimeError):
    pass


@dataclass(frozen=True)
class PluginManifest:
    plugin_id: str
    version: str
    operations: tuple[str, ...]
    capability: str = "compute.scalar/v1"

    def __post_init__(self):
        if not isinstance(self.plugin_id, str) or not 0 < len(self.plugin_id) <= 128:
            raise ValueError("Invalid plugin ID")
        if not isinstance(self.version, str) or not 0 < len(self.version) <= 64:
            raise ValueError("Invalid plugin version")
        if self.capability != "compute.scalar/v1" or not self.operations:
            raise ValueError("Unsupported worker capability")
        if len(set(self.operations)) != len(self.operations) or not set(self.operations) <= OPERATIONS:
            raise ValueError("Unsupported worker operations")


class PluginRegistry:
    def __init__(self):
        self._entries: dict[str, PluginManifest] = {}

    def register(self, manifest: PluginManifest):
        if not isinstance(manifest, PluginManifest):
            raise TypeError("Manifest required")
        prior = self._entries.get(manifest.plugin_id)
        if prior is not None and prior != manifest:
            raise ValueError("Immutable registration conflict")
        self._entries[manifest.plugin_id] = manifest

    def require(self, plugin_id: str, version: str, operation: str):
        item = self._entries.get(plugin_id)
        if item is None or item.version != version or operation not in item.operations:
            raise PermissionError("Plugin, version or operation denied")
        return item


class PluginWorker:
    def __init__(self, registry: PluginRegistry, *, timeout: float = 3.0, stream: EventStream | None = None):
        if not isinstance(registry, PluginRegistry):
            raise TypeError("Registry required")
        if type(timeout) not in (float, int) or not 0 < timeout <= 30:
            raise ValueError("Timeout must be between 0 and 30 seconds")
        self.registry, self.timeout, self.stream = registry, timeout, stream

    def invoke(self, plugin_id: str, version: str, operation: str, state: Mapping):
        try:
            self.registry.require(plugin_id, version, operation)
        except PermissionError:
            if self.stream:
                self.stream.emit("plugin.denied", operator=plugin_id[:128], reason="unregistered")
            raise
        if not isinstance(state, Mapping) or set(state) != {"value", "amount"}:
            raise ValueError("Worker requires value and amount only")
        if any(type(value) is not int for value in state.values()):
            raise ValueError("Worker requires integer inputs")
        if abs(state["value"]) > 1_000_000_000 or not 0 <= state["amount"] <= 1_000_000:
            raise ValueError("Worker input out of range")
        data = json.dumps({"operation": operation, "state": dict(state)}, separators=(",", ":"))
        worker = str(Path(__file__).with_name("plugin_worker.py"))
        try:
            result = subprocess.run(
                [sys.executable, worker], input=data,
                text=True, capture_output=True, timeout=self.timeout, check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise PluginFailure("Worker timed out") from exc
        if result.returncode or len(result.stdout) > 8192:
            raise PluginFailure("Worker exited unsuccessfully")
        try:
            response = json.loads(result.stdout)
        except ValueError as exc:
            raise PluginFailure("Worker returned malformed JSON") from exc
        if not isinstance(response, dict) or response.get("ok") is not True:
            raise PluginFailure("Worker rejected request")
        value = response.get("state")
        if not isinstance(value, dict) or set(value) != {"value"} or type(value["value"]) is not int:
            raise PluginFailure("Invalid worker response")
        if self.stream:
            self.stream.emit("plugin.invoke", operator=plugin_id[:128], verified=False)
        return value

    def paired_operator(self, plugin_id: str, version: str, *,
                        forward_op: str, reverse_op: str, axis=(1, 0, 0),
                        forward_amount=1, reverse_amount=1, estimated_cost=2.0):
        self.registry.require(plugin_id, version, forward_op)
        self.registry.require(plugin_id, version, reverse_op)

        def call(op, amount, state):
            if set(state) != {"value"}:
                raise ValueError("Paired worker accepts scalar state only")
            return self.invoke(plugin_id, version, op, {"value": state["value"], "amount": amount})

        return PairedOperator(
            name=f"{plugin_id}@{version}", axis=axis, estimated_cost=estimated_cost,
            forward=lambda state: call(forward_op, forward_amount, state),
            reverse=lambda state: call(reverse_op, reverse_amount, state),
        )
