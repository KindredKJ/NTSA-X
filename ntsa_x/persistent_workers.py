"""Supervised reusable subprocesses for the repository's fixed scalar worker.

The pool is an in-process coordinator of trusted code, NOT a sandbox for arbitrary
untrusted plugins. It never loads third-party modules or executes shell commands.
Each leased process handles at most one request at a time. Timeouts/errors fail
that request and recycle the process; no silent retries or unverified commits.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from queue import Empty, Queue, SimpleQueue
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Lock, Thread
from uuid import uuid4

from .plugins import PluginFailure, PluginRegistry, PluginWorker
from .events import EventStream


@dataclass
class _Slot:
    process: subprocess.Popen
    inbox: SimpleQueue
    reader: Thread
    completed: int = 0


class PersistentPluginWorker(PluginWorker):
    """Thread-safe leased worker pool with bounded concurrency and recycling."""

    def __init__(
        self, registry: PluginRegistry, *, workers: int = 8,
        timeout: float = 3.0, startup_timeout: float = 5.0,
        max_tasks_per_worker: int = 1000,
        stream: EventStream | None = None,
    ):
        super().__init__(registry, timeout=timeout, stream=stream)
        if type(workers) is not int or not 1 <= workers <= 100:
            raise ValueError("workers must be in [1, 100]")
        if type(max_tasks_per_worker) is not int or max_tasks_per_worker < 1:
            raise ValueError("max_tasks_per_worker must be positive")
        if type(startup_timeout) not in (int, float) or not 0 < startup_timeout <= 30:
            raise ValueError("startup_timeout must be in (0,30]")
        self.workers = workers
        self.startup_timeout = startup_timeout
        self.max_tasks_per_worker = max_tasks_per_worker
        self._lock = Lock()
        self._idle = Queue(maxsize=workers)
        self._all = []
        self._temp = None
        self._started = False
        self._closed = False
        self._counts = {"starts": 0, "restarts": 0, "failures": 0, "requests": 0}

    @property
    def stats(self):
        with self._lock:
            return dict(self._counts)

    def _spawn(self):
        worker_file = str(Path(__file__).with_name("plugin_worker.py"))
        env = {"PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1"}
        for name in ("SystemRoot", "WINDIR"):
            if name in os.environ:
                env[name] = os.environ[name]
        process = subprocess.Popen(
            [sys.executable, "-I", "-u", worker_file, "--serve"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, cwd=self._temp.name, env=env,
        )
        inbox = SimpleQueue()

        def read_lines():
            try:
                while True:
                    # Bound each JSON response frame even if the worker misbehaves.
                    line = process.stdout.readline(8193)
                    if not line:
                        break
                    inbox.put(line)
            except (OSError, ValueError):
                pass
            finally:
                inbox.put(None)

        reader = Thread(target=read_lines, daemon=True, name="ntsa-worker-rpc")
        reader.start()
        slot = _Slot(process, inbox, reader)
        try:
            message = inbox.get(timeout=self.startup_timeout)
            if not message or len(message) > 8192:
                raise PluginFailure("Worker handshake missing or too large")
            hello = json.loads(message)
            if hello != {"ready": True, "protocol": "ntsa-fixed/v1"}:
                raise PluginFailure("Worker handshake protocol mismatch")
        except (Empty, ValueError, TypeError, PluginFailure) as exc:
            self._stop_slot(slot)
            raise PluginFailure("Worker startup failed") from exc
        with self._lock:
            self._counts["starts"] += 1
        return slot

    @staticmethod
    def _stop_slot(slot):
        process = slot.process
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2.0)
        for pipe in (process.stdin, process.stdout):
            if pipe is not None:
                try:
                    pipe.close()
                except (OSError, ValueError):
                    pass
        slot.reader.join(timeout=1.0)

    def start(self):
        with self._lock:
            if self._closed:
                raise PluginFailure("Worker pool closed")
            if self._started:
                return self
            # Only one thread may initialize and publish the pool.
            self._started = True
        try:
            self._temp = TemporaryDirectory(prefix="ntsa-fixed-pool-")
            for _ in range(self.workers):
                slot = self._spawn()
                self._all.append(slot)
                self._idle.put_nowait(slot)
        except BaseException:
            for slot in self._all:
                self._stop_slot(slot)
            self._all.clear()
            if self._temp:
                self._temp.cleanup()
                self._temp = None
            with self._lock:
                self._closed = True
            raise
        return self

    def _replace(self, slot):
        self._stop_slot(slot)
        self._all.remove(slot)
        with self._lock:
            self._counts["restarts"] += 1
        replacement = self._spawn()
        self._all.append(replacement)
        return replacement

    def invoke(self, plugin_id: str, version: str, operation: str, state):
        try:
            self.registry.require(plugin_id, version, operation)
        except PermissionError:
            if self.stream:
                self.stream.emit("plugin.denied", operator=str(plugin_id)[:128],
                                 reason="unregistered")
            raise
        if not isinstance(state, dict) or set(state) != {"value", "amount"}:
            raise ValueError("Worker requires value and amount only")
        if any(type(value) is not int for value in state.values()):
            raise ValueError("Worker requires integer inputs")
        if abs(state["value"]) > 1_000_000_000 or not 0 <= state["amount"] <= 1_000_000:
            raise ValueError("Worker input out of range")
        self.start()
        try:
            slot = self._idle.get(timeout=self.timeout)
        except Empty as exc:
            raise PluginFailure("No worker available within timeout") from exc
        failed = False
        try:
            if slot.process.poll() is not None:
                raise PluginFailure("Worker exited unexpectedly")
            request_id = uuid4().hex
            raw = json.dumps({
                "id": request_id, "operation": operation, "state": dict(state),
            }, separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n"
            if len(raw) > 8192:
                raise PluginFailure("Request frame too large")
            slot.process.stdin.write(raw)
            slot.process.stdin.flush()
            try:
                message = slot.inbox.get(timeout=self.timeout)
            except Empty as exc:
                raise PluginFailure("Worker response timeout") from exc
            if message is None or len(message) > 8192:
                raise PluginFailure("Worker response unavailable or too large")
            try:
                result = json.loads(message)
            except (ValueError, UnicodeDecodeError) as exc:
                raise PluginFailure("Invalid worker response JSON") from exc
            if (not isinstance(result, dict)
                or result.get("id") != request_id
                or result.get("ok") is not True):
                raise PluginFailure("Worker response rejected or ID mismatch")
            output = result.get("state")
            if (not isinstance(output, dict) or set(output) != {"value"}
                or type(output["value"]) is not int
                or abs(output["value"]) > 1_000_000_000):
                raise PluginFailure("Invalid worker output")
            slot.completed += 1
            with self._lock:
                self._counts["requests"] += 1
            if self.stream:
                self.stream.emit("plugin.invoke", operator=plugin_id[:128], verified=False)
            return output
        except (OSError, BrokenPipeError, ValueError, PluginFailure) as exc:
            failed = True
            with self._lock:
                self._counts["failures"] += 1
            raise PluginFailure("Persistent worker request failed") from exc
        finally:
            if failed or slot.completed >= self.max_tasks_per_worker:
                try:
                    slot = self._replace(slot)
                except (OSError, PluginFailure):
                    # The pool has lost a slot: make subsequent submissions fail
                    # rather than queue indefinitely or imply resilience.
                    with self._lock:
                        self._closed = True
                    slot = None
            if slot is not None:
                self._idle.put_nowait(slot)

    def close(self):
        # Must be called after work has finished, not concurrently with invoke().
        with self._lock:
            if self._closed and self._temp is None:
                return
            self._closed = True
        for slot in list(self._all):
            self._stop_slot(slot)
        self._all.clear()
        if self._temp:
            self._temp.cleanup()
            self._temp = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *_exc):
        self.close()
