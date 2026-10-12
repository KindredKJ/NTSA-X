"""Local-only read-only Materializer event bridge and live SSE viewer.

The bridge does not issue commands, perform remote writes, or authenticate users.
It is deliberately bound to loopback and must not be exposed through a proxy.
"""
from collections import deque
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from queue import Empty, Full, Queue
from threading import RLock, Thread

from .events import SAFE_FIELDS


class LocalMaterializer:
    def __init__(self, *, host="127.0.0.1", port=0, history=512, queue_size=128):
        if host not in ("127.0.0.1", "::1", "localhost"):
            raise ValueError("Materializer must bind only to loopback")
        if type(history) is not int or history < 1 or type(queue_size) is not int or queue_size < 1:
            raise ValueError("Buffer limits must be positive integers")
        self._lock = RLock()
        self._history = deque(maxlen=history)
        self._queues: set[Queue] = set()
        self.queue_size = queue_size
        self.dropped = 0
        self._thread = None
        bridge = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, fmt, *args):
                pass

            def _bytes(self, content: bytes, mime: str):
                self.send_response(200)
                self.send_header("Content-Type", mime)
                self.send_header("Content-Length", str(len(content)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; object-src 'none'; base-uri 'none'")
                self.end_headers()
                self.wfile.write(content)

            def do_GET(self):
                route = self.path.split("?", 1)[0]
                assets = {
                    "/": ("materializer.html", "text/html; charset=utf-8"),
                    "/app.js": ("materializer.js", "application/javascript; charset=utf-8"),
                    "/style.css": ("materializer.css", "text/css; charset=utf-8"),
                }
                if route in assets:
                    name, mime = assets[route]
                    self._bytes((Path(__file__).parent / name).read_bytes(), mime)
                elif route == "/api/events":
                    with bridge._lock:
                        payload = {"events": list(bridge._history), "dropped": bridge.dropped}
                    self._bytes(json.dumps(payload, separators=(",", ":")).encode("utf-8"), "application/json")
                elif route == "/health":
                    self._bytes(b'{"status":"ok","scope":"loopback"}', "application/json")
                elif route == "/events":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache, no-store")
                    self.send_header("Connection", "keep-alive")
                    self.send_header("X-Accel-Buffering", "no")
                    self.end_headers()
                    subscriber = Queue(maxsize=bridge.queue_size)
                    with bridge._lock:
                        bridge._queues.add(subscriber)
                    try:
                        self.wfile.write(b": connected\n\n")
                        self.wfile.flush()
                        while True:
                            try:
                                item = subscriber.get(timeout=3)
                                self.wfile.write(b"data: " + item + b"\n\n")
                            except Empty:
                                self.wfile.write(b": keepalive\n\n")
                            self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        pass
                    finally:
                        with bridge._lock:
                            bridge._queues.discard(subscriber)
                        self.close_connection = True
                else:
                    self.send_error(404, "Not found")

        self._http = ThreadingHTTPServer((host, port), Handler)
        self.host, self.port = self._http.server_address[:2]

    @property
    def url(self):
        return f"http://{self.host}:{self.port}/"

    def start(self):
        if self._thread is None:
            self._thread = Thread(target=self._http.serve_forever, name="ntsa-materializer", daemon=True)
            self._thread.start()
        return self

    def publish(self, envelope: dict):
        """Strictly validate metadata envelope before broadcast to local viewers."""
        if not isinstance(envelope, dict) or envelope.get("schema_version") != "1.0" or envelope.get("redaction_level") != "metadata":
            raise ValueError("Only schema v1 metadata events are accepted")
        payload = envelope.get("payload")
        if not isinstance(payload, dict) or set(payload).difference(SAFE_FIELDS):
            raise ValueError("Payload has forbidden fields")
        if any(type(v) not in (int, float, bool, str, type(None)) for v in payload.values()):
            raise ValueError("Payload values must be scalars")
        if any(isinstance(v, str) and len(v) > 256 for v in payload.values()):
            raise ValueError("Payload strings must be bounded")
        for field in ("type", "event_id", "trace_id", "timestamp"):
            v = envelope.get(field)
            if not isinstance(v, str) or not v or len(v) > 256:
                raise ValueError(f"Malformed {field}")
        packet = {
            "schema_version": "1.0", "event_id": envelope["event_id"],
            "trace_id": envelope["trace_id"], "timestamp": envelope["timestamp"],
            "type": envelope["type"], "payload": deepcopy(payload),
            "provenance": "ntsa-x", "redaction_level": "metadata",
        }
        wire = json.dumps(packet, separators=(",", ":"), allow_nan=False).encode("utf-8")
        with self._lock:
            self._history.append(packet)
            for subscriber in tuple(self._queues):
                try:
                    subscriber.put_nowait(wire)
                except Full:
                    self.dropped += 1
                    # Drop slow subscribers rather than ever blocking the kernel.
                    self._queues.discard(subscriber)

    def close(self):
        self._http.shutdown()
        self._http.server_close()
        if self._thread is not None:
            self._thread.join(timeout=3)

    def __enter__(self):
        return self.start()

    def __exit__(self, *_exc):
        self.close()
