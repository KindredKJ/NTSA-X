"""Regression tests for the opt-in local streaming conversation bridge."""
from contextlib import closing
from http.client import HTTPConnection
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread
from unittest import TestCase
from unittest.mock import patch
from http.server import ThreadingHTTPServer
import json
import socket

from ntsa_x.live_gateway import (
    GatewayError, LiveService, LocalInference, TurnStore, make_handler,
)


class FakeInference:
    def __init__(self, chunks=("hello", " world"), failure=False):
        self.chunks = chunks
        self.failure = failure
        self.messages = None

    def stream(self, messages):
        self.messages = messages
        yield from self.chunks
        if self.failure:
            raise GatewayError("Synthetic provider failure")


class GatewayTests(TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.db = str(Path(self.temp.name) / "conversations.sqlite")
        self.store = TurnStore(self.db)

    def tearDown(self):
        self.temp.cleanup()

    def test_verified_nexus_never_receives_unverified_dialogue(self):
        service = LiveService(self.store, FakeInference())
        self.assertEqual("".join(service.stream("session_0001", "hello")),
                         "hello world")
        self.assertEqual(
            self.store.history("session_0001"),
            [{"role": "user", "content": "hello"},
             {"role": "assistant", "content": "hello world"}],
        )
        with self.store._connect() as conn:
            self.assertEqual(conn.execute(
                "SELECT DISTINCT verification FROM dialogue").fetchall(),
                [("unverified",)])
        self.assertFalse((Path(self.temp.name) / "nexus.db").exists())

    def test_context_is_scoped_to_session_and_persists_on_reopen(self):
        first = FakeInference()
        list(LiveService(self.store, first).stream("session_0001", "one"))
        second = FakeInference()
        list(LiveService(TurnStore(self.db), second).stream("session_0002", "two"))
        self.assertEqual([m["content"] for m in second.messages],
                         [second.messages[0]["content"], "two"])
        third = FakeInference()
        list(LiveService(TurnStore(self.db), third).stream("session_0001", "again"))
        self.assertEqual([m["role"] for m in third.messages],
                         ["system", "user", "assistant", "user"])

    def test_failure_is_atomic_and_retry_is_possible(self):
        service = LiveService(self.store, FakeInference(failure=True))
        with self.assertRaises(GatewayError):
            list(service.stream("session_0001", "message"))
        self.assertEqual(self.store.history("session_0001"), [])
        service.inference = FakeInference()
        self.assertEqual("".join(service.stream("session_0001", "message")),
                         "hello world")

    def test_concurrent_same_session_is_blocked_until_finished(self):
        service = LiveService(self.store, FakeInference())
        generator = service.stream("session_0001", "first")
        self.assertEqual(next(generator), "hello")
        with self.assertRaisesRegex(GatewayError, "active response"):
            list(service.stream("session_0001", "second"))
        generator.close()
        self.assertEqual(self.store.history("session_0001"), [])
        self.assertEqual("".join(service.stream("session_0001", "third")),
                         "hello world")

    def test_invalid_ids_input_and_clear(self):
        service = LiveService(self.store, FakeInference())
        for session in ("../bad", "short", "a" * 65, "abc defgh"):
            with self.assertRaises(GatewayError):
                list(service.stream(session, "hello"))
        for prompt in ("", "  ", "x" * 4001):
            with self.assertRaises(GatewayError):
                list(service.stream("session_0001", prompt))
        list(service.stream("session_0001", "valid"))
        self.store.clear("session_0001")
        self.assertEqual(self.store.history("session_0001"), [])

    def test_oversized_output_does_not_commit(self):
        service = LiveService(self.store, FakeInference(chunks=("x" * 16001,)))
        with self.assertRaisesRegex(GatewayError, "limit"):
            list(service.stream("session_0001", "hello"))
        self.assertEqual(self.store.history("session_0001"), [])

    def test_backend_rejects_remote_urls(self):
        for endpoint in (
            "https://127.0.0.1:8080/v1/chat/completions",
            "http://8.8.8.8:8080/v1/chat/completions",
            "http://localhost:8080/redirect",
            "http://localhost:8080/v1/chat/completions?forward=1",
            "http://user:pass@localhost:8080/v1/chat/completions",
        ):
            with self.assertRaises(ValueError, msg=endpoint):
                LocalInference(endpoint, "real-model")

    def test_backend_parses_openai_sse(self):
        frames = [
            b'data: {"choices":[{"delta":{"content":"Hi"}}]}\n', b'\n',
            b'data: {"choices":[{"delta":{"content":" there"}}]}\n', b'\n',
            b'data: [DONE]\n', b'\n',
        ]

        class Stream:
            def __enter__(self): return self
            def __exit__(self, *_): return False
            def __iter__(self): return iter(frames)

        model = LocalInference(
            "http://127.0.0.1:8080/v1/chat/completions", "real-model")
        class Opener:
            def open(self, *_args, **_kwargs):
                return Stream()

        with patch("ntsa_x.live_gateway.build_opener", return_value=Opener()):
            self.assertEqual("".join(model.stream([{"role": "user", "content": "hi"}])),
                             "Hi there")

    def test_http_stream_and_origin_restrictions(self):
        # Bind a local dynamic port for deterministic, credential-free test traffic.
        with closing(socket.socket()) as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        service = LiveService(self.store, FakeInference())
        server = ThreadingHTTPServer(
            ("127.0.0.1", port), make_handler(service, port=port)
        )
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            conn = HTTPConnection("127.0.0.1", port, timeout=5)
            body = json.dumps({"session_id": "session_0001", "message": "hello"})
            conn.request("POST", "/api/chat", body=body,
                         headers={"Content-Type": "application/json"})
            response = conn.getresponse()
            self.assertEqual(response.status, 200)
            wire = response.read().decode("utf-8")
            self.assertIn("event: delta", wire)
            self.assertIn('"saved": true', wire)
            conn.close()

            conn = HTTPConnection("127.0.0.1", port, timeout=5)
            conn.request("POST", "/api/clear",
                         body=json.dumps({"session_id": "session_0001"}),
                         headers={"Content-Type": "application/json",
                                  "Origin": "https://untrusted.example"})
            response = conn.getresponse()
            self.assertEqual(response.status, 403)
            response.read()
            conn.close()
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


if __name__ == "__main__":
    import unittest
    unittest.main()
