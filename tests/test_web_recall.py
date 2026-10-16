"""Deterministic M16 browser web recall tests: no live requests or browser needed."""
from __future__ import annotations

from hashlib import sha256
from unittest import TestCase
from unittest.mock import patch

from ntsa_x.web_recall import (
    BrowserWebRecall, SearchHit, WebAccessError, parse_search_html, query_for_web)
from ntsa_x.unified_cognition import UnifiedCognitiveMechanism, web_recall_extension

BOS, USER, EOS, ASSISTANT = 256, 259, 258, 260


class DummyModel:
    config = type("Config", (), {"max_context": 2048, "max_answer": 48})()

    def __init__(self):
        self.prefix = []

    def unified_draft(self, prefix, **_kw):
        self.prefix = prefix
        return list(b"Evidence retrieved; check the sources.")


def prefix(text):
    return [BOS, USER, *text.encode("utf-8"), EOS, ASSISTANT]


class WebRecallTests(TestCase):
    def test_parse_sanitizes_urls_and_duplicates(self):
        html = """<article class='result'><h3><a href='https://example.org/a'>Example A</a></h3>
        <p class='content'>Trusted looking description</p></article>
        <article class='result'><h3><a href='https://example.org/a'>Again</a></h3></article>
        <article class='result'><h3><a href='http://127.0.0.1/private'>Local</a></h3></article>
        <article class='result'><h3><a href='https://user:password@example.org/'>Bad</a></h3></article>
        <article class='result'><h3><a href='https://example.com:9999/'>Bad port</a></h3></article>"""
        hits = parse_search_html(html)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].url, "https://example.org/a")
        self.assertEqual(hits[0].snippet, "Trusted looking description")

    def test_browser_configuration_fail_closed(self):
        for port in (0, 100, 65536):
            with self.assertRaises(ValueError):
                BrowserWebRecall(search_port=port)
        with self.assertRaises(ValueError):
            BrowserWebRecall(allowed_source_hosts=("127.0.0.1",), max_pages=9)

    def test_research_fetched_sources_not_claimed_verified(self):
        browser = BrowserWebRecall(max_pages=2)
        browser.search = lambda _q, **_kw: [
            SearchHit("Alpha", "https://example.org/a", "a"),
            SearchHit("Beta", "https://example.com/b", "b")]
        with patch("ntsa_x.web_recall.fetch_text", side_effect=[
                {"text": "Evidence A", "truncated": False},
                {"text": "Evidence B", "truncated": False}]):
            report = browser.research("example")
        self.assertEqual(len(report["sources"]), 2)
        self.assertEqual(report["sources"][0]["sha256"],
                         sha256(b"Evidence A").hexdigest())
        self.assertFalse(report["model_training"])
        self.assertEqual(report["sources"][1]["status"], "fetched_unverified")
        self.assertIn("retrieved_not_independently_verified",
                      report["evidence_status"])

    def test_only_explicit_intent_or_fresh_news(self):
        self.assertEqual(query_for_web("Search the web for PyTorch updates"),
                         "PyTorch updates")
        self.assertEqual(query_for_web("look up online current standards"),
                         "current standards")
        self.assertIsNone(query_for_web("summarize my confidential files"))
        self.assertIsNone(query_for_web("what is my current account balance?"))
        self.assertEqual(query_for_web("latest news on research"),
                         "latest news on research")

    def test_web_sources_enter_shared_state_not_model_weights(self):
        class FixedBrowser:
            def context(self, _query, *, max_bytes=1200):
                return "[1] Public evidence https://example.org/a (retrieved now)"
        model = DummyModel()
        engine = UnifiedCognitiveMechanism(
            model, extensions=(web_recall_extension(FixedBrowser()),))
        answer = bytes(engine.generate(prefix("search the web for example")))
        self.assertIn(b"Evidence retrieved", answer)
        conditioned = bytes(byte for byte in model.prefix if 0 <= byte < 256)
        self.assertIn(b"https://example.org/a", conditioned)
        self.assertIn("web.recall.native",
                      engine.last_receipt["native_extensions"])
        self.assertFalse(engine.last_receipt["verified_external_truth"])
        other = DummyModel()
        idle = UnifiedCognitiveMechanism(
            other, extensions=(web_recall_extension(FixedBrowser()),))
        bytes(idle.generate(prefix("my private question")))
        self.assertEqual(idle.last_receipt["native_extensions"], [])

    def test_public_source_allowlist(self):
        browser = BrowserWebRecall(
            allowed_source_hosts=("example.org",), max_pages=1)
        browser.search = lambda _q, **_kw: [
            SearchHit("Wrong host", "https://example.com/a", ""),
            SearchHit("Allowed", "https://example.org/b", "")]
        with patch("ntsa_x.web_recall.fetch_text",
                   return_value={"text": "Confirmed read"}) as fetch:
            report = browser.research("test")
        self.assertEqual([s["url"] for s in report["sources"]],
                         ["https://example.org/b"])
        fetch.assert_called_once()


if __name__ == "__main__":
    import unittest
    unittest.main()
