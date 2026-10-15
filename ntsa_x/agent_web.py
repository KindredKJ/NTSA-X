"""Optional public-web research and read-only browser adapters.

Both tools require explicit operator opt-in in agent_workbench. External web
content is untrusted input. They are NOT general sandboxed internet access.
"""
from __future__ import annotations

from html.parser import HTMLParser
from http.client import HTTPSConnection
from ipaddress import ip_address
from urllib.parse import urlsplit
import json
import socket
import ssl


class WebAccessError(ValueError):
    pass


def public_https_url(url: str, allowed_hosts: tuple[str, ...]) -> str:
    if not isinstance(url, str) or len(url) > 2048 or any(
            ord(c) < 32 or ord(c) == 127 for c in url):
        raise WebAccessError("Invalid URL")
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError as exc:
        raise WebAccessError("Malformed URL") from exc
    host = parsed.hostname
    if (parsed.scheme != "https" or not host or host not in allowed_hosts
            or parsed.username is not None or parsed.password is not None
            or parsed.fragment or port not in (None, 443)
            or url.startswith("https://.") or "\\" in url):
        raise WebAccessError("HTTPS URL must match an explicitly allowed exact hostname")
    try:
        ip_address(host)
    except ValueError:
        pass
    else:
        raise WebAccessError("IP-address URL targets are forbidden")
    if host == "localhost" or host.endswith(".localhost") or host.endswith(".local"):
        raise WebAccessError("Local network targets are forbidden")
    return url


def _public_addresses(hostname: str) -> list[str]:
    try:
        addresses = socket.getaddrinfo(hostname, 443, type=socket.SOCK_STREAM)
    except OSError as exc:
        raise WebAccessError("Could not resolve allowed website") from exc
    distinct = []
    for info in addresses:
        value = info[4][0]
        parsed = ip_address(value)
        # Reject the ENTIRE hostname if ANY DNS answer is non-global.
        if not parsed.is_global:
            raise WebAccessError("DNS result is non-public or reserved")
        if value not in distinct:
            distinct.append(value)
    if not distinct:
        raise WebAccessError("No public DNS answers")
    return distinct


class _PinnedHTTPS(HTTPSConnection):
    def __init__(self, hostname: str, ip: str):
        super().__init__(hostname, timeout=10,
                         context=ssl.create_default_context())
        self._pinned_ip = ip

    def connect(self):
        # Prevent a DNS rebind between validation and TCP connection.
        sock = socket.create_connection((self._pinned_ip, 443), self.timeout)
        try:
            self.sock = self._context.wrap_socket(
                sock, server_hostname=self.host)
        except BaseException:
            sock.close()
            raise


class _Text(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "noscript", "svg"):
            self.skip += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style", "noscript", "svg") and self.skip:
            self.skip -= 1

    def handle_data(self, text):
        if not self.skip:
            self.parts.append(text)


def fetch_text(url: str, allowed_hosts: tuple[str, ...]) -> dict:
    """TLS-verified, DNS-pinned, no redirects, bounded text GET."""
    url = public_https_url(url, allowed_hosts)
    uri = urlsplit(url)
    destinations = _public_addresses(uri.hostname)
    conn = _PinnedHTTPS(uri.hostname, destinations[0])
    try:
        path = uri.path or "/"
        if uri.query:
            path += "?" + uri.query
        conn.request("GET", path, headers={
            "Host": uri.hostname, "Accept": "text/html, text/plain, application/json",
            "Accept-Encoding": "identity", "User-Agent": "NTSA-X-Research/0.1",
            "Connection": "close"})
        response = conn.getresponse()
        if response.status != 200:
            raise WebAccessError(f"Fetch rejected HTTP status {response.status}")
        mime = response.getheader("Content-Type", "").split(";")[0].lower()
        if mime not in ("text/html", "text/plain", "application/json"):
            raise WebAccessError("Unexpected binary or unsupported response type")
        data = response.read(100_001)
        if len(data) > 100_000:
            raise WebAccessError("Remote response exceeds 100 KB limit")
        text = data.decode("utf-8", errors="replace")
        if mime == "text/html":
            parser = _Text()
            parser.feed(text)
            text = " ".join(" ".join(parser.parts).split())
        return {"url": url, "content_type": mime, "text": text[:12_000],
                "truncated": len(text) > 12_000,
                "untrusted_external_content": True}
    except (OSError, ssl.SSLError) as exc:
        raise WebAccessError("Public HTTPS request failed") from exc
    finally:
        conn.close()


def browser_snapshot(url: str, allowed_hosts: tuple[str, ...]) -> dict:
    """Read-only DOM snapshot. Optional Playwright Chromium is not bundled.

    Chromium is NOT a hardened sandbox; run in an isolated OS user/session.
    All subresource requests to unapproved origins are blocked. Click, fill,
    login, downloads, permissions and persistent profiles are not supported.
    """
    url = public_https_url(url, allowed_hosts)
    # Browser navigations still need a trusted DNS/network sandbox outside
    # Python. Verify the hostname is public immediately before navigation.
    _public_addresses(urlsplit(url).hostname)
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise WebAccessError(
            "Optional Playwright missing. Install playwright and Chromium explicitly"
        ) from exc

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            context = browser.new_context(
                accept_downloads=False,
                service_workers="block",
                permissions=[],
                java_script_enabled=True)
            def gate(route):
                try:
                    public_https_url(route.request.url, allowed_hosts)
                except WebAccessError:
                    route.abort()
                else:
                    route.continue_()
            context.route("**/*", gate)
            page = context.new_page()
            response = page.goto(url, wait_until="domcontentloaded", timeout=15_000)
            if response is None or response.status >= 400:
                raise WebAccessError("Website navigation returned an error")
            title = page.title()[:300]
            visible = page.locator("body").inner_text(timeout=8_000)
            return {"url": page.url, "title": title, "text": visible[:12_000],
                    "truncated": len(visible) > 12_000,
                    "untrusted_external_content": True,
                    "read_only_browser_session": True}
        finally:
            browser.close()
