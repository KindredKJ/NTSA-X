# Milestone 16 — Browser-backed live web recall

**Purpose:** Add on-demand web search, freshly fetched source excerpts, UTC
retrieval timestamps and source links to the **existing** NTSA-X M15
recall-before-inference pipeline. No neural layers or model weights change.

This is an independently built implementation of the *capability pattern* of
live search and citations; it is not the proprietary ChatGPT search backend.
It is not automatically unlimited: upstream search engines, sites and the
host machine have throughput, availability, policy and resource constraints.

## Data flow

User explicitly asks to "search the web for ..." (or asks about latest news)
→ existing M15 native extension chooses retrieval
→ fresh ephemeral **Playwright Chromium** browser opens **local SearXNG** result page
→ result URLs are filtered to public HTTPS
→ original pages are opened through the existing DNS-pinned, TLS-checked,
bounded, no-redirect text fetcher
→ source URLs, UTC retrieval times and SHA-256 excerpt digests recorded
→ excerpts enter the **same** recurrent cognitive input as existing recall
→ model runs normal unified inference
→ separately emitted "Sources retrieved" links, **not** unearned verification.

Retrieval timestamps are **not publication dates**. Results are unverified
external data, NOT trusted instructions. M15 Nexus verification, explicit
memory writes and human effect permissions remain unchanged. This feature
does NOT train the model or install a general-purpose browser agent.

## Install a free, self-hosted browser search front end

On Windows, with Docker Desktop and Python 3.12:

```powershell
Set-Location -LiteralPath (Join-Path $HOME "KindredLabs\NTSA-X-GitHub")
py -3.12 -m pip install playwright
py -3.12 -m playwright install chromium
docker run -d --name ntsax-search --restart unless-stopped -p 127.0.0.1:8888:8080 docker.io/searxng/searxng:latest
```

Open `http://127.0.0.1:8888/` in your browser and check that SearXNG returns
search results. This is your local search interface, not a paid Google API.
SearXNG can be configured to query Google among its external search providers,
subject to Google's and each provider's limits and terms. Do **not** assume
unlimited Google access. For commercial deployment, pin the container image
to a tested immutable digest, place strict network egress controls around
browser workloads, and scale the search/search-cache services independently.

Preview real source receipts **without a running CogFu model**:

```powershell
py -3.12 -m ntsa_x.web_recall --query "NTSA architecture 2026" --max-pages 2
py -3.12 -m unittest discover -s tests -p test_web_recall.py -v
```

The CLI returns JSON with `sources[].url`, `sources[].retrieved_at_utc`,
`sources[].sha256`, `sources[].excerpt`, and unverified status flags. It
never stores retrieved content in the Nexus proof ledger or automatically
updates model weights.

## Equip the existing native chat runtime

Use the real on-disk path for a current CogFu checkpoint. This example uses
the previously documented local `.local/cog-fu-language-v1` checkpoint:

```powershell
py -3.12 -m ntsa_x.cog_fu_server `
    --checkpoint .local/cog-fu-language-v1 `
    --model-id ntsa-cog-fu-web-v1 `
    --decoder unified `
    --enable-web-recall `
    --web-search-port 8888 `
    --max-new-tokens 48 `
    --context-limit 2048 `
    --port 8081
```

Second terminal:

```powershell
py -3.12 -m ntsa_x.live_gateway `
    --student-model ntsa-cog-fu-web-v1 `
    --student-max-tokens 48 `
    --student-endpoint http://127.0.0.1:8081/v1/chat/completions `
    --use-student
```

Open `http://127.0.0.1:8765` and type:

```
search the web for the latest NTSA-X research
```

The same bounded M15 unified cognition loop receives source-conditioned
context. It can also append a machine-generated list of **retrieved sources**,
but source presence does not prove that generated claims are supported. The
tiny language checkpoint may not generate good English; that requires
training and evaluation, not a search integration alone.

To restrict source retrieval to selected authoritative sites, repeat
`--web-source-host docs.python.org` for each exact allowed domain.
When no source hosts are supplied, only public HTTPS hosts from SearXNG
results are eligible; SSRF defenses still reject localhost, IP literals,
non-public DNS resolutions, redirections and oversized pages.

## Limits and safety

- Web browsing is explicitly disabled unless `--enable-web-recall` is passed.
- Only explicit web-search phrases and narrow news freshness phrases trigger
  a query. Normal conversations, personal queries, secret files and financial
  details are not automatically sent to any search engine.
- The ephemeral browser contacts **only the local loopback SearXNG front end**.
  The separate public page fetch uses pinned DNS and rejects redirects.
- No cookie jar, login, browser profile, permission grant, arbitrary click,
  form submission or remote script execution is enabled.
- Search results may contain malicious prompt injections. They remain tagged
  as untrusted context; the model is not allowed to authorize external effects.
- Browser subprocess isolation and containerized network egress controls
  are still required before public deployment. SearXNG relies on upstream
  providers; there is no guaranteed unmetered web index.
- This is not proven real-time *truth*, learning, autonomous research planning,
  or a replacement for empirical model quality benchmarks.

Useful upstream references:

- https://docs.searxng.org/dev/search_api.html
- https://docs.searxng.org/admin/installation-docker
- https://playwright.dev/python/docs/network
