"""Explicit-rights, document-disjoint local corpus preparation for COG-FU.

No automatic web downloads, no private file crawling, no implicit license
assumption. Documents are bounded, checked, and held out by *whole source*.
"""
from __future__ import annotations
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import json

from .cog_fu import CogFuConfig
from .native_model import ByteCodec


@dataclass(frozen=True)
class LanguageWindow:
    prefix: tuple[int, ...]
    continuation: tuple[int, ...]
    document_sha256: str


def _segments(text: str, byte_budget: int):
    """Partition text on Unicode scalar boundaries, not arbitrary raw bytes."""
    items = []
    current = []
    count = 0
    for char in text:
        size = len(char.encode("utf-8"))
        if count + size > byte_budget and current:
            items.append("".join(current))
            current, count = [], 0
        current.append(char)
        count += size
    if current:
        items.append("".join(current))
    return items


def _last_chars_under_budget(text: str, budget: int) -> str:
    result = []
    used = 0
    for char in reversed(text):
        length = len(char.encode("utf-8"))
        if used + length > budget:
            break
        result.append(char)
        used += length
    return "".join(reversed(result))


def build_windows(text: str, digest: str, config: CogFuConfig) -> list[LanguageWindow]:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Document must contain readable text")
    # Need at least one context byte, with a byte to be predicted.
    answer_budget = min(config.max_answer, config.max_context - 4)
    if answer_budget < 1:
        raise ValueError("Insufficient pretraining context")
    prefix_budget = config.max_context - answer_budget - 1
    chunks = _segments(text, answer_budget)
    output = []
    rolling_context = ""
    for index, chunk in enumerate(chunks):
        prior = _last_chars_under_budget(rolling_context, prefix_budget)
        prefix = (ByteCodec.BOS, *prior.encode("utf-8"))
        answer = (*chunk.encode("utf-8"),)
        if index == len(chunks) - 1 and len(answer) < config.max_answer:
            answer = (*answer, ByteCodec.EOS)
        if len(prefix) + len(answer) > config.max_context:
            raise RuntimeError("Internal window construction overflow")
        output.append(LanguageWindow(prefix, answer, digest))
        rolling_context = _last_chars_under_budget(
            prior + chunk, prefix_budget)
    return output


def load_corpus(manifest: str | Path, config: CogFuConfig, *,
                max_document_bytes: int = 1_000_000,
                max_total_bytes: int = 20_000_000):
    """Load only explicitly approved UTF-8 documents under manifest parent.

    Manifest shape:
    {"version":1,"documents":[{"path":"texts/a.txt",
       "license":"user-owned-original","rights_confirmed":true},...]}.
    The caller supplies *actual* rights, not merely a filename.
    """
    path = Path(manifest).expanduser().resolve()
    root = path.parent
    info = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(info, dict) or set(info) != {"version", "documents"}
            or info["version"] != 1 or
            not isinstance(info["documents"], list) or
            len(info["documents"]) < 2 or len(info["documents"]) > 1000):
        raise ValueError("Expected a v1 manifest with 2-1000 documents")
    documents = []
    content_seen = set()
    paths_seen = set()
    consumed = 0
    for row in info["documents"]:
        if (not isinstance(row, dict) or
                set(row) != {"path", "license", "rights_confirmed"} or
                row["rights_confirmed"] is not True or
                not isinstance(row["license"], str) or
                not row["license"].strip() or len(row["license"]) > 200 or
                not isinstance(row["path"], str) or not row["path"].strip()):
            raise ValueError("Each document requires path, explicit rights and license")
        candidate = (root / row["path"]).resolve()
        if (not candidate.is_relative_to(root) or
                candidate.suffix.lower() != ".txt" or
                not candidate.is_file()):
            raise ValueError("Only existing manifest-relative .txt files are allowed")
        if candidate in paths_seen:
            raise ValueError("Repeated manifest source")
        paths_seen.add(candidate)
        size = candidate.stat().st_size
        if not 32 <= size <= max_document_bytes:
            raise ValueError("Each document must be 32 bytes to 1 MB")
        consumed += size
        if consumed > max_total_bytes:
            raise ValueError("Corpus exceeds size budget")
        raw = candidate.read_bytes()
        value = raw.decode("utf-8", errors="strict")
        if not value.strip():
            raise ValueError("Empty document")
        digest = sha256(raw).hexdigest()
        if digest in content_seen:
            raise ValueError("Duplicate document content would contaminate holdout")
        content_seen.add(digest)
        documents.append((digest, value, row["license"], str(candidate.relative_to(root))))
    documents.sort(key=lambda row: row[0])
    holdout_count = max(1, len(documents) // 5)
    held = documents[:holdout_count]
    training = documents[holdout_count:]
    train_windows = [item for digest, text, _, _ in training
                     for item in build_windows(text, digest, config)]
    eval_windows = [item for digest, text, _, _ in held
                    for item in build_windows(text, digest, config)]
    if not train_windows or not eval_windows:
        raise ValueError("Training and holdout windows required")
    return train_windows, eval_windows, {
        "document_count": len(documents),
        "training_documents": len(training),
        "holdout_documents": len(held),
        "training_windows": len(train_windows),
        "holdout_windows": len(eval_windows),
        "training_document_sha256": [d[0] for d in training],
        "holdout_document_sha256": [d[0] for d in held],
        "sources": [{"sha256": d[0], "path": d[3], "license": d[2]}
                    for d in documents],
        "holdout_strategy": "sorted-document-sha256-first-20pct-min-1",
        "exact_duplicate_documents_rejected": True,
        "near_duplicate_documents_checked": False,
        "rights_attestations_are_user_supplied": True,
    }
