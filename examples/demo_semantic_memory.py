"""Manual dense semantic-memory smoke test with a real pretrained embedding model.

Install: python -m pip install sentence-transformers
Run: python -m examples.demo_semantic_memory --db semantic.sqlite3
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from ntsa_x.semantic_memory import SemanticMemory, SentenceTransformersEmbedder


def main():
    parser = argparse.ArgumentParser(description="Kindred persistent semantic memory demonstration")
    parser.add_argument("--db", default="absorbits-semantic.sqlite3")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    args = parser.parse_args()
    encoder = SentenceTransformersEmbedder(args.model)
    with SemanticMemory(Path(args.db), embedder=encoder, tenant="kindred-labs", writable=True) as memory:
        ident = memory.remember(
            "semantic", "Kindred Genesis is the public companion interface at KindredLabs.ai.",
            source="founder-defined-architecture", session="genesis",
        )
        matches = memory.recall_hits("Where can people talk to our AI companions?", session="genesis")
        print(json.dumps({
            "memory_id": ident,
            "model": encoder.model_id,
            "hits": [{"text": h.record.content, "score": round(h.similarity, 4),
                      "source": h.record.source, "verified": h.record.verified} for h in matches],
            "scope": memory.summary(session="genesis"),
        }, indent=2))


if __name__ == "__main__":
    main()
