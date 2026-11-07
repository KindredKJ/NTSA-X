# NTSA-X — AbsOrBits Semantic Memory v0.2

**Research stage:** additive, private, opt-in secondary memory. Does not change the
existing tokenizer, weights, public API, production deployments, verified Nexus
convergence, or the current COG-FU checkpoints.

## Motivation and boundaries

The existing MemoryFabric supports typed working, episodic, semantic,
procedural and relational memories, but its retrieval ranking is **lexical**.
The new SemanticMemory stores model-generated document embeddings and retrieves
relevant records using query/document embedding cosine similarity.

Semantic similarity is useful context, **not proof that a fact is true** and
not a replacement for the independently revalidated exact convergence cache
in ConvergenceLibrary and IndexedCounterOrbitEngine.

## Flow

1. User/tool observation is assessed for future usefulness.
2. An explicitly writable SemanticMemory saves content + provenance + tenant,
   session, expiry and its document embedding in SQLite.
3. On a later query, semantic retrieval runs **before model inference**.
4. FusedRecallMemory combines semantic results with existing MemoryFabric
   records, keeping returned records inside the active session.
5. UnifiedCognitiveMechanism accepts FusedRecallMemory via its established
   recall/remember interface, incorporates bounded **unverified** memory notes,
   and then runs its existing recurrent generator.
6. Only the separate, independently reverified convergence index may skip
   computational execution. A semantic match never bypasses verification.

## Install

From an NTSA-X checkout (Python 3.12 or newer):

~~~bash
python -m pip install sentence-transformers
python -m unittest discover -s tests -p 'test_semantic_memory.py' -v
python -m examples.demo_semantic_memory --db semantic.sqlite3
~~~

The optional model weights are NOT bundled; the first use will download the
model from the selected model repository if the model is not already cached.
A pretrained model is required for actual language-level semantic quality.
CI's lightweight tests use **synthetic fixed embeddings** only to verify
correct retrieval plumbing, state persistence, isolation, gating and expiry.

## Attach to the current neural runtime

~~~python
from ntsa_x.memory_fabric import MemoryFabric
from ntsa_x.semantic_memory import SemanticMemory, FusedRecallMemory, SentenceTransformersEmbedder
from ntsa_x.unified_cognition import UnifiedCognitiveMechanism

primary = MemoryFabric("primary.sqlite3", writable=True)
secondary = SemanticMemory(
    "semantic.sqlite3",
    tenant="kindred-labs",
    embedder=SentenceTransformersEmbedder(),
    writable=True,
)
memory = FusedRecallMemory(primary, secondary)
memory.remember(
    "semantic",
    "Genesis Sanctuary is the public companion interface.",
    source="founder-defined-context",
    session="founder",
)

# model is an existing loaded and trained native COG-FU checkpoint;
# do not create an untrained checkpoint and call it intelligent.
runtime = UnifiedCognitiveMechanism(model, memory=memory, session="founder")
# runtime.generate(prefix, ...) will recall before recurrent generation.
~~~

AbsOrBits may separately learn adaptive parameters from retrieved features.
The current proposal wires **persistent retrieval to the NTSA-X memory port**;
it does not claim online neural-weight training, native multimodal embeddings,
production semantic accuracy or autonomous planning as completed.

## Storage model and operating notes

- SQLite/WAL, exact scope (tenant + session + embedding-model identity).
- Bounded content (4 KB), source attribution, optional TTL, deduplication.
- Optional retention gate: caller supplies importance, durability, confidence.
  They are assessments, not verified probability calibration.
- Default 20,000-item bounded cosine scan; larger memory requires a reviewed
  approximate-nearest-neighbor extension, migration and performance tests.
- Raw records and vectors are **not encrypted**: restrict file access and
  do not store passwords, API keys or private financial/medical information.
- For multiple customers, use separately isolated databases and audited access
  controls. A query embedding model may have its own license/deployment costs.
- Changes in the embedding model ID cause index separation; data must be
  explicitly re-embedded to migrate old records.
- The fused port does not modify MemoryFabric's original cross-session recall
  behavior; it filters records at the fusion layer.

## Verification status

The isolated implementation and tests ran locally with a synthetic embedder.
Full GitHub Actions results and real-model semantic-retrieval benchmarks must
be inspected independently before promoting beyond research staging.

Reference: https://www.sbert.net/examples/sentence_transformer/applications/semantic-search/README.html
