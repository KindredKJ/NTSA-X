# NTSA-X

**Nexus Transforming Stream Architecture** — an experimental architecture and reference kernel for stateful, evidence-gated, adaptive computation.

> Research-stage project. This is **not** yet a trained foundational AI model or a verified transformer replacement. Do not treat claimed performance, energy savings, external proof, or security boundaries as established.

## Research threads

NTSA-01 Token-Stream · NTSA-02 Byte-to-Byte · NTSA-03 Nexus-Byte · NTSA-04 Byte/Bit-Stream · NTSA-05 Transforming Byte-Scaler · NTSA-06 Chain-Graph / Graph-to-Graph · **NTSA-X** independent recurrent stream machine.

Full lineage and scientific hypotheses: [Architecture](docs/ARCHITECTURE.md).

## Implemented in the reference kernel

- **Orbital scheduler** selects pure in-process operators using priority, changing predicates, remaining abstract cost budget, and bounded steps.
- **Three-axis position** tracks X duality, Y abstraction, and Z lineage coordinates as signed integer orbit deltas.
- **Convergence engine** requires explicit check categories (state, semantic, causal, constraint, evidence, delta), rejects missing checks and contradictions, and can veto on mismatched paired trajectories.
- **Recall-before-compute** rechecks canonical SHA-256 lineage, context matching, and current verification before reuse.
- **Verification-gated commits** only append to an in-memory hash-linked chain after a passing verifier report.
- **Materializer event stream** emits bounded metadata-only trace events through a non-authoritative consumer callback.

This is a prototype, not an authorization sandbox: arbitrary Python callbacks must be trusted. Verification check labels are attestations from an injected oracle, not cryptographically established external facts. The foundation above is complemented by later milestones, but trained-model validation, authenticated external evidence, production Materializer connectivity, and empirical benchmarks remain future work.

## Milestone 02 — Paired orbits + Nexus graph + live local Materializer

**Implemented in the research prototype:**

- Two coordinated forward/reverse computational paths, agreement checks, contradiction veto and a **shared bounded abstract compute budget**.
- **Persistent SQLite Nexus DAG** with multiple parents, canonical SHA-256 lineage verification, cross-session recall and transactional append-only application operations.
- **Local Materializer browser dashboard** with Server-Sent Events, read-only orbit traces, convergence/commit events and Nexus counters. The viewer is hosted only on loopback and is not a deployed integration with an existing Kindred Materializer system.

**Watch locally:**

```bash
python -m unittest discover -s tests -v
python -m examples.demo_counter_orbit --watch
```

Open the localhost URL printed by the process and follow the terminal prompt. See [Milestone 02 instructions](docs/MILESTONE-02.md).

This remains a research prototype; external-proof labels are verifier assertions, not independent attestation, and in-process callbacks are not sandboxed. Do not store secrets in this prototype.

## Milestone 03 — Verified reuse and capability-bounded workers

- `ConvergenceLibrary` builds an **additive**, immutable SHA-linked lookup index for exact-input, context-scoped and verifier-revision-scoped convergence reuse.
- `IndexedCounterOrbitEngine` checks the current evidence and both trajectory projections again before accepting prior results. It preserves the existing `CounterOrbitEngine` API.
- Higher-order convergence composition explicitly links multiple verified source nodes; composite results must undergo their own verification.
- Versioned fixed-function `PluginWorker` operations run in separate time-limited Python processes with registered capability manifests. These are **not hardened sandboxes** for untrusted code.
- `KindredContracts` defines local-only, metadata-based adapter packets for Zero Point, UMR/uDECA, Strata, Kindred Root, Brainstem, and Materializer. **No remote Kindred services have been connected.**

Run:

```bash
python -m unittest discover -s tests -v
python -m examples.demo_milestone03
```

Read [Milestone 03 documentation](docs/MILESTONE-03.md) for architecture, limitations, and security requirements.

## Milestone 04 — ×100 verified production-capacity initiative

- **Factory scheduler:** bounded, dependency-aware work batches; a 100-task example and up to 100 explicitly configured local worker slots (default 8).
- **No blind promotion:** each fixed-function worker result is independently recalculated by a local arithmetic oracle before an append-only, parent-linked Nexus commit. Failed tasks block dependent work.
- **Factory Materializer:** read-only local dashboard for queue progress, verified completions, worker peaks and measured task rate.
- **Measured benchmarks:** JSON reports compare fresh sequential and parallel 100-task batches. Windows and Linux CI compile source, run tests/demos, capture benchmark artifacts, and enforce a combined test gate.

**×100 is a goal, not a measured speedup.** Worker throughput varies by machine and workload; this prototype is neither a trained AI model nor a production plugin sandbox.

```bash
python -m examples.demo_factory --tasks 100 --workers 8 --watch
python -m examples.benchmark_factory --tasks 100 --workers 8 --compare
```

See [Milestone 04: Factory ×100](docs/MILESTONE-04-FACTORY-X100.md).

## Milestone 05 — Persistent Scalar

**Persistent Scalar** keeps a bounded number of trusted scalar worker processes warm, reuses them across tasks, verifies request/response IDs, and recycles or replaces failing workers. The existing one-shot API remains intact. It plugs into the verified factory without changing the Nexus ledger contract.

The 100-task benchmark compares **1, 4, 8, and 16 persistent workers** against the one-shot single-worker baseline, reporting absolute rates and variability on Windows and Linux. This is an optimization of synthetic fixed-function task processing, **not** a claim of 100× AI-model acceleration.

```bash
python -m examples.benchmark_persistent_scalar --tasks 100 --workers 1,4,8,16 --repeats 3
```

See [Persistent Scalar technical specification](docs/MILESTONE-05-PERSISTENT-SCALAR.md).

## Milestone 06 — Local live conversation bridge

**New, experimental:** a streaming conversational interface that connects to an already running **local** OpenAI-compatible language-model server. Session history is stored in a separate SQLite conversation database; model replies are explicitly unverified and are **never** inserted into the verified Nexus memory chain. The browser UI supports streamed text and optional browser-mediated dictation/speech synthesis (browser speech recognition may use external services). This adapter does not convert NTSA-X into a trained conversational model or connect UMR Kindred.

Run your inference server at http://127.0.0.1:8080 and from the repository root:

```powershell
py -3.12 -m unittest discover -s tests -v
py -3.12 -m ntsa_x.live_gateway --model YOUR_ACTUAL_LOCAL_MODEL_ID
```

Open http://127.0.0.1:8765. The gateway binds to loopback only; do not expose it publicly without a separate authenticated, TLS-protected and reviewed production service. See [Milestone 06 guide](docs/MILESTONE-06-LIVE-CONVERSATION.md).

## Milestone 07 — Recall-first teacher-to-student learning

The local conversation gateway now supports **exact reviewed recall before
teacher inference**, opt-in collection of teacher responses, single-sample
approval with explicit training rights, prompt-disjoint dataset export, and
real LoRA student parameter training (optional PyTorch/Transformers/PEFT).
A separately hosted student can be selected in **teacher-free mode** with
`--student-model kindred-student-v1 --use-student`, which never contacts the
teacher. Model output is not independently verified; neither caching nor
student fine-tuning constitutes training of the native NTSA-X reasoning kernel.

**This is an experimental teacher-retirement pathway, not yet autonomous
online learning.** Read [Milestone 07](docs/MILESTONE-07-RECALL-TEACHER-STUDENT.md)
for consent, export, actual training, licensing, validation and student serving.

## Milestone 08 — Native trainable recurrent NTSA-X

NTSA-X now includes an **independent, from-scratch non-Transformer model**
with learned X/Y/Z recurrent latent-state updates, a byte-stream language
head, gradient-based training and held-out evaluation. It uses PyTorch as
its numerical/autograd engine but **does not instantiate a Transformer or
load teacher weights**. Its standalone local SSE server connects directly
to the existing recall-first conversational UI with student-only mode.

The native model is a research-scale starting point. It is not pretrained,
frontier-benchmarked, or automatically ready to replace a teacher model
for useful everyday questions. See [Milestone 08 architecture and execution](docs/MILESTONE-08-NATIVE-TRAINABLE-STATE.md).

## Milestone 16 — Browser-backed live web recall

Opt-in native web recall now uses a fresh, ephemeral browser to query a locally
operated SearXNG front end, then independently fetches public HTTPS source
pages using DNS-pinned, bounded reads. URL/timestamp/source excerpts feed the
existing unified cognitive loop before generation. Retrieved source links can
be emitted separately from native generated text. No extra neural model, teacher,
new model weights, or paid search API is required. This is NOT access to
ChatGPT's proprietary search system, a trusted entire-internet index, or
guaranteed unlimited web search.

Setup: [M16 browser web recall](docs/MILESTONE-16-BROWSER-WEB-RECALL.md).

## Milestone 15 — Unified cognition, background levels, memory and native extensions

One checkpoint-preserving recurrent cognitive loop now decodes and denoises
each byte with feedback into its updated recurrent state. A shared runtime
coordinates **L0 generation**, **L1 memory recall**, **L2 direct concurrent
trusted plugin computations**, **L3 explicitly authorized episodic storage**,
and **L4 offline learning disabled by default**. This is a prototype with
unproven conversational improvements, not an autonomous always-on process.
Read [M15 implementation guide](docs/MILESTONE-15-UNIFIED-COGNITION.md).

## Milestone 14 — Academy supervised learning with context-aware versioning

The original Academy instructions exceed the 128-byte COG-FU checkpoint:
the M14 preparer measures full ByteCodec conversation lengths and creates
separate train/development JSONL sets, excluding reserved exam questions.
Continuation now permits an explicit maximum-context-only extension with
strict weight transfer to a NEW model checkpoint. No weights are changed until
a local operator starts a training run. See
[Milestone 14 guide](docs/MILESTONE-14-ACADEMY-TRAINING.md).

## Milestone 13 — Kindred Academy: accelerated curriculum

Original language, math, and computer-science micro-lessons with explanations,
worked examples, separate practice and reserved exams, prerequisite mastery
gates, and a no-overwrite offline JSONL export. No model weights are trained
or upgraded in this milestone. See
[Milestone 13 guide](docs/MILESTONE-13-KINDRED-ACADEMY.md).

## Milestone 12 — Bounded multi-step agent missions

Added an **explicit mission controller** over Milestone 11 tools: inspect a
repository, investigate a source file, diagnose unit tests, or retrieve
an allowlisted research page with optional approved browser snapshot.
The controller plans within a fixed template, verifies observed tool
results, fails closed on nonzero exit status, and gives recovery guidance.
Consequential tools still need an **exact reviewed mission digest** and
separate opt-in capabilities. Untrusted web output cannot alter permissions
or cause unreviewed actions.

**This is deterministic orchestration, not a trained autonomous
intelligence or a fluent language model.** See
[Milestone 12 mission instructions](docs/MILESTONE-12-AGENT-MISSIONS.md)
for repeatable local PowerShell commands and limitations.

## Milestone 11 — NTSA-X agent skills workbench

The new **research-grade, local-first agent workbench** adds typed,
capability-gated source inspection/writes, Git status/diff, fixed Python test
execution, DevOps config inventory, approved HTTPS text research and optional
read-only Playwright browser snapshots. Structured 1–20-step plans require an
exact SHA-256 approval and separate capability enablement for any effect.
A conservative English phrase router proposes limited commands but **does not
give the current COG-FU model general language fluency or independent agency**.
See [Milestone 11 workbench instructions](docs/MILESTONE-11-AGENT-WORKBENCH.md).

## Milestone 10 — COG-FU native language acquisition

**Implemented research tools:** Train the recurrent cognitive cortex and
masked-byte diffusion denoiser jointly from **rights-confirmed local UTF-8
documents**, with document-disjoint held-out evaluation, versioned checkpoints,
optional matched recurrent-only baselines, safe UTF-8 output transport, and
operator-authored local identity context.

**No large language training has been performed or independently certified.**
These additions do not make the tiny smoke-test checkpoint conversationally
competent or establish improved reasoning. See
[Milestone 10: native language acquisition](docs/MILESTONE-10-COG-FU-LANGUAGE.md)
for the explicit data manifest, CPU pilot commands, benchmark definitions,
scientific limits and the teacher-free launch procedure.

## Milestone 09 — COG-FU cognitive–diffusion hybrid

**COG-FU** couples the existing native NTSA-X X/Y/Z recurrent cognitive
state to a trainable discrete masked-byte diffusion denoiser, with
**simultaneous causal and diffusion objectives**. Both pathways learn in
one PyTorch training pass; the denoiser's loss backpropagates into the same
cognitive state encoder. Native checkpoints can initialize its cortex,
and the existing verified-recall-first live UI can run COG-FU in
**teacher-free student mode**.

This is a research-scale, unpretrained hybrid—not a proven frontier model
or a verified reasoning system. The diffusion method is discrete iterative
mask refinement, not Gaussian image diffusion. See the
[COG-FU implementation guide](docs/MILESTONE-09-COG-FU.md) for training,
local hosting, evaluation, licensing and operational limits.

## Run locally

Python 3.12+; no third-party dependencies required:

```bash
python -m unittest discover -s tests -v
PYTHONPATH=. python examples/demo_orbit.py
```

See [Execution prototype](docs/EXECUTION.md), [Materializer integration contract](docs/INTEGRATIONS.md), and [Validation gates](docs/VALIDATION.md).

## Public repository notice

The repository is public; a publishing action alone does not grant a software license. See [LICENSE-NOTICE.md](LICENSE-NOTICE.md). Avoid adding secrets or unpublished proprietary assets.
