# Validation and research gates

The NTSA-X design is unproven until reproducible evaluation.

1. Determinism: canonical state serialization, hash stability, replay of event and commit histories.
2. Safety: reject unverified commits, missing parents, unauthorized external effects, and over-budget operations.
3. Correctness: compare against external ground truth and adversarial counterexamples; do not accept internal counter-orbit agreement alone.
4. Retrieval: measure precision, stale-result detection, invalidation, and provenance coverage.
5. Performance: compare fixed-budget baselines against attention, SSM, retrieval-augmented and graph approaches using equivalent hardware, datasets and quality targets.
6. Efficiency: measure wall-clock latency, tokens/bytes processed, memory, power/joules where instruments exist, and failure/retry overhead.
7. Robustness: partial memory corruption, plugin failure, network loss, resume and deterministic recovery.
8. Visualization: verify Materializer event compatibility without allowing the renderer to affect decisions.

Publish datasets, configurations, seeds, hardware, traces and uncertainty intervals. Do not claim superhuman capability, guaranteed savings, or superiority without measured evidence.
