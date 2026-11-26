# SW12 — compositional convergence: evidence and extension specification

Base: 8ea3b87d0e84a0bc64870748aba600180d245298.
Benchmark and test revision: 50e5443c53c6baf46808e25902c55a888609d427.
Execution: local Python; five relevant source files reconstructed from GitHub and independently checked with git hash-object before execution. Three targeted unittest cases passed. Full repository checkout and full CI suite were not executed locally; GitHub Actions run 45803549313 failed and sampled job steps were empty.

## Measured evidence

Synthetic independent four-coordinate rings with contraction q=7/8. Both iterative schedules were checked against an analytic rational reference. The final integer canonicalization is specific to the planted integer fixtures; zero recorded final residual does not establish arbitrary numerical accuracy.

| Blocks | Trials per reuse level | Cold ms | 75% ms | 100% component reuse ms | 75% cold speedup | Direct ms | No-store iterative ms |
|---:|---:|---:|---:|---:|---:|---:|---:|
|4|3|9.446|4.436|3.270|2.129|0.583|1.542|
|16|3|29.988|12.645|7.393|2.371|1.952|6.203|
|64|3|115.203|45.498|24.516|2.532|7.635|23.490|
|256|1|613.073|211.874|109.206|2.894|29.995|96.169|
|1024|1|1892.446|702.784|432.507|2.693|119.129|406.580|

55 completed condition-trials across 0/25/50/75/100% reusable components; all recorded final residuals and errors zero. One-repeat large cases are exploratory, not statistical estimates. The initial all-size three-repeat run timed out; smaller batches completed. Priming is measured separately; at 1024 blocks, 75% priming was 1179.690 ms and the warm workload 702.784 ms. This is not a demonstrated advantage over the strongest direct baseline. Exact composite recall is measured separately. No neural or distributed worker claim.

## Mathematical extension (proposed, not implemented by SW12)

A state identity binds objective, canonical input, operator/model/verifier revisions, constraints, evidence/dependency versions, numeric representation and ordered parent digests. Separate applicability from content identity. Let Z be the vector of local states and F the joint update map on a complete invariant domain. If componentwise differences satisfy |F_i(Z)-F_i(Y)| <= sum_j L_ij |Z_j-Y_j| for a nonnegative gain matrix L with spectral radius < 1, choose w>0 and q<1 with Lw<=qw. In the weighted infinity norm this gives a contraction and unique global fixed point. For approximate state Zhat with residual vector r, the componentwise error is bounded by (I-L)^(-1)r, subject to the same domain assumptions. A new coupling or dependency can invalidate the certificate. Local contraction alone does not imply joint contraction.

Compose existing verified states using ConvergenceLibrary.compose only after joint current verification. Exact recall requires matching identity and current applicability. Changed but compatible state is a warm start, not a verified output. A learned transferable operator must be independently validated on held-out novel tasks. Changes create immutable new revisions and propagate through reverse dependencies; coupled regions form strongly connected components, while historical lineage remains a DAG. Reverify unaffected states against dependency epochs; do not overwrite historical records.

## Next execution sequence

1. Instrument per-operation lookup, database, hashing, lineage traversal, reference verification and materialization costs. Implement iterative shared-ancestor traversal with snapshot-scoped memoization and corruption tests. Do not skip evidence validation.
2. Add weakly and strongly coupled blocks, sparse/dense matrices, perturbations, bounded asynchronous schedules and selective SCC re-solving. Compare against direct and incremental numerical baselines.
3. Add an applicability interface to the existing recall layer: exact match, certified transformed result, warm start, new computation. Measure total amortized cost including priming, maintenance and storage.
4. Integrate the same state representation into a single native recurrent model only after branch-specific Auxiliary source reconciliation. Compare recall-disabled, memoization, warm-start, composition, reciprocal and full variants using matched parameter/training/inference budgets and held-out combinations.

Reproduction:
    python -m unittest discover -s tests -p test_compositional_reuse_benchmark.py -v
    python -m examples.benchmark_compositional_reuse --sizes 4 16 64 --repeats 3 --output sw12.json
    python -m examples.benchmark_compositional_reuse --sizes 256 --repeats 1 --output sw12-256.json
    python -m examples.benchmark_compositional_reuse --sizes 1024 --repeats 1 --output sw12-1024.json
