"""Watch a worker-backed convergence get promoted and reused from persistent Nexus."""
import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

from ntsa_x import (
    Check, ConvergenceLibrary, IndexedCounterOrbitEngine, NexusStore,
    PluginManifest, PluginRegistry, PluginWorker, EventStream, KindredContracts,
)

CATEGORIES = ("state", "semantic", "causal", "constraint", "evidence", "delta")


def toy_verifier(state):
    # This demonstrates plumbing only; it is not independently authenticated truth.
    good = state.get("value") == 3
    return [Check(c, good, "toy-local-verifier", external=(c == "evidence"))
            for c in CATEGORIES]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=None, help="Persistent SQLite Nexus path")
    args = parser.parse_args()
    with TemporaryDirectory() as temp:
        db = args.db if args.db is not None else Path(temp) / "reusable.sqlite3"
        with NexusStore(db) as store:
            stream = EventStream()
            registry = PluginRegistry()
            registry.register(PluginManifest("kindred.scalar.demo", "1.0.0", ("increment",)))
            worker = PluginWorker(registry, timeout=5, stream=stream)
            operator = worker.paired_operator(
                "kindred.scalar.demo", "1.0.0",
                forward_op="increment", reverse_op="increment",
            )
            engine = IndexedCounterOrbitEngine(
                library=ConvergenceLibrary(store), store=store, stream=stream,
                verifier=toy_verifier, project_forward=lambda state: state,
                project_reverse=lambda state: state, max_steps=5, max_cost=10,
            )
            params = dict(context_id="sample-goal-3", reuse_scope="scalar-demo@1.0.0+goal-3",
                          verifier_revision="toy-verifier/v1")
            first = engine.run({"value": 0}, {"value": 0}, [operator], **params)
            second = engine.run({"value": 0}, {"value": 0}, [], **params)
            print("First pass:", first.reason, "steps:", first.steps,
                  "digest:", first.committed_digest)
            print("Second pass:", second.reason, "steps:", second.steps)
            print("Nexus records:", store.stats())
            print("Kindred event envelope:",
                  KindredContracts().wrap_event("kindred.materializer",
                                                stream.events[-1].to_dict())["schema_version"])
            assert first.verified and second.verified and second.steps == 0


if __name__ == "__main__":
    main()
