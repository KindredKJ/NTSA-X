"""NTSA-X paired orbital execution with local Materializer and durable Nexus."""
import argparse
from pathlib import Path
import tempfile
from time import sleep

from ntsa_x import Check, EventStream
from ntsa_x.counter_orbit import CounterOrbitEngine, PairedOperator
from ntsa_x.materializer import LocalMaterializer
from ntsa_x.nexus import NexusStore

CATEGORIES = ("state", "semantic", "causal", "constraint", "evidence", "delta")


def toy_oracle(projected):
    # Illustrative test oracle. Not production-grade external evidence.
    valid = projected.get("value") == 3
    return [Check(c, valid, "toy-oracle", external=(c == "evidence")) for c in CATEGORIES]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true", help="Pause before/during run so you can watch the localhost dashboard")
    parser.add_argument("--db", type=Path, help="SQLite path; default is a temporary demo database")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as temp, LocalMaterializer() as viewer:
        database = args.db or Path(temp) / "ntsa-x-demo.sqlite3"
        with NexusStore(database) as store:
            stream = EventStream(sink=viewer.publish)
            def forward_step(state):
                if args.watch:
                    sleep(0.3)
                return {"value": state["value"] + 1}
            def reverse_step(state):
                if args.watch:
                    sleep(0.3)
                return {"distance_to_goal": state["distance_to_goal"] - 1}
            operator = PairedOperator(
                "predict-and-reconstruct", forward_step, reverse_step, axis=(1, 1, -1),
            )
            engine = CounterOrbitEngine(
                verifier=toy_oracle,
                project_forward=lambda s: {"value": s["value"]},
                project_reverse=lambda s: {"value": 3 - s["distance_to_goal"]},
                store=store, stream=stream, max_steps=5, max_cost=10,
            )
            print("Materializer dashboard:", viewer.url, flush=True)
            if args.watch:
                input("Open the URL in a browser, then press Enter to begin...")
            result = engine.run({"value": 0}, {"distance_to_goal": 3}, [operator], context_id="demo-value-3")
            print("verified:", result.verified, "steps:", result.steps,
                  "state:", result.resolved, "node:", result.committed_digest)
            print("Nexus stats:", store.stats())
            recalled = engine.run(
                {"value": 0}, {"distance_to_goal": 3}, [operator], context_id="demo-value-3",
                recall_digest=result.committed_digest,
            )
            print("recall:", recalled.reason, "steps:", recalled.steps)
            if args.watch:
                input("Inspect the completed dashboard, then press Enter to stop...")


if __name__ == "__main__":
    main()
