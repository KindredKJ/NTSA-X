"""Run a verified 100-task production batch, optionally watching Materializer."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from ntsa_x.events import EventStream
from ntsa_x.factory import FactoryTask, ProductionFactory
from ntsa_x.materializer import LocalMaterializer
from ntsa_x.nexus import NexusStore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=int, default=100)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--db", type=Path, default=None)
    parser.add_argument("--watch", action="store_true",
                        help="Keep local Materializer open for visual inspection")
    args = parser.parse_args()
    if not 1 <= args.tasks <= 1000 or not 1 <= args.workers <= 100:
        parser.error("--tasks must be 1..1000 and --workers must be 1..100")
    tasks = [
        FactoryTask(task_id=f"task-{i:04d}", operation="increment", seed=i, amount=2)
        for i in range(args.tasks)
    ]
    with ExitStack() as stack:
        tmp = stack.enter_context(TemporaryDirectory())
        db_path = args.db or (Path(tmp) / "ntsa-factory-demo.sqlite3")
        store = stack.enter_context(NexusStore(db_path))
        viewer = stack.enter_context(LocalMaterializer()) if args.watch else None
        stream = EventStream(sink=viewer.publish if viewer else None,
                             max_events=max(1000, 2 * len(tasks) + 10))
        factory = ProductionFactory(store, workers=args.workers, stream=stream)
        if viewer:
            print("Live local Materializer:", viewer.url, flush=True)
            input("Open the URL in your browser, then press Enter to start 100-task batch...")
        result = factory.run(tasks, batch_id=f"demo-{args.tasks}-workers-{args.workers}")
        print(json.dumps(result.to_dict(), indent=2))
        print("Nexus:", store.stats())
        if viewer:
            input("View the completed factory telemetry, then press Enter to stop...")
        if result.completed != len(tasks):
            raise SystemExit("Factory did not verify all tasks")


if __name__ == "__main__":
    main()
