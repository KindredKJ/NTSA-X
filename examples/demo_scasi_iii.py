"""One-shot executable: python -m examples.demo_scasi_iii"""
import json
from ntsa_x.memory_fabric import MemoryFabric
from ntsa_x.scasi_iii import Objective, PredictiveSuperCycle


def main():
    goals = (
        Objective("deployment", ("deploy service", "go live"),
                  ("release manifest", "rollback plan", "service credentials")),
        Objective("research", ("train model", "knowledge gap"),
                  ("training data rights", "held out evaluation")),
    )
    with MemoryFabric(":memory:", writable=True) as memory:
        memory.remember("procedural", "release manifest is staged", source="local-demo", session="scasi-demo")
        engine = PredictiveSuperCycle(goals, memory=memory, session="scasi-demo")
        # One-shot precompute happens before the simulated next input.
        engine.warm()
        response = engine.resolve("Can we deploy service now?")
        print(json.dumps({"response": response, "telemetry": engine.snapshot()}, indent=2))


if __name__ == "__main__":
    main()
