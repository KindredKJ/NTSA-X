"""Local demo. This verifier is a deterministic test oracle, not external proof."""
from ntsa_x import Check, EventStream, Operator, OrbitEngine

CATEGORIES = ("state", "semantic", "causal", "constraint", "evidence", "delta")


def test_verifier(candidate):
    matches_target = candidate.get("value") == 3
    return [
        Check(category, matches_target, "demo-oracle", external=category == "evidence")
        for category in CATEGORIES
    ]


def main():
    telemetry = EventStream()
    engine = OrbitEngine(verifier=test_verifier, stream=telemetry, max_steps=5, max_cost=5)
    increment = Operator("increment", lambda s: {"value": s["value"] + 1}, axis=(1, 1, 0))
    result = engine.run({"value": 0}, [increment], context_id="demo-target-3")
    print("verified:", result.verified, "steps:", result.steps)
    print("commit:", result.committed_digest)
    print("event types:", [event.type for event in telemetry.events])
    recalled = engine.run({"value": 0}, [increment], context_id="demo-target-3", recall_digest=result.committed_digest)
    print("recalled:", recalled.reason, "steps:", recalled.steps)


if __name__ == "__main__":
    main()
