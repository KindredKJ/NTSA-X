"""Research-only interventional memory and reproducible contextual experiments.

This is a small Bayesian action/outcome learner, not a trained AGI model.
It has no execution privileges and cannot establish a structural causal graph.
Synthetic results never qualify as external frontier benchmark scores.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import argparse
import json
import math
import random
import statistics
from collections.abc import Iterable


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value: object) -> str:
    return sha256(canonical(value)).hexdigest()


@dataclass(frozen=True)
class Belief:
    context: int
    action: int
    successes: int
    failures: int
    posterior_mean: float
    observations: int
    source: str = "LOCAL_INTERVENTIONAL_OBSERVATIONS_ONLY"


class InterventionalBeliefMemory:
    """Beta-Bernoulli predictive beliefs from measured, explicitly chosen actions."""

    def __init__(self, contexts: int, actions: int) -> None:
        if type(contexts) is not int or not 1 <= contexts <= 256:
            raise ValueError("contexts must be 1..256")
        if type(actions) is not int or not 2 <= actions <= 16:
            raise ValueError("actions must be 2..16")
        self.contexts, self.actions = contexts, actions
        self._success = [[0] * actions for _ in range(contexts)]
        self._failure = [[0] * actions for _ in range(contexts)]
        self._chain = "0" * 64
        self.events = 0

    def _location(self, context: int, action: int) -> None:
        if (type(context) is not int or not 0 <= context < self.contexts
                or type(action) is not int or not 0 <= action < self.actions):
            raise ValueError("invalid context or action")

    def predict(self, context: int, action: int) -> Belief:
        self._location(context, action)
        yes = self._success[context][action]
        no = self._failure[context][action]
        return Belief(context, action, yes, no, (yes + 1) / (yes + no + 2), yes + no)

    def choose(self, context: int, rng: random.Random) -> int:
        if type(context) is not int or not 0 <= context < self.contexts:
            raise ValueError("invalid context")
        if not isinstance(rng, random.Random):
            raise TypeError("explicit seeded RNG required for reproducibility")
        # Thompson sampling: randomized posterior interventions, not a
        # generic planner or proof of a structural causal effect.
        scores = [
            rng.betavariate(1 + self._success[context][a],
                            1 + self._failure[context][a])
            for a in range(self.actions)
        ]
        return max(range(self.actions), key=scores.__getitem__)

    def observe(self, context: int, action: int, reward: int) -> str:
        self._location(context, action)
        if type(reward) is not int or reward not in (0, 1):
            raise ValueError("reward must be integer 0 or 1")
        if reward:
            self._success[context][action] += 1
        else:
            self._failure[context][action] += 1
        self.events += 1
        self._chain = digest({
            "previous_sha256": self._chain,
            "event_index": self.events,
            "context": context,
            "action": action,
            "observed_reward": reward,
        })
        return self._chain

    @property
    def trace_sha256(self) -> str:
        return self._chain


class SyntheticContextualWorld:
    """Hidden synthetic rewards, common deterministic action-conditional streams."""

    def __init__(self, seed: int, contexts: int, actions: int) -> None:
        if type(seed) is not int or seed < 0 or seed > 2**32 - 1:
            raise ValueError("seed outside 32-bit unsigned range")
        if not 1 <= contexts <= 256 or not 2 <= actions <= 16:
            raise ValueError("invalid world shape")
        self.seed, self.contexts, self.actions = seed, contexts, actions
        generator = random.Random(seed)
        self._probs: list[tuple[float, ...]] = []
        for _ in range(contexts):
            probabilities = [0.08 + 0.84 * i / (actions - 1) for i in range(actions)]
            generator.shuffle(probabilities)
            self._probs.append(tuple(probabilities))

    def intervention(self, context: int, action: int, ordinal: int) -> int:
        if (type(context) is not int or not 0 <= context < self.contexts
                or type(action) is not int or not 0 <= action < self.actions
                or type(ordinal) is not int or ordinal < 0):
            raise ValueError("invalid synthetic intervention")
        payload = f"{self.seed}:{context}:{action}:{ordinal}".encode()
        draw = int.from_bytes(sha256(payload).digest()[:8], "big") / (1 << 64)
        return int(draw < self._probs[context][action])

    def oracle_mean(self) -> float:
        return statistics.mean(max(p) for p in self._probs)

    def hidden_task_hash(self) -> str:
        return digest({"seed": self.seed, "probabilities": self._probs})


def _experiment(seed: int, steps: int, warmup: int, contexts: int, actions: int) -> dict:
    world = SyntheticContextualWorld(seed, contexts, actions)
    memory = InterventionalBeliefMemory(contexts, actions)
    rng = random.Random(seed + 100000)
    baseline_rng = random.Random(seed + 200000)
    context_rng = random.Random(seed + 300000)
    visits_agent = [[0] * actions for _ in range(contexts)]
    visits_random = [[0] * actions for _ in range(contexts)]
    learned_rewards: list[int] = []
    random_rewards: list[int] = []
    for step in range(steps):
        context = context_rng.randrange(contexts)
        learned_action = memory.choose(context, rng)
        random_action = baseline_rng.randrange(actions)
        ar = world.intervention(context, learned_action, visits_agent[context][learned_action])
        rr = world.intervention(context, random_action, visits_random[context][random_action])
        visits_agent[context][learned_action] += 1
        visits_random[context][random_action] += 1
        memory.observe(context, learned_action, ar)
        if step >= warmup:
            learned_rewards.append(ar)
            random_rewards.append(rr)
    return {
        "seed": seed,
        "world_sha256": world.hidden_task_hash(),
        "trace_sha256": memory.trace_sha256,
        "learned_success_rate": statistics.mean(learned_rewards),
        "random_success_rate": statistics.mean(random_rewards),
        "oracle_expected_success_rate": world.oracle_mean(),
        "intervention_count": memory.events,
    }


def benchmark(seeds: Iterable[int] = range(12), steps: int = 800,
              warmup: int = 300, contexts: int = 8, actions: int = 3) -> dict:
    seeds = tuple(seeds)
    if not seeds or len(seeds) > 128 or len(set(seeds)) != len(seeds) or any(
        type(s) is not int or not 0 <= s < 2**32 for s in seeds
    ):
        raise ValueError("seeds must be 1..128 unique unsigned 32-bit integers")
    if (type(steps) is not int or type(warmup) is not int
            or not 100 <= steps <= 100000 or not 1 <= warmup < steps
            or type(contexts) is not int or not 1 <= contexts <= 256
            or type(actions) is not int or not 2 <= actions <= 16):
        raise ValueError("invalid experiment dimensions")
    rows = [_experiment(seed, steps, warmup, contexts, actions) for seed in seeds]
    differences = [r["learned_success_rate"] - r["random_success_rate"] for r in rows]
    delta = statistics.mean(differences)
    sd = statistics.stdev(differences) if len(differences) > 1 else 0
    margin = 1.96 * sd / math.sqrt(len(differences)) if len(differences) > 1 else None
    return {
        "experiment": "NTSA-X_INTERVENTIONAL_BELIEF_SYNTHETIC_V1",
        "verification_status": "INTERNAL_SYNTHETIC_RESEARCH_ONLY",
        "is_external_benchmark": False,
        "is_agi_proof": False,
        "is_frontier_rank_one_proof": False,
        "method": "Beta-Bernoulli Thompson sampling with action/outcome evidence",
        "baseline": "seeded random action policy",
        "protocol": {
            "seeds": list(seeds), "steps": steps, "warmup": warmup,
            "contexts": contexts, "actions": actions,
            "identical_context_schedule": True,
            "shared_action_conditional_reward_streams": True,
        },
        "protocol_sha256": digest({"seeds": seeds, "steps": steps, "warmup": warmup,
                                   "contexts": contexts, "actions": actions}),
        "learned_mean_success_rate": statistics.mean(r["learned_success_rate"] for r in rows),
        "random_mean_success_rate": statistics.mean(r["random_success_rate"] for r in rows),
        "oracle_expected_success_rate": statistics.mean(r["oracle_expected_success_rate"] for r in rows),
        "paired_gain_over_random": delta,
        "paired_gain_ci_95_normal_approx": [delta-margin, delta+margin] if margin is not None else None,
        "limitations": [
            "Synthetic contextual stochastic rewards are not a public AGI dataset.",
            "Task and reward structure are predefined by the evaluator, not discovered as a structural causal graph.",
            "95% CI is a normal approximation across environment seeds, not a guarantee of real-world generalization.",
            "No PyTorch foundation-model weights, agent deployment, customer, or independent evaluator are involved.",
        ],
        "runs": rows,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Measure an interventional memory policy against a fair random baseline")
    parser.add_argument("--seeds", type=int, default=12)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--warmup", type=int, default=300)
    parser.add_argument("--contexts", type=int, default=8)
    parser.add_argument("--actions", type=int, default=3)
    parser.add_argument("--output", help="JSON output path, omitted to print stdout")
    args = parser.parse_args(argv)
    try:
        report = benchmark(range(args.seeds), args.steps, args.warmup,
                           args.contexts, args.actions)
        encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
        if args.output:
            from pathlib import Path
            path = Path(args.output)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(encoded, encoding="utf-8")
        print(encoded, end="")
        return 0
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
