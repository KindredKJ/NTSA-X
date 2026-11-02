"""Native NTSA-X finite-budget observation/novelty policy for the official ARC-AGI-3 SDK.

No language-model API and no model weights. Scores originate only from the
organizer's actual ARC-AGI-3 Arcade SDK. Missing access is reported as BLOCKED.
This is a developer experiment, NOT an official Kaggle competition submission.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


class NoveltyPolicy:
    """Bounded exploration memory and UCB over observed action outcomes."""

    def __init__(self, seed: int):
        self.rng = random.Random(seed)
        self.counts = {}
        self.gain = {}
        self.visited = set()
        self.steps = 0
        self.previous = None
        self.chain = "0" * 64

    def fingerprint(self, observation):
        if observation is None:
            return None
        # FrameDataRaw contains real observations, sometimes animated frame stacks.
        frame = getattr(observation, "frame", None)
        if frame is None:
            frame = getattr(observation, "frame_data", None)
        if frame is None:
            return None
        payload = json.dumps(frame, sort_keys=True, default=str, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    def pick(self, actions):
        if not actions:
            raise ValueError("no official actions available")
        if self.steps < len(actions):
            return actions[self.steps]
        total = max(1, sum(self.counts.values()))
        def score(action):
            name = str(getattr(action, "name", action))
            observed = self.counts.get(name, 0)
            if not observed:
                return float("inf")
            return self.gain.get(name, 0) / observed + math.sqrt(2 * math.log(total + 1) / observed)
        scores = [score(action) for action in actions]
        best = max(scores)
        options = [i for i, value in enumerate(scores) if value == best]
        return actions[self.rng.choice(options)]

    def record(self, action, observation):
        stamp = self.fingerprint(observation)
        new_state = bool(stamp and stamp not in self.visited)
        if stamp:
            self.visited.add(stamp)
        name = str(getattr(action, "name", action))
        self.counts[name] = self.counts.get(name, 0) + 1
        self.gain[name] = self.gain.get(name, 0) + int(new_state)
        self.steps += 1
        self.chain = hashlib.sha256(
            json.dumps({
                "prev": self.chain, "index": self.steps,
                "action": name, "frame_digest": stamp,
                "novel_state": new_state,
            }, sort_keys=True).encode()
        ).hexdigest()
        self.previous = stamp
        return new_state


def run(game: str, max_steps: int, seed: int):
    if not game or len(game) > 20 or any(not c.isalnum() for c in game):
        raise ValueError("invalid game id")
    if not 1 <= max_steps <= 100 or not 0 <= seed < 2**32:
        raise ValueError("invalid budget or seed")
    report = {
        "benchmark": "ARC-AGI-3",
        "game_requested": game,
        "agent": "NTSA-X novelty-memory native fixed-function agent",
        "model": "NONE: no trained Kindred or provider foundation model",
        "mode": "ARC_OFFICIAL_SDK_DEVELOPER_EXPERIMENT",
        "competition_submission": False,
        "official_leaderboard_result": False,
        "independent_agi_evidence": False,
        "human_actions_during_run": 0,
        "action_budget": max_steps,
        "seed": seed,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "NOT_EXECUTED",
        "score": None,
        "steps_executed": 0,
    }
    try:
        import arc_agi
        from arcengine import GameState
        report["sdk_package"] = getattr(arc_agi, "__version__", "package_version_not_exposed")
        policy = NoveltyPolicy(seed)
        arcade = arc_agi.Arcade()
        # A real organizer-hosted environment, never a synthetic replacement.
        env = arcade.make(game)
        if env is None:
            raise RuntimeError("ARC SDK did not grant access to requested game")
        actions = tuple(env.action_space)
        if not actions:
            raise RuntimeError("ARC returned no authorized actions")
        report["action_space"] = [str(getattr(a, "name", a)) for a in actions]
        started = time.monotonic()
        last_state = None
        while policy.steps < max_steps:
            if time.monotonic() - started >= 240:
                report["timeout_budget_seconds"] = 240
                break
            action = policy.pick(actions)
            payload = {}
            if action.is_complex():
                payload = {"x": policy.rng.randrange(64), "y": policy.rng.randrange(64)}
            observation = env.step(action, data=payload)
            policy.record(action, observation)
            if observation is not None:
                last_state = str(getattr(getattr(observation, "state", None), "name", "UNKNOWN"))
                if observation.state == GameState.WIN or observation.state == GameState.GAME_OVER:
                    break
        report["steps_executed"] = policy.steps
        report["distinct_observation_hashes"] = len(policy.visited)
        report["observation_trace_sha256"] = policy.chain
        report["last_state"] = last_state
        scorecard = arcade.get_scorecard()
        if scorecard is None:
            report["status"] = "ACTIONS_EXECUTED_NO_ORGANIZER_SCORECARD"
        else:
            score = getattr(scorecard, "score", None)
            if type(score) in (int, float) and math.isfinite(score):
                report["score"] = float(score)
                report["status"] = "ORGANIZER_SDK_SCORECARD_RETURNED"
                report["organizer_scorecard_link_independently_verified"] = False
            else:
                report["status"] = "SCORECARD_RECEIVED_WITHOUT_NUMERIC_SCORE"
    except Exception as exc:
        report["status"] = "BLOCKED_OR_RUN_FAILED"
        report["error_type"] = type(exc).__name__
        report["error_summary"] = str(exc)[:400].replace("\n", " ")
    report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="ls20")
    parser.add_argument("--steps", default=30, type=int)
    parser.add_argument("--seed", default=23, type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(args.game, args.steps, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes((json.dumps(result, indent=2, sort_keys=True) + "\n").encode())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] in {
        "ORGANIZER_SDK_SCORECARD_RETURNED",
        "ACTIONS_EXECUTED_NO_ORGANIZER_SCORECARD",
        "SCORECARD_RECEIVED_WITHOUT_NUMERIC_SCORE",
    } else 2


if __name__ == "__main__":
    sys.exit(main())
