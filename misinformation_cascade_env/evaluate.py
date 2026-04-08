"""
Deterministic benchmark runner for the misinformation cascade environment.

Usage:
    python -m misinformation_cascade_env.evaluate --episodes 20
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Callable

from misinformation_cascade_env.env import MisinformationCascadeEnv
from misinformation_cascade_env.models import (
    ACTION_COSTS,
    TASK_SEEDS,
    CascadeAction,
    CascadeObservation,
)


PolicyFn = Callable[[CascadeObservation, random.Random], CascadeAction]


@dataclass
class EpisodeResult:
    reward: float
    steps: int
    infected_final: int
    quarantined_final: int
    inoculated_final: int


def wait_policy(obs: CascadeObservation, rng: random.Random) -> CascadeAction:
    _ = obs
    _ = rng
    return CascadeAction(action_type="WAIT")


def random_policy(obs: CascadeObservation, rng: random.Random) -> CascadeAction:
    candidates = sorted({n.node_id for n in obs.top_nodes + obs.at_risk_nodes})
    if not candidates:
        return CascadeAction(action_type="WAIT")

    actions = ["WAIT"]
    if obs.budget_remaining >= ACTION_COSTS["FACTCHECK"]:
        actions.append("FACTCHECK")
    if obs.budget_remaining >= ACTION_COSTS["BOOST_CORRECTION"]:
        actions.append("BOOST_CORRECTION")
    if obs.budget_remaining >= ACTION_COSTS["INOCULATE"]:
        actions.append("INOCULATE")
    if obs.budget_remaining >= ACTION_COSTS["QUARANTINE"]:
        actions.append("QUARANTINE")

    choice = rng.choice(actions)
    if choice == "WAIT":
        return CascadeAction(action_type="WAIT")
    return CascadeAction(action_type=choice, target_node_id=rng.choice(candidates))


def greedy_containment_policy(
    obs: CascadeObservation, rng: random.Random
) -> CascadeAction:
    _ = rng

    confirmed = sorted(
        obs.confirmed_infected,
        key=lambda n: n.influence_score,
        reverse=True,
    )
    at_risk = sorted(
        obs.at_risk_nodes,
        key=lambda n: n.influence_score,
        reverse=True,
    )
    top = sorted(
        obs.top_nodes,
        key=lambda n: n.influence_score,
        reverse=True,
    )

    if confirmed and obs.budget_remaining >= ACTION_COSTS["QUARANTINE"]:
        return CascadeAction(action_type="QUARANTINE", target_node_id=confirmed[0].node_id)

    if at_risk and obs.budget_remaining >= ACTION_COSTS["INOCULATE"]:
        return CascadeAction(action_type="INOCULATE", target_node_id=at_risk[0].node_id)

    if top and obs.budget_remaining >= ACTION_COSTS["BOOST_CORRECTION"]:
        return CascadeAction(action_type="BOOST_CORRECTION", target_node_id=top[0].node_id)

    if at_risk and obs.budget_remaining >= ACTION_COSTS["FACTCHECK"]:
        return CascadeAction(action_type="FACTCHECK", target_node_id=at_risk[0].node_id)

    return CascadeAction(action_type="WAIT")


POLICIES: dict[str, PolicyFn] = {
    "wait": wait_policy,
    "random": random_policy,
    "greedy_containment": greedy_containment_policy,
}


def run_episode(
    difficulty: str,
    seed: int,
    policy_name: str,
) -> EpisodeResult:
    env = MisinformationCascadeEnv(difficulty=difficulty, seed=seed)
    rng = random.Random(seed + 100_000)
    obs = env.reset(seed=seed)

    steps = 0
    while not obs.done:
        action = POLICIES[policy_name](obs, rng)
        obs = env.step(action)
        steps += 1

    return EpisodeResult(
        reward=float(obs.reward),
        steps=steps,
        infected_final=obs.infected_count,
        quarantined_final=obs.quarantined_count,
        inoculated_final=obs.inoculated_count,
    )


def benchmark(
    episodes: int,
    policies: list[str],
    difficulties: list[str],
) -> dict:
    out: dict = {"episodes": episodes, "results": {}}

    for policy in policies:
        out["results"][policy] = {}
        for difficulty in difficulties:
            base = TASK_SEEDS[difficulty]
            runs = [run_episode(difficulty, base + i, policy) for i in range(episodes)]
            out["results"][policy][difficulty] = {
                "avg_reward": round(mean(r.reward for r in runs), 4),
                "avg_steps": round(mean(r.steps for r in runs), 2),
                "avg_final_infected": round(mean(r.infected_final for r in runs), 2),
                "avg_final_quarantined": round(
                    mean(r.quarantined_final for r in runs), 2
                ),
                "avg_final_inoculated": round(mean(r.inoculated_final for r in runs), 2),
            }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument(
        "--policies",
        nargs="+",
        default=["wait", "random", "greedy_containment"],
    )
    parser.add_argument(
        "--difficulties",
        nargs="+",
        default=["easy", "medium", "hard"],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/benchmark_results.json"),
    )
    args = parser.parse_args()

    for p in args.policies:
        if p not in POLICIES:
            raise ValueError(f"Unknown policy '{p}'. Available: {sorted(POLICIES)}")

    data = benchmark(
        episodes=args.episodes,
        policies=args.policies,
        difficulties=args.difficulties,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2), encoding="utf-8")

    print(f"Saved benchmark report: {args.output}")
    for policy in args.policies:
        print(f"\n[{policy}]")
        for difficulty in args.difficulties:
            row = data["results"][policy][difficulty]
            print(
                f"  {difficulty:>6}  reward={row['avg_reward']:.4f}  "
                f"steps={row['avg_steps']:.2f}  infected={row['avg_final_infected']:.2f}"
            )


if __name__ == "__main__":
    main()
