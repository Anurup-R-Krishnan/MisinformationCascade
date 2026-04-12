"""
Real-world KPI evaluator for the misinformation cascade environment.

Usage:
    python -m misinformation_cascade_env.evaluate_realworld --episodes 20
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

from misinformation_cascade_env.env import MisinformationCascadeEnv
from misinformation_cascade_env.evaluate import POLICIES
from misinformation_cascade_env.models import STARTING_BUDGET, TASK_CONFIG, TASK_SEEDS, CascadeObservation


@dataclass
class EpisodeTrace:
    reward: float
    steps: int
    peak_infected: int
    avg_infected: float
    spread_events: int
    budget_spent: int
    first_action_step: int


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def run_episode_trace(difficulty: str, seed: int, policy_name: str) -> EpisodeTrace:
    env = MisinformationCascadeEnv(difficulty=difficulty, seed=seed)
    rng = random.Random(seed + 100_000)
    obs = env.reset(seed=seed)

    budget_initial = obs.budget_remaining
    infected_series = [obs.infected_count]
    spread_events = 0
    first_action_step = obs.max_steps + 1

    steps = 0
    while not obs.done:
        action = POLICIES[policy_name](obs, rng)
        if action.action_type != "WAIT" and first_action_step == obs.max_steps + 1:
            first_action_step = obs.step
        obs = env.step(action)
        steps += 1

        infected_series.append(obs.infected_count)
        spread_events += max(0, obs.spread_delta_last_step)

    if first_action_step == obs.max_steps + 1:
        first_action_step = obs.max_steps + 1

    return EpisodeTrace(
        reward=float(obs.reward),
        steps=steps,
        peak_infected=max(infected_series),
        avg_infected=mean(infected_series),
        spread_events=spread_events,
        budget_spent=budget_initial - obs.budget_remaining,
        first_action_step=first_action_step,
    )


def _score_domains(aggregates: dict[str, float]) -> dict[str, float]:
    reward = _clamp01(aggregates["avg_reward"])
    peak_control = _clamp01(1.0 - aggregates["avg_peak_ratio"])
    sustained_control = _clamp01(1.0 - aggregates["avg_auc_ratio"])
    spread_control = _clamp01(1.0 - aggregates["avg_spread_ratio"])
    response_speed = _clamp01(1.0 - aggregates["avg_first_action_ratio"])
    budget_efficiency = _clamp01(1.0 - aggregates["avg_budget_spent_ratio"])

    trust_safety = _clamp01(0.45 * reward + 0.35 * peak_control + 0.20 * sustained_control)
    election_integrity = _clamp01(0.40 * reward + 0.35 * spread_control + 0.25 * peak_control)
    public_health = _clamp01(0.40 * sustained_control + 0.30 * reward + 0.30 * response_speed)
    crisis_response = _clamp01(0.45 * response_speed + 0.35 * peak_control + 0.20 * reward)
    network_resilience = _clamp01(0.45 * reward + 0.30 * peak_control + 0.25 * budget_efficiency)

    overall = _clamp01(
        mean(
            [
                trust_safety,
                election_integrity,
                public_health,
                crisis_response,
                network_resilience,
            ]
        )
    )

    return {
        "trust_safety": round(trust_safety, 4),
        "election_integrity": round(election_integrity, 4),
        "public_health": round(public_health, 4),
        "crisis_response": round(crisis_response, 4),
        "network_resilience": round(network_resilience, 4),
        "overall_real_world_utility": round(overall, 4),
    }


def evaluate_real_world_kpis(
    episodes: int,
    policies: list[str],
    difficulties: list[str],
) -> dict:
    out: dict = {
        "episodes": episodes,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "policies": policies,
        "difficulties": difficulties,
        "difficulty_weights": {"easy": 0.2, "medium": 0.3, "hard": 0.5},
        "results": {},
    }

    difficulty_weights = {"easy": 0.2, "medium": 0.3, "hard": 0.5}

    for policy in policies:
        policy_row: dict = {"by_difficulty": {}, "domain_kpis": {}}
        weighted_aggregates = {
            "avg_reward": 0.0,
            "avg_peak_ratio": 0.0,
            "avg_auc_ratio": 0.0,
            "avg_spread_ratio": 0.0,
            "avg_first_action_ratio": 0.0,
            "avg_budget_spent_ratio": 0.0,
        }

        for difficulty in difficulties:
            cfg = TASK_CONFIG[difficulty]
            n_nodes = cfg["n_nodes"]
            max_steps = cfg["max_steps"]
            runs = [
                run_episode_trace(difficulty, TASK_SEEDS[difficulty] + i, policy)
                for i in range(episodes)
            ]

            aggregates = {
                "avg_reward": mean(r.reward for r in runs),
                "avg_peak_ratio": mean(r.peak_infected / n_nodes for r in runs),
                "avg_auc_ratio": mean(r.avg_infected / n_nodes for r in runs),
                "avg_spread_ratio": mean(r.spread_events / n_nodes for r in runs),
                "avg_first_action_ratio": mean(
                    min(1.0, r.first_action_step / max_steps) for r in runs
                ),
                "avg_budget_spent_ratio": mean(
                    r.budget_spent / STARTING_BUDGET for r in runs
                ),
            }

            policy_row["by_difficulty"][difficulty] = {
                "aggregates": {k: round(v, 4) for k, v in aggregates.items()},
                "domain_kpis": _score_domains(aggregates),
            }

            weight = difficulty_weights.get(difficulty, 0.0)
            for key in weighted_aggregates:
                weighted_aggregates[key] += aggregates[key] * weight

        policy_row["domain_kpis"] = _score_domains(weighted_aggregates)
        out["results"][policy] = policy_row

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
        default=Path("artifacts/real_world_kpi_results.json"),
    )
    args = parser.parse_args()

    for p in args.policies:
        if p not in POLICIES:
            raise ValueError(f"Unknown policy '{p}'. Available: {sorted(POLICIES)}")

    for d in args.difficulties:
        if d not in TASK_CONFIG:
            raise ValueError(f"Unknown difficulty '{d}'. Available: {sorted(TASK_CONFIG)}")

    data = evaluate_real_world_kpis(
        episodes=args.episodes,
        policies=args.policies,
        difficulties=args.difficulties,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2), encoding="utf-8")

    print(f"Saved real-world KPI report: {args.output}")
    for policy in args.policies:
        kpis = data["results"][policy]["domain_kpis"]
        print(
            f"[{policy}] utility={kpis['overall_real_world_utility']:.4f} "
            f"trust_safety={kpis['trust_safety']:.4f} "
            f"election={kpis['election_integrity']:.4f}"
        )


if __name__ == "__main__":
    main()
