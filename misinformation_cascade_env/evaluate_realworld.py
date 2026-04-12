"""
Real-world KPI evaluator for the misinformation cascade environment.

Computes domain-specific Key Performance Indicators (trust & safety,
election integrity, public health, crisis response, network resilience)
by running deterministic benchmark episodes and scoring outcomes.

All emitted scores lie in the strict open interval (0, 1).

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
from misinformation_cascade_env.models import (
    STARTING_BUDGET,
    TASK_CONFIG,
    TASK_SEEDS,
    CascadeObservation,
)
from misinformation_cascade_env.task_grader import clamp_score

# ---------------------------------------------------------------------------
# Difficulty weights for the overall KPI aggregation.
# ---------------------------------------------------------------------------

DIFFICULTY_WEIGHTS: dict[str, float] = {"easy": 0.2, "medium": 0.3, "hard": 0.5}

# Names of the raw aggregate keys computed per difficulty.
_AGGREGATE_KEYS = (
    "avg_reward",
    "avg_peak_ratio",
    "avg_auc_ratio",
    "avg_spread_ratio",
    "avg_first_action_ratio",
    "avg_budget_spent_ratio",
)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class EpisodeTrace:
    """Metrics collected from a single benchmark episode."""

    reward: float
    steps: int
    peak_infected: int
    avg_infected: float
    spread_events: int
    budget_spent: int
    first_action_step: int


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode_trace(difficulty: str, seed: int, policy_name: str) -> EpisodeTrace:
    """Run a single episode and collect detailed trace metrics."""
    env = MisinformationCascadeEnv(difficulty=difficulty, seed=seed)
    rng = random.Random(seed + 100_000)
    obs = env.reset(seed=seed)

    budget_initial = obs.budget_remaining
    infected_series = [obs.infected_count]
    spread_events = 0
    first_action_step: int | None = None
    steps = 0

    while not obs.done:
        action = POLICIES[policy_name](obs, rng)
        if first_action_step is None and action.action_type != "WAIT":
            first_action_step = obs.step
        obs = env.step(action)
        steps += 1
        infected_series.append(obs.infected_count)
        spread_events += max(0, obs.spread_delta_last_step)

    return EpisodeTrace(
        reward=float(obs.reward),
        steps=steps,
        peak_infected=max(infected_series),
        avg_infected=mean(infected_series),
        spread_events=spread_events,
        budget_spent=budget_initial - obs.budget_remaining,
        first_action_step=first_action_step if first_action_step is not None else obs.max_steps + 1,
    )


# ---------------------------------------------------------------------------
# Domain KPI scoring
# ---------------------------------------------------------------------------

def _compute_aggregates(
    runs: list[EpisodeTrace],
    n_nodes: int,
    max_steps: int,
) -> dict[str, float]:
    """Compute normalised aggregate metrics from a batch of episode traces."""
    return {
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


def _score_domains(aggregates: dict[str, float]) -> dict[str, float]:
    """Derive five domain KPI scores + overall utility from raw aggregates.

    Every output score is clamped to strict (0, 1) via ``clamp_score``.
    """
    # Normalised input signals — higher is better for all.
    reward = clamp_score(aggregates["avg_reward"])
    peak_control = clamp_score(1.0 - aggregates["avg_peak_ratio"])
    sustained_control = clamp_score(1.0 - aggregates["avg_auc_ratio"])
    spread_control = clamp_score(1.0 - aggregates["avg_spread_ratio"])
    response_speed = clamp_score(1.0 - aggregates["avg_first_action_ratio"])
    budget_efficiency = clamp_score(1.0 - aggregates["avg_budget_spent_ratio"])

    # Weighted domain composites.
    trust_safety = clamp_score(
        0.45 * reward + 0.35 * peak_control + 0.20 * sustained_control
    )
    election_integrity = clamp_score(
        0.40 * reward + 0.35 * spread_control + 0.25 * peak_control
    )
    public_health = clamp_score(
        0.40 * sustained_control + 0.30 * reward + 0.30 * response_speed
    )
    crisis_response = clamp_score(
        0.45 * response_speed + 0.35 * peak_control + 0.20 * reward
    )
    network_resilience = clamp_score(
        0.45 * reward + 0.30 * peak_control + 0.25 * budget_efficiency
    )

    overall = clamp_score(
        mean([
            trust_safety,
            election_integrity,
            public_health,
            crisis_response,
            network_resilience,
        ])
    )

    return {
        "trust_safety": round(trust_safety, 4),
        "election_integrity": round(election_integrity, 4),
        "public_health": round(public_health, 4),
        "crisis_response": round(crisis_response, 4),
        "network_resilience": round(network_resilience, 4),
        "overall_real_world_utility": round(overall, 4),
    }


# ---------------------------------------------------------------------------
# Main benchmark driver
# ---------------------------------------------------------------------------

def evaluate_real_world_kpis(
    episodes: int,
    policies: list[str],
    difficulties: list[str],
) -> dict:
    """Run episodes across policies × difficulties and return a full KPI report."""
    report: dict = {
        "episodes": episodes,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "policies": policies,
        "difficulties": difficulties,
        "difficulty_weights": DIFFICULTY_WEIGHTS,
        "results": {},
    }

    for policy in policies:
        policy_row: dict = {"by_difficulty": {}, "domain_kpis": {}}
        weighted_agg = {k: 0.0 for k in _AGGREGATE_KEYS}

        for difficulty in difficulties:
            cfg = TASK_CONFIG[difficulty]
            n_nodes = cfg["n_nodes"]
            max_steps = cfg["max_steps"]

            runs = [
                run_episode_trace(difficulty, TASK_SEEDS[difficulty] + i, policy)
                for i in range(episodes)
            ]

            aggregates = _compute_aggregates(runs, n_nodes, max_steps)

            policy_row["by_difficulty"][difficulty] = {
                "aggregates": {k: round(v, 4) for k, v in aggregates.items()},
                "domain_kpis": _score_domains(aggregates),
            }

            weight = DIFFICULTY_WEIGHTS.get(difficulty, 0.0)
            for key in weighted_agg:
                weighted_agg[key] += aggregates[key] * weight

        policy_row["domain_kpis"] = _score_domains(weighted_agg)
        report["results"][policy] = policy_row

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Real-world KPI evaluator")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument(
        "--policies", nargs="+", default=["wait", "random", "greedy_containment"],
    )
    parser.add_argument(
        "--difficulties", nargs="+", default=["easy", "medium", "hard"],
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/real_world_kpi_results.json"),
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
