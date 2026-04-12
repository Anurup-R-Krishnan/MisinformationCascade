"""Real-world KPI evaluator for the misinformation cascade environment.

Computes domain-specific KPIs (trust & safety, election integrity,
public health, crisis response, network resilience) by running
deterministic benchmark episodes and scoring outcomes.

All emitted scores lie in the strict open interval (0, 1).

Usage::

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
from misinformation_cascade_env.models import STARTING_BUDGET, TASK_CONFIG, TASK_SEEDS
from misinformation_cascade_env.task_grader import clamp_score

DIFFICULTY_WEIGHTS: dict[str, float] = {"easy": 0.2, "medium": 0.3, "hard": 0.5}

_AGG_KEYS = (
    "avg_reward", "avg_peak_ratio", "avg_auc_ratio",
    "avg_spread_ratio", "avg_first_action_ratio", "avg_budget_spent_ratio",
)

# ── Data ──────────────────────────────────────────────────────────────────


@dataclass
class EpisodeTrace:
    reward: float
    steps: int
    peak_infected: int
    avg_infected: float
    spread_events: int
    budget_spent: int
    first_action_step: int


# ── Episode Runner ────────────────────────────────────────────────────────


def _run_trace(difficulty: str, seed: int, policy: str) -> EpisodeTrace:
    env = MisinformationCascadeEnv(difficulty=difficulty, seed=seed)
    rng = random.Random(seed + 100_000)
    obs = env.reset(seed=seed)

    budget_init = obs.budget_remaining
    infected = [obs.infected_count]
    spreads = 0
    first_act: int | None = None
    steps = 0

    while not obs.done:
        action = POLICIES[policy](obs, rng)
        if first_act is None and action.action_type != "WAIT":
            first_act = obs.step
        obs = env.step(action)
        steps += 1
        infected.append(obs.infected_count)
        spreads += max(0, obs.spread_delta_last_step)

    return EpisodeTrace(
        reward=float(obs.reward),
        steps=steps,
        peak_infected=max(infected),
        avg_infected=mean(infected),
        spread_events=spreads,
        budget_spent=budget_init - obs.budget_remaining,
        first_action_step=first_act if first_act is not None else obs.max_steps + 1,
    )


# ── Aggregation ───────────────────────────────────────────────────────────


def _aggregate(runs: list[EpisodeTrace], n_nodes: int, max_steps: int) -> dict[str, float]:
    return {
        "avg_reward":             mean(r.reward for r in runs),
        "avg_peak_ratio":         mean(r.peak_infected / n_nodes for r in runs),
        "avg_auc_ratio":          mean(r.avg_infected / n_nodes for r in runs),
        "avg_spread_ratio":       mean(r.spread_events / n_nodes for r in runs),
        "avg_first_action_ratio": mean(min(1.0, r.first_action_step / max_steps) for r in runs),
        "avg_budget_spent_ratio": mean(r.budget_spent / STARTING_BUDGET for r in runs),
    }


# ── Domain KPI Scoring ────────────────────────────────────────────────────


def _domain_kpis(agg: dict[str, float]) -> dict[str, float]:
    """Five domain KPIs + overall utility; all clamped to strict (0, 1)."""
    c = clamp_score  # alias for readability

    reward     = c(agg["avg_reward"])
    peak       = c(1.0 - agg["avg_peak_ratio"])
    sustained  = c(1.0 - agg["avg_auc_ratio"])
    spread     = c(1.0 - agg["avg_spread_ratio"])
    speed      = c(1.0 - agg["avg_first_action_ratio"])
    budget_eff = c(1.0 - agg["avg_budget_spent_ratio"])

    ts = c(0.45 * reward + 0.35 * peak + 0.20 * sustained)
    ei = c(0.40 * reward + 0.35 * spread + 0.25 * peak)
    ph = c(0.40 * sustained + 0.30 * reward + 0.30 * speed)
    cr = c(0.45 * speed + 0.35 * peak + 0.20 * reward)
    nr = c(0.45 * reward + 0.30 * peak + 0.25 * budget_eff)
    ov = c(mean([ts, ei, ph, cr, nr]))

    return {
        "trust_safety":              round(ts, 4),
        "election_integrity":        round(ei, 4),
        "public_health":             round(ph, 4),
        "crisis_response":           round(cr, 4),
        "network_resilience":        round(nr, 4),
        "overall_real_world_utility": round(ov, 4),
    }


# ── Main Driver ───────────────────────────────────────────────────────────


def evaluate_real_world_kpis(
    episodes: int, policies: list[str], difficulties: list[str],
) -> dict:
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
        row: dict = {"by_difficulty": {}, "domain_kpis": {}}
        weighted = {k: 0.0 for k in _AGG_KEYS}

        for diff in difficulties:
            cfg = TASK_CONFIG[diff]
            runs = [
                _run_trace(diff, TASK_SEEDS[diff] + i, policy)
                for i in range(episodes)
            ]
            agg = _aggregate(runs, cfg["n_nodes"], cfg["max_steps"])

            row["by_difficulty"][diff] = {
                "aggregates": {k: round(v, 4) for k, v in agg.items()},
                "domain_kpis": _domain_kpis(agg),
            }
            w = DIFFICULTY_WEIGHTS.get(diff, 0.0)
            for k in weighted:
                weighted[k] += agg[k] * w

        row["domain_kpis"] = _domain_kpis(weighted)
        report["results"][policy] = row

    return report


# ── CLI ───────────────────────────────────────────────────────────────────


def main() -> None:
    p = argparse.ArgumentParser(description="Real-world KPI evaluator")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--policies", nargs="+", default=["wait", "random", "greedy_containment"])
    p.add_argument("--difficulties", nargs="+", default=["easy", "medium", "hard"])
    p.add_argument("--output", type=Path, default=Path("artifacts/real_world_kpi_results.json"))
    args = p.parse_args()

    for name in args.policies:
        if name not in POLICIES:
            raise ValueError(f"Unknown policy '{name}'. Available: {sorted(POLICIES)}")
    for d in args.difficulties:
        if d not in TASK_CONFIG:
            raise ValueError(f"Unknown difficulty '{d}'. Available: {sorted(TASK_CONFIG)}")

    data = evaluate_real_world_kpis(args.episodes, args.policies, args.difficulties)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2), encoding="utf-8")

    print(f"Saved: {args.output}")
    for pol in args.policies:
        kpi = data["results"][pol]["domain_kpis"]
        print(f"  [{pol}] utility={kpi['overall_real_world_utility']:.4f} "
              f"trust={kpi['trust_safety']:.4f} election={kpi['election_integrity']:.4f}")


if __name__ == "__main__":
    main()
