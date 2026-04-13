"""Task registry and deterministic grading for the misinformation cascade benchmark.

Every exported score is guaranteed to lie in the open interval (0, 1)
as required by the Phase-2 validator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

try:
    from .models import STARTING_BUDGET, TASK_CONFIG, TASK_SEEDS, CascadeObservation
except ImportError:
    from models import STARTING_BUDGET, TASK_CONFIG, TASK_SEEDS, CascadeObservation

# Phase-2 validator requires strict (0, 1).  Wide margin for safety.
SCORE_EPSILON: float = 1e-6


def clamp_score(value: float) -> float:
    """Clamp *value* into the strict open interval (ε, 1 − ε)."""
    return max(SCORE_EPSILON, min(value, 1.0 - SCORE_EPSILON))


# ── Task Registry ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CascadeTask:
    task_id: str
    difficulty: str
    seed: int
    success_threshold: float
    description: str

    @property
    def max_steps(self) -> int:
        return int(TASK_CONFIG[self.difficulty]["max_steps"])


TASKS: tuple[CascadeTask, ...] = (
    CascadeTask(
        task_id="cascade-easy",
        difficulty="easy",
        seed=TASK_SEEDS["easy"],
        success_threshold=0.62,
        description="Contain spread on a small Erdos-Renyi graph with limited seeding.",
    ),
    CascadeTask(
        task_id="cascade-medium",
        difficulty="medium",
        seed=TASK_SEEDS["medium"],
        success_threshold=0.40,
        description="Balance budget vs spread on a denser medium-sized graph.",
    ),
    CascadeTask(
        task_id="cascade-hard",
        difficulty="hard",
        seed=TASK_SEEDS["hard"],
        success_threshold=0.20,
        description="Handle hub-heavy graph dynamics with periodic external seeding.",
    ),
)

_BY_ID: dict[str, CascadeTask] = {t.task_id: t for t in TASKS}
_BY_DIFF: dict[str, CascadeTask] = {t.difficulty: t for t in TASKS}


def list_tasks() -> list[CascadeTask]:
    return list(TASKS)


def resolve_tasks(selector: str | None) -> list[CascadeTask]:
    """Resolve a comma-separated selector (task ids or difficulty names).

    Returns all tasks when *selector* is falsy.
    """
    if not selector:
        return list_tasks()

    selected: list[CascadeTask] = []
    seen: set[str] = set()

    for key in (k.strip() for k in selector.split(",") if k.strip()):
        task = _BY_ID.get(key) or _BY_DIFF.get(key)
        if task is None:
            raise ValueError(
                f"Unknown task selector '{key}'. Valid: {sorted([*_BY_ID, *_BY_DIFF])}"
            )
        if task.task_id not in seen:
            selected.append(task)
            seen.add(task.task_id)

    return selected


# ── Episode Grading ───────────────────────────────────────────────────────


def grade_episode(
    final_obs: CascadeObservation,
    step_rewards: Iterable[float],
) -> float:
    """Deterministic grader → strict (0, 1).

    Blend:
      70 % terminal containment (env counterfactual score)
      15 % infection minimisation (1 − infected / total)
      10 % resource efficiency   (budget_remaining / starting)
       5 % trajectory stability  (fraction of non-negative reward steps)
    """
    rewards = list(step_rewards)

    terminal = _sat(float(final_obs.reward))
    infection = final_obs.infected_count / max(1, final_obs.total_nodes)
    resource = _sat(final_obs.budget_remaining / max(1, STARTING_BUDGET))

    n = max(1, len(rewards))
    stability = _sat(1.0 - sum(r < 0 for r in rewards) / n)

    raw = (
        0.70 * terminal + 0.15 * (1.0 - infection) + 0.10 * resource + 0.05 * stability
    )
    return round(clamp_score(raw), 4)


def is_task_success(task: CascadeTask, score: float) -> bool:
    return score >= task.success_threshold


def _sat(v: float) -> float:
    """Saturate to [0, 1] for internal arithmetic."""
    return max(0.0, min(v, 1.0))
