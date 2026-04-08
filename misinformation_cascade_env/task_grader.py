"""
Task registry and deterministic graders for the misinformation cascade benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

try:
    from .models import STARTING_BUDGET, TASK_CONFIG, TASK_SEEDS, CascadeObservation
except ImportError:
    from models import STARTING_BUDGET, TASK_CONFIG, TASK_SEEDS, CascadeObservation


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
        success_threshold=0.56,
        description="Balance budget vs spread on a denser medium-sized graph.",
    ),
    CascadeTask(
        task_id="cascade-hard",
        difficulty="hard",
        seed=TASK_SEEDS["hard"],
        success_threshold=0.50,
        description="Handle hub-heavy graph dynamics with periodic external seeding.",
    ),
)


TASKS_BY_ID: dict[str, CascadeTask] = {task.task_id: task for task in TASKS}
TASKS_BY_DIFFICULTY: dict[str, CascadeTask] = {task.difficulty: task for task in TASKS}


def list_tasks() -> list[CascadeTask]:
    return list(TASKS)


def resolve_tasks(task_selector: str | None) -> list[CascadeTask]:
    """
    Resolve a comma-separated task selector into concrete tasks.

    Supports:
        - task ids: cascade-easy,cascade-hard
        - difficulties: easy,medium,hard
        - defaults to all tasks when selector is empty
    """
    if not task_selector:
        return list_tasks()

    selected: list[CascadeTask] = []
    for raw in task_selector.split(","):
        key = raw.strip()
        if not key:
            continue
        if key in TASKS_BY_ID:
            selected.append(TASKS_BY_ID[key])
            continue
        if key in TASKS_BY_DIFFICULTY:
            selected.append(TASKS_BY_DIFFICULTY[key])
            continue
        valid = sorted(list(TASKS_BY_ID) + list(TASKS_BY_DIFFICULTY))
        raise ValueError(f"Unknown task selector '{key}'. Valid values: {valid}")
    return selected


def grade_episode(
    final_observation: CascadeObservation,
    step_rewards: Iterable[float],
) -> float:
    """
    Deterministic 0.0-1.0 grader using outcome and trajectory quality.

    The final score blends:
      - terminal containment score (majority weight)
      - infection minimization
      - resource efficiency
      - trajectory stability (avoid consistently negative step rewards)
    """
    rewards = list(step_rewards)
    terminal_score = _clip01(float(final_observation.reward))
    infection_ratio = final_observation.infected_count / max(1, final_observation.total_nodes)
    resource_ratio = final_observation.budget_remaining / max(1, STARTING_BUDGET)

    neg_steps = sum(1 for r in rewards if r < 0.0)
    stability = 1.0 - (neg_steps / max(1, len(rewards)))

    score = (
        0.70 * terminal_score
        + 0.15 * (1.0 - infection_ratio)
        + 0.10 * _clip01(resource_ratio)
        + 0.05 * _clip01(stability)
    )
    return round(_clip01(score), 4)


def is_task_success(task: CascadeTask, grade_score: float) -> bool:
    return grade_score >= task.success_threshold


def _clip01(value: float) -> float:
    return min(max(value, 0.0), 1.0)
