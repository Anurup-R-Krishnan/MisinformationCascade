"""
Task registry and deterministic graders for the misinformation cascade benchmark.

All exported scores are guaranteed to lie in the open interval (0, 1) — never
exactly 0.0 or 1.0 — as required by the Phase-2 validator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

try:
    from .models import STARTING_BUDGET, TASK_CONFIG, TASK_SEEDS, CascadeObservation
except ImportError:
    from models import STARTING_BUDGET, TASK_CONFIG, TASK_SEEDS, CascadeObservation


# ---------------------------------------------------------------------------
# Score-boundary constant — Phase-2 validator requires strict (0, 1).
# All scores are rounded to 4 dp; epsilon must survive rounding.
# ---------------------------------------------------------------------------

SCORE_EPSILON = 1e-3  # 0.001 — safely above 4-decimal rounding threshold


# ---------------------------------------------------------------------------
# Clamping helpers (exported for use by inference.py / evaluate_realworld.py)
# ---------------------------------------------------------------------------

def clamp_score(value: float) -> float:
    """Clamp *value* into the strict open interval (SCORE_EPSILON, 1 − SCORE_EPSILON).

    Use this for ANY score that will be reported to the validator.
    """
    return min(max(value, SCORE_EPSILON), 1.0 - SCORE_EPSILON)


def _clamp01(value: float) -> float:
    """Clamp to the closed [0, 1] interval (internal arithmetic only)."""
    return min(max(value, 0.0), 1.0)


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

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

TASKS_BY_ID: dict[str, CascadeTask] = {t.task_id: t for t in TASKS}
TASKS_BY_DIFFICULTY: dict[str, CascadeTask] = {t.difficulty: t for t in TASKS}


def list_tasks() -> list[CascadeTask]:
    """Return all registered benchmark tasks."""
    return list(TASKS)


def resolve_tasks(task_selector: str | None) -> list[CascadeTask]:
    """Resolve a comma-separated selector (task ids or difficulty names) into tasks.

    Returns all tasks when *task_selector* is empty or ``None``.
    """
    if not task_selector:
        return list_tasks()

    selected: list[CascadeTask] = []
    seen: set[str] = set()

    for key in (k.strip() for k in task_selector.split(",")):
        if not key:
            continue

        task = TASKS_BY_ID.get(key) or TASKS_BY_DIFFICULTY.get(key)
        if task is None:
            valid = sorted([*TASKS_BY_ID, *TASKS_BY_DIFFICULTY])
            raise ValueError(f"Unknown task selector '{key}'. Valid values: {valid}")

        if task.task_id not in seen:
            selected.append(task)
            seen.add(task.task_id)

    return selected


# ---------------------------------------------------------------------------
# Episode grading
# ---------------------------------------------------------------------------

def grade_episode(
    final_observation: CascadeObservation,
    step_rewards: Iterable[float],
) -> float:
    """Deterministic grader returning a score in strict (0, 1).

    Blends four normalised components:
      - **terminal containment** (70 %) — the env's counterfactual score.
      - **infection minimisation** (15 %) — 1 minus the final infection ratio.
      - **resource efficiency** (10 %) — fraction of budget remaining.
      - **trajectory stability** (5 %) — fraction of steps with non-negative reward.
    """
    rewards = list(step_rewards)

    # 1. Terminal containment score (already in [ε, 1−ε] from the env).
    terminal = _clamp01(float(final_observation.reward))

    # 2. Infection minimisation — lower infected ratio → better.
    total = max(1, final_observation.total_nodes)
    infection_ratio = final_observation.infected_count / total

    # 3. Resource efficiency — higher remaining budget → better.
    resource_ratio = _clamp01(final_observation.budget_remaining / max(1, STARTING_BUDGET))

    # 4. Trajectory stability — fewer negative-reward steps → better.
    n_steps = max(1, len(rewards))
    neg_steps = sum(1 for r in rewards if r < 0.0)
    stability = _clamp01(1.0 - neg_steps / n_steps)

    raw = (
        0.70 * terminal
        + 0.15 * (1.0 - infection_ratio)
        + 0.10 * resource_ratio
        + 0.05 * stability
    )

    # Final clamp + round — guarantees strict (0, 1).
    return round(clamp_score(raw), 4)


def is_task_success(task: CascadeTask, grade_score: float) -> bool:
    """Return whether *grade_score* meets the task's success threshold."""
    return grade_score >= task.success_threshold
