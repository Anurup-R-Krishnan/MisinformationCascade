"""
Baseline inference script for OpenEnv hackathon evaluation.

Requirements covered:
- Uses OpenAI client for model calls (API_BASE_URL, MODEL_NAME, HF_TOKEN)
- Emits strict structured logs: [START] / [STEP] / [END]
- Runs all three benchmark tasks (easy/medium/hard) by default
- All task scores are strictly within (0, 1) — never 0.0 or 1.0.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Optional

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - handled at runtime
    OpenAI = None  # type: ignore[assignment]

try:
    from .env import MisinformationCascadeEnv as CascadeSimulator
    from .models import CascadeAction, CascadeObservation
    from .prompt_utils import SYSTEM_PROMPT, build_user_prompt, parse_action_payload
    from .task_grader import grade_episode, is_task_success, resolve_tasks, clamp_score
except ImportError:
    from env import MisinformationCascadeEnv as CascadeSimulator
    from models import CascadeAction, CascadeObservation
    from prompt_utils import SYSTEM_PROMPT, build_user_prompt, parse_action_payload
    from task_grader import grade_episode, is_task_success, resolve_tasks, clamp_score

# ---------------------------------------------------------------------------
# Configuration from environment variables
# ---------------------------------------------------------------------------

HF_TOKEN = os.getenv("HF_TOKEN")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
BENCHMARK = os.getenv("MY_ENV_V4_BENCHMARK") or "misinformation_cascade_env"
TASK_SELECTOR = os.getenv("CASCADE_TASKS") or "easy,medium,hard"
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "180"))
API_TIMEOUT_S = float(os.getenv("API_TIMEOUT_S", "20"))
API_MAX_RETRIES = int(os.getenv("API_MAX_RETRIES", "1"))

# WAIT action used for fallback / forced termination — avoids repeated construction.
_WAIT_ACTION = CascadeAction(action_type="WAIT", reasoning="Fallback.")


# ---------------------------------------------------------------------------
# Structured logging helpers (validator-required format)
# ---------------------------------------------------------------------------

def _sanitize(value: Optional[str]) -> str:
    """Collapse whitespace for safe single-line log output."""
    if value is None:
        return "null"
    return " ".join(str(value).split())


def _log_start(task_id: str) -> None:
    print(f"[START] task={task_id} env={BENCHMARK} model={MODEL_NAME}", flush=True)


def _log_step(
    step: int,
    action: CascadeAction,
    reward: float,
    done: bool,
    error: Optional[str],
) -> None:
    action_str = (
        "wait()"
        if action.action_type == "WAIT"
        else f"{action.action_type.lower()}('{action.target_node_id}')"
    )
    error_str = _sanitize(error) if error else "null"
    print(
        f"[STEP] step={step} action={action_str} "
        f"reward={reward:.2f} done={str(done).lower()} error={error_str}",
        flush=True,
    )


def _log_end(success: bool, steps: int, rewards: list[float]) -> None:
    rewards_csv = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} rewards={rewards_csv}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# OpenAI client builder
# ---------------------------------------------------------------------------

def build_openai_client() -> Any:
    """Build and return an OpenAI-compatible client for HF Inference."""
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN must be set for baseline inference.")
    if OpenAI is None:
        raise RuntimeError(
            "HF_TOKEN is set but `openai` package is missing. Install with `uv sync`."
        )
    return OpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN,
        timeout=API_TIMEOUT_S,
        max_retries=API_MAX_RETRIES,
    )


# ---------------------------------------------------------------------------
# Action selection
# ---------------------------------------------------------------------------

def _visible_node_ids(obs: CascadeObservation) -> set[str]:
    """Return all node IDs currently visible to the agent."""
    return {
        n.node_id
        for n in (*obs.top_nodes, *obs.confirmed_infected, *obs.at_risk_nodes)
    }


def pick_action(client: Any, observation: CascadeObservation) -> CascadeAction:
    """Query the LLM for a single action, with graceful fallback to WAIT."""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            timeout=API_TIMEOUT_S,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(observation)},
            ],
        )
        content = response.choices[0].message.content or ""
        action_type, target_node_id, reasoning = parse_action_payload(content)
        action = CascadeAction(
            action_type=action_type,
            target_node_id=target_node_id,
            reasoning=reasoning,
        )

        # Validate that the target node is actually visible to the agent.
        if action.action_type != "WAIT" and action.target_node_id not in _visible_node_ids(observation):
            return CascadeAction(
                action_type="WAIT",
                reasoning=f"Target '{action.target_node_id}' not visible; fallback WAIT.",
            )
        return action

    except Exception as exc:
        return CascadeAction(
            action_type="WAIT",
            reasoning=f"Model call fallback: {exc.__class__.__name__}",
        )


# ---------------------------------------------------------------------------
# Error extraction from env feedback
# ---------------------------------------------------------------------------

def _extract_action_error(effect: str) -> Optional[str]:
    """Pull the human-readable error from an env last_action_effect string, if any."""
    if "INVALID ACTION" not in effect:
        return None
    # Split on em-dash or fall back to full text.
    detail = effect.split("—", 1)[-1]
    return detail.split("Step consumed", 1)[0].strip() or None


# ---------------------------------------------------------------------------
# Core episode runner
# ---------------------------------------------------------------------------

def _execute_step(
    env: CascadeSimulator,
    action: CascadeAction,
) -> tuple[CascadeObservation, float]:
    """Execute a single step and return (observation, reward)."""
    obs = env.step(action)
    reward = float(obs.reward or 0.0)
    return obs, reward


def run_task(task, client: Any) -> float:
    """Run a complete task episode and return a grade score in strict (0, 1).

    The function guarantees the returned score is NEVER exactly 0.0 or 1.0,
    even if an exception aborts the episode early.
    """
    env = CascadeSimulator(difficulty=task.difficulty, seed=task.seed)
    observation = env.reset(seed=task.seed)
    rewards: list[float] = []
    steps = 0
    success = False

    _log_start(task.task_id)

    try:
        # Phase 1: Agent-driven steps (LLM picks actions)
        while not observation.done and steps < task.max_steps:
            action = pick_action(client, observation)
            observation, reward = _execute_step(env, action)
            steps += 1
            rewards.append(reward)
            _log_step(
                step=steps,
                action=action,
                reward=reward,
                done=observation.done,
                error=_extract_action_error(observation.last_action_effect),
            )

        # Phase 2: Force termination if episode didn't end naturally
        while not observation.done:
            wait = CascadeAction(action_type="WAIT", reasoning="Forcing terminal state.")
            observation, reward = _execute_step(env, wait)
            steps += 1
            rewards.append(reward)
            _log_step(
                step=steps,
                action=wait,
                reward=reward,
                done=observation.done,
                error=_extract_action_error(observation.last_action_effect),
            )

        # Compute grade — grade_episode already returns strict (0, 1).
        grade_score = grade_episode(observation, rewards)
        success = is_task_success(task, grade_score)

    except Exception as exc:
        print(
            f"[stderr] task={task.task_id} error={exc.__class__.__name__}:{exc}",
            file=sys.stderr,
            flush=True,
        )
        # Guarantee a valid score even on failure.
        grade_score = clamp_score(0.0)

    finally:
        _log_end(success=success, steps=steps, rewards=rewards)

    return grade_score


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run all selected tasks and report aggregate score."""
    tasks = resolve_tasks(TASK_SELECTOR)
    client = build_openai_client()

    scores = [run_task(task, client) for task in tasks]

    avg = sum(scores) / max(1, len(scores))
    print(f"[stderr] aggregate_avg_score={avg:.4f}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
