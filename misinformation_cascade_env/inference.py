"""
Baseline inference script for OpenEnv hackathon evaluation.

Requirements covered:
- Uses OpenAI client for model calls (`API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`)
- Emits strict structured logs:
  [START] / [STEP] / [END]
- Runs all three benchmark tasks (easy/medium/hard) by default
"""

from __future__ import annotations

import os
import sys
from typing import Any, Optional

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - handled at runtime for offline mode
    OpenAI = None  # type: ignore[assignment]

try:
    from .env import MisinformationCascadeEnv as CascadeSimulator
    from .models import CascadeAction, CascadeObservation
    from .offline_advisor import choose_action as choose_offline_action
    from .prompt_utils import SYSTEM_PROMPT, build_user_prompt, parse_action_payload
    from .task_grader import grade_episode, is_task_success, resolve_tasks
except ImportError:
    from env import MisinformationCascadeEnv as CascadeSimulator
    from models import CascadeAction, CascadeObservation
    from offline_advisor import choose_action as choose_offline_action
    from prompt_utils import SYSTEM_PROMPT, build_user_prompt, parse_action_payload
    from task_grader import grade_episode, is_task_success, resolve_tasks


API_KEY = (
    os.getenv("HF_TOKEN")
    or os.getenv("API_KEY")
    or os.getenv("OPENAI_API_KEY")
    or os.getenv("GROQ_API_KEY")
)
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
BENCHMARK = os.getenv("MY_ENV_V4_BENCHMARK") or "misinformation_cascade_env"
TASK_SELECTOR = os.getenv("CASCADE_TASKS") or "easy,medium,hard"
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "180"))


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def build_openai_client() -> Optional[Any]:
    if not API_KEY:
        return None
    if OpenAI is None:
        raise RuntimeError(
            "HF_TOKEN/API_KEY is set but `openai` package is missing. Install with `uv sync`."
        )
    return OpenAI(base_url=API_BASE_URL, api_key=API_KEY)


def pick_action(
    client: Optional[Any],
    observation: CascadeObservation,
) -> CascadeAction:
    fallback = choose_offline_action(observation)
    if client is None:
        return fallback

    user_prompt = build_user_prompt(observation)
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = response.choices[0].message.content or ""
        action_type, target_node_id, reasoning = parse_action_payload(content)
        action = CascadeAction(
            action_type=action_type,
            target_node_id=target_node_id,
            reasoning=reasoning,
        )
        return sanitize_action(observation, action, fallback)
    except Exception as exc:
        fallback.reasoning = f"LLM fallback: {exc.__class__.__name__}"
        return fallback


def sanitize_action(
    observation: CascadeObservation,
    action: CascadeAction,
    fallback: CascadeAction,
) -> CascadeAction:
    if action.action_type == "WAIT":
        return action

    known_ids = {
        n.node_id
        for n in (
            observation.top_nodes
            + observation.confirmed_infected
            + observation.at_risk_nodes
        )
    }
    if action.target_node_id in known_ids:
        return action
    return fallback


def action_to_log(action: CascadeAction) -> str:
    if action.action_type == "WAIT":
        return "wait()"
    return f"{action.action_type.lower()}('{action.target_node_id}')"


def extract_last_action_error(last_action_effect: str) -> Optional[str]:
    if "INVALID ACTION" not in last_action_effect:
        return None
    if "—" in last_action_effect:
        detail = last_action_effect.split("—", 1)[1]
    else:
        detail = last_action_effect
    return detail.split("Step consumed", 1)[0].strip()


def run_task(task, client: Optional[Any]) -> float:
    env = CascadeSimulator(difficulty=task.difficulty, seed=task.seed)
    observation = env.reset(seed=task.seed)
    rewards: list[float] = []
    steps = 0
    grade_score = 0.0
    success = False

    log_start(task=task.task_id, env=BENCHMARK, model=MODEL_NAME)
    try:
        while not observation.done and steps < task.max_steps:
            action = pick_action(client, observation)
            observation = env.step(action)
            steps += 1
            reward = float(observation.reward or 0.0)
            rewards.append(reward)
            log_step(
                step=steps,
                action=action_to_log(action),
                reward=reward,
                done=bool(observation.done),
                error=extract_last_action_error(observation.last_action_effect),
            )

        if not observation.done:
            while not observation.done:
                action = CascadeAction(action_type="WAIT", reasoning="Force terminal state.")
                observation = env.step(action)
                steps += 1
                reward = float(observation.reward or 0.0)
                rewards.append(reward)
                log_step(
                    step=steps,
                    action=action_to_log(action),
                    reward=reward,
                    done=bool(observation.done),
                    error=extract_last_action_error(observation.last_action_effect),
                )

        grade_score = grade_episode(observation, rewards)
        success = is_task_success(task, grade_score)
    except Exception as exc:
        print(
            f"[stderr] task={task.task_id} error={exc.__class__.__name__}:{exc}",
            file=sys.stderr,
            flush=True,
        )
    finally:
        log_end(success=success, steps=steps, score=grade_score, rewards=rewards)

    return grade_score


def main() -> None:
    tasks = resolve_tasks(TASK_SELECTOR)
    client = build_openai_client()

    if client is None:
        print("[stderr] HF_TOKEN/API_KEY not set; using offline advisor.", file=sys.stderr, flush=True)

    scores: list[float] = []
    for task in tasks:
        scores.append(run_task(task, client))

    # Keep final output machine-friendly while still exposing aggregate.
    avg = sum(scores) / max(1, len(scores))
    print(f"[stderr] aggregate_avg_score={avg:.4f}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
