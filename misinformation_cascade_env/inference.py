"""Baseline inference script for the OpenEnv hackathon evaluation.

Connects to an OpenAI-compatible endpoint (via HF Inference), runs
each benchmark task as a full episode, and reports grades in strict (0, 1).
"""

from __future__ import annotations

import os
import sys
from typing import Any, Optional

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore[assignment]

try:
    from .env import MisinformationCascadeEnv as Simulator
    from .models import CascadeAction, CascadeObservation
    from .prompt_utils import SYSTEM_PROMPT, build_user_prompt, parse_action_payload
    from .task_grader import clamp_score, grade_episode, is_task_success, resolve_tasks
except ImportError:
    from env import MisinformationCascadeEnv as Simulator
    from models import CascadeAction, CascadeObservation
    from prompt_utils import SYSTEM_PROMPT, build_user_prompt, parse_action_payload
    from task_grader import clamp_score, grade_episode, is_task_success, resolve_tasks


def sanitize_log_value(value: str) -> str:
    """Compact newlines and tabs in a log value for single-line output."""
    return value.replace("\n", " ").replace("\t", " ").replace("\r", " ")

# ── Config ────────────────────────────────────────────────────────────────

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

# ── Structured Logging (validator format) ─────────────────────────────────


def _fmt_action(a: CascadeAction) -> str:
    if a.action_type == "WAIT":
        return "wait()"
    return f"{a.action_type.lower()}('{a.target_node_id}')"


def _safe(v: Optional[str]) -> str:
    return "null" if v is None else " ".join(str(v).split())


def _log_start(task_id: str) -> None:
    print(f"[START] task={task_id} env={BENCHMARK} model={MODEL_NAME}", flush=True)


def _log_step(step: int, action: CascadeAction, reward: float,
              done: bool, error: Optional[str]) -> None:
    print(
        f"[STEP] step={step} action={_fmt_action(action)} "
        f"reward={reward:.2f} done={str(done).lower()} "
        f"error={_safe(error)}",
        flush=True,
    )


def _log_end(success: bool, steps: int, rewards: list[float]) -> None:
    csv = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={csv}", flush=True)


# ── Client ────────────────────────────────────────────────────────────────


def build_openai_client() -> Any:
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN must be set for baseline inference.")
    if OpenAI is None:
        raise RuntimeError("openai package is missing. Install with `uv sync`.")
    return OpenAI(
        base_url=API_BASE_URL, api_key=HF_TOKEN,
        timeout=API_TIMEOUT_S, max_retries=API_MAX_RETRIES,
    )


# ── Action Selection ──────────────────────────────────────────────────────


def _visible_ids(obs: CascadeObservation) -> set[str]:
    return {n.node_id for n in (*obs.top_nodes, *obs.confirmed_infected, *obs.at_risk_nodes)}


def _wait(reason: str = "") -> CascadeAction:
    return CascadeAction(action_type="WAIT", reasoning=reason)


def pick_action(client: Any, obs: CascadeObservation) -> CascadeAction:
    """Ask the LLM for one action; fall back to WAIT on any failure."""
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME, temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS, timeout=API_TIMEOUT_S,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(obs)},
            ],
        )
        content = resp.choices[0].message.content or ""
        atype, target, reasoning = parse_action_payload(content)
        action = CascadeAction(action_type=atype, target_node_id=target, reasoning=reasoning)

        if action.action_type != "WAIT" and action.target_node_id not in _visible_ids(obs):
            return _wait(f"Target '{action.target_node_id}' not visible.")
        return action

    except Exception as exc:
        return _wait(f"Fallback: {exc.__class__.__name__}")


# ── Error Extraction ──────────────────────────────────────────────────────


def _parse_error(effect: str) -> Optional[str]:
    if "INVALID ACTION" not in effect:
        return None
    detail = effect.split("\u2014", 1)[-1]  # split on em-dash
    return detail.split("Step consumed", 1)[0].strip() or None


# ── Episode Runner ────────────────────────────────────────────────────────


def _step(env: Simulator, action: CascadeAction) -> tuple[CascadeObservation, float]:
    obs = env.step(action)
    return obs, float(obs.reward or 0.0)


def run_task(task, client: Any) -> float:
    """Run one episode end-to-end.  Always returns a score in strict (0, 1)."""
    env = Simulator(difficulty=task.difficulty, seed=task.seed)
    obs = env.reset(seed=task.seed)
    rewards: list[float] = []
    steps = 0
    success = False

    _log_start(task.task_id)
    try:
        # Agent-driven phase
        while not obs.done and steps < task.max_steps:
            action = pick_action(client, obs)
            obs, r = _step(env, action)
            steps += 1
            rewards.append(r)
            _log_step(steps, action, r, obs.done, _parse_error(obs.last_action_effect))

        # Force terminal state
        while not obs.done:
            w = _wait("Forcing terminal.")
            obs, r = _step(env, w)
            steps += 1
            rewards.append(r)
            _log_step(steps, w, r, obs.done, _parse_error(obs.last_action_effect))

        score = grade_episode(obs, rewards)
        success = is_task_success(task, score)

    except Exception as exc:
        print(f"[stderr] task={task.task_id} error={type(exc).__name__}:{exc}",
              file=sys.stderr, flush=True)
        score = clamp_score(0.0)

    finally:
        _log_end(success, steps, rewards)

    return score


# ── Entry Point ───────────────────────────────────────────────────────────


def main() -> None:
    tasks = resolve_tasks(TASK_SELECTOR)
    client = build_openai_client()
    scores = [run_task(t, client) for t in tasks]

    avg = sum(scores) / max(1, len(scores))
    print(f"[stderr] aggregate_avg_score={avg:.4f}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
