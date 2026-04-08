import pytest

from misinformation_cascade_env.env import MisinformationCascadeEnv
from misinformation_cascade_env.models import CascadeAction
from misinformation_cascade_env.task_grader import grade_episode, list_tasks


def test_reset_returns_valid_observation():
    env = MisinformationCascadeEnv(difficulty="easy", seed=42)
    obs = env.reset(seed=42)

    assert obs.step == 1
    assert obs.max_steps == 15
    assert obs.total_nodes == 20
    assert obs.budget_remaining == 20
    assert obs.done is False


def test_wait_action_advances_episode():
    env = MisinformationCascadeEnv(difficulty="easy", seed=42)
    env.reset(seed=42)
    obs = env.step(CascadeAction(action_type="WAIT"))

    assert obs.step >= 1
    assert obs.budget_remaining == 20


def test_invalid_target_action_consumes_step_without_budget_spend():
    env = MisinformationCascadeEnv(difficulty="easy", seed=42)
    env.reset(seed=42)
    obs = env.step(
        CascadeAction(action_type="FACTCHECK", target_node_id="does_not_exist")
    )

    assert "INVALID ACTION" in obs.last_action_effect
    assert obs.budget_remaining == 20


def test_terminal_reward_is_bounded():
    env = MisinformationCascadeEnv(difficulty="easy", seed=42)
    obs = env.reset(seed=42)

    while not obs.done:
        obs = env.step(CascadeAction(action_type="WAIT"))

    assert 0.0 <= obs.reward <= 1.0


def test_non_terminal_reward_has_shaping_signal():
    env = MisinformationCascadeEnv(difficulty="easy", seed=42)
    env.reset(seed=42)
    obs = env.step(CascadeAction(action_type="WAIT"))

    assert "Step reward:" in obs.last_action_effect
    assert -0.25 <= obs.reward <= 0.25


def test_task_grader_and_registry_contract():
    tasks = list_tasks()
    assert len(tasks) >= 3

    env = MisinformationCascadeEnv(difficulty="easy", seed=42)
    obs = env.reset(seed=42)
    rewards = []
    while not obs.done:
        obs = env.step(CascadeAction(action_type="WAIT"))
        rewards.append(obs.reward)

    score = grade_episode(obs, rewards)
    assert 0.0 <= score <= 1.0


def test_step_after_done_raises_runtime_error():
    env = MisinformationCascadeEnv(difficulty="easy", seed=42)
    obs = env.reset(seed=42)
    while not obs.done:
        obs = env.step(CascadeAction(action_type="WAIT"))

    with pytest.raises(RuntimeError):
        env.step(CascadeAction(action_type="WAIT"))


def test_insufficient_budget_action_is_rejected_without_spend():
    env = MisinformationCascadeEnv(difficulty="easy", seed=42)
    obs = env.reset(seed=42)
    target_id = obs.top_nodes[0].node_id
    env._budget = 0  # test edge-case budget floor behavior

    obs = env.step(CascadeAction(action_type="INOCULATE", target_node_id=target_id))
    assert "Insufficient budget" in obs.last_action_effect
    assert obs.budget_remaining == 0
