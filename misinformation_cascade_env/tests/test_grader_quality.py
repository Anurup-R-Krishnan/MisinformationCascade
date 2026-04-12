import random
from statistics import mean

from misinformation_cascade_env.env import MisinformationCascadeEnv
from misinformation_cascade_env.evaluate import greedy_containment_policy, wait_policy
from misinformation_cascade_env.models import TASK_SEEDS
from misinformation_cascade_env.task_grader import grade_episode


def _run_episode(difficulty: str, seed: int, policy_fn):
    env = MisinformationCascadeEnv(difficulty=difficulty, seed=seed)
    obs = env.reset(seed=seed)
    rewards = []
    rng = random.Random(seed + 17_000)

    while not obs.done:
        action = policy_fn(obs, rng)
        obs = env.step(action)
        rewards.append(float(obs.reward))

    return obs, rewards


def test_grader_is_deterministic_for_same_episode_trace():
    obs, rewards = _run_episode("medium", TASK_SEEDS["medium"], greedy_containment_policy)

    score_a = grade_episode(obs, rewards)
    score_b = grade_episode(obs, rewards)
    assert score_a == score_b


def test_grader_is_not_constant_across_policy_quality():
    base = TASK_SEEDS["easy"]
    wait_scores = []
    greedy_scores = []
    for i in range(8):
        obs_wait, rewards_wait = _run_episode("easy", base + i, wait_policy)
        obs_greedy, rewards_greedy = _run_episode(
            "easy", base + i, greedy_containment_policy
        )
        wait_scores.append(grade_episode(obs_wait, rewards_wait))
        greedy_scores.append(grade_episode(obs_greedy, rewards_greedy))

    wait_avg = mean(wait_scores)
    greedy_avg = mean(greedy_scores)

    assert 0.0 <= wait_avg <= 1.0
    assert 0.0 <= greedy_avg <= 1.0
    assert greedy_avg > wait_avg


def test_difficulty_progression_with_fixed_policy():
    def avg_score(difficulty: str) -> float:
        base = TASK_SEEDS[difficulty]
        scores = []
        for i in range(5):
            obs, rewards = _run_episode(difficulty, base + i, greedy_containment_policy)
            scores.append(grade_episode(obs, rewards))
        return mean(scores)

    easy = avg_score("easy")
    medium = avg_score("medium")
    hard = avg_score("hard")

    assert easy > medium > hard
