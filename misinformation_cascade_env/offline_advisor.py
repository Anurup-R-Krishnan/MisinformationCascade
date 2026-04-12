"""
Deterministic fallback advisor for local/offline baseline execution.
"""

from __future__ import annotations

try:
    from .models import ACTION_COSTS, CascadeAction, CascadeObservation
except ImportError:
    from models import ACTION_COSTS, CascadeAction, CascadeObservation


def _is_emergency(observation: CascadeObservation) -> bool:
    infected_ratio = observation.infected_count / max(1, observation.total_nodes)
    return infected_ratio >= 0.25 or observation.spread_delta_last_step >= 3


def _can_spend(cost: int, observation: CascadeObservation, emergency: bool) -> bool:
    if observation.budget_remaining < cost:
        return False
    if emergency:
        return True

    steps_left = max(1, observation.max_steps - observation.step + 1)
    budget_per_step = observation.budget_remaining / steps_left
    return cost <= (budget_per_step * 1.8)


def choose_action(observation: CascadeObservation) -> CascadeAction:
    confirmed = sorted(
        observation.confirmed_infected,
        key=lambda n: n.influence_score,
        reverse=True,
    )
    at_risk = sorted(
        observation.at_risk_nodes,
        key=lambda n: n.influence_score,
        reverse=True,
    )
    top_nodes = sorted(
        observation.top_nodes,
        key=lambda n: n.influence_score,
        reverse=True,
    )

    budget = observation.budget_remaining
    emergency = _is_emergency(observation)
    is_small_graph = observation.total_nodes <= 20 and observation.max_steps <= 15
    top_confirmed = confirmed[0] if confirmed else None
    top_at_risk = at_risk[0] if at_risk else None
    top = top_nodes[0] if top_nodes else None

    if (
        top_confirmed
        and (
            (is_small_graph and observation.budget_remaining >= ACTION_COSTS["QUARANTINE"])
            or _can_spend(ACTION_COSTS["QUARANTINE"], observation, emergency)
        )
        and (is_small_graph or emergency or top_confirmed.influence_score >= 0.65)
    ):
        return CascadeAction(
            action_type="QUARANTINE",
            target_node_id=top_confirmed.node_id,
            reasoning="Emergency/high-impact confirmed spreader containment.",
        )

    if (
        top_at_risk
        and _can_spend(ACTION_COSTS["INOCULATE"], observation, emergency)
        and (emergency or top_at_risk.influence_score >= 0.6)
    ):
        return CascadeAction(
            action_type="INOCULATE",
            target_node_id=top_at_risk.node_id,
            reasoning="Protect high-risk influential node before confirmation.",
        )

    if top and _can_spend(ACTION_COSTS["BOOST_CORRECTION"], observation, emergency):
        return CascadeAction(
            action_type="BOOST_CORRECTION",
            target_node_id=top.node_id,
            reasoning="Reduce spread pressure on influential node while pacing budget.",
        )

    if top_at_risk and budget >= ACTION_COSTS["FACTCHECK"]:
        return CascadeAction(
            action_type="FACTCHECK",
            target_node_id=top_at_risk.node_id,
            reasoning="Low-cost probe while preserving budget runway.",
        )

    if top_confirmed and budget >= ACTION_COSTS["QUARANTINE"]:
        return CascadeAction(
            action_type="QUARANTINE",
            target_node_id=top_confirmed.node_id,
            reasoning="Fallback containment for confirmed spreader.",
        )

    return CascadeAction(action_type="WAIT", reasoning="No cost-feasible action.")
