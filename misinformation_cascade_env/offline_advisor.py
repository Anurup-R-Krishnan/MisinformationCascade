"""
Deterministic fallback advisor for local/offline baseline execution.
"""

from __future__ import annotations

try:
    from .models import ACTION_COSTS, CascadeAction, CascadeObservation
except ImportError:
    from models import ACTION_COSTS, CascadeAction, CascadeObservation


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

    if confirmed and budget >= ACTION_COSTS["QUARANTINE"]:
        return CascadeAction(
            action_type="QUARANTINE",
            target_node_id=confirmed[0].node_id,
            reasoning="Contain highest-impact confirmed spreader first.",
        )

    if at_risk and budget >= ACTION_COSTS["INOCULATE"]:
        return CascadeAction(
            action_type="INOCULATE",
            target_node_id=at_risk[0].node_id,
            reasoning="Prevent likely infection on most influential exposed node.",
        )

    if top_nodes and budget >= ACTION_COSTS["BOOST_CORRECTION"]:
        return CascadeAction(
            action_type="BOOST_CORRECTION",
            target_node_id=top_nodes[0].node_id,
            reasoning="Lower transmission chance on the most influential node.",
        )

    if at_risk and budget >= ACTION_COSTS["FACTCHECK"]:
        return CascadeAction(
            action_type="FACTCHECK",
            target_node_id=at_risk[0].node_id,
            reasoning="Gather state information while preserving budget.",
        )

    return CascadeAction(action_type="WAIT", reasoning="No cost-feasible action.")
