from misinformation_cascade_env.models import CascadeObservation, NodeSummary
from misinformation_cascade_env.offline_advisor import choose_action


def _node(node_id: str, influence: float, status: str) -> NodeSummary:
    kwargs = {
        "node_id": node_id,
        "influence_score": influence,
        "status": status,
        "boost_active": False,
    }
    if status == "AT_RISK":
        kwargs["infected_neighbor"] = "n_inf"
        kwargs["turns_at_risk"] = 1
    return NodeSummary(**kwargs)


def _observation(
    *,
    budget_remaining: int,
    step: int,
    max_steps: int,
    infected_count: int,
    spread_delta_last_step: int,
) -> CascadeObservation:
    return CascadeObservation(
        top_nodes=[_node("n_top", 0.95, "SUSCEPTIBLE")],
        confirmed_infected=[_node("n_conf", 0.6, "CONFIRMED_INFECTED")],
        at_risk_nodes=[_node("n_risk", 0.8, "AT_RISK")],
        budget_remaining=budget_remaining,
        step=step,
        max_steps=max_steps,
        total_nodes=50,
        infected_count=infected_count,
        inoculated_count=0,
        quarantined_count=0,
        spread_delta_last_step=spread_delta_last_step,
        last_action_effect="",
        reward=0.0,
        done=False,
    )


def test_choose_action_uses_budget_pacing_in_non_emergency():
    obs = _observation(
        budget_remaining=5,
        step=2,
        max_steps=25,
        infected_count=4,
        spread_delta_last_step=0,
    )

    action = choose_action(obs)
    assert action.action_type in {"BOOST_CORRECTION", "FACTCHECK", "INOCULATE"}
    assert action.action_type != "QUARANTINE"


def test_choose_action_prefers_quarantine_in_emergency():
    obs = _observation(
        budget_remaining=10,
        step=3,
        max_steps=25,
        infected_count=20,
        spread_delta_last_step=4,
    )

    action = choose_action(obs)
    assert action.action_type == "QUARANTINE"
    assert action.target_node_id == "n_conf"
