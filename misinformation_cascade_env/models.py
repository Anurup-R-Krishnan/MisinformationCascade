# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Pydantic models and configuration for the misinformation cascade environment.
"""

from __future__ import annotations

from typing import Literal, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, Field, computed_field, model_validator


VALID_ACTION_TYPES = {
    "FACTCHECK",
    "QUARANTINE",
    "INOCULATE",
    "BOOST_CORRECTION",
    "WAIT",
}

ACTION_COSTS: dict[str, int] = {
    "WAIT": 0,
    "FACTCHECK": 1,
    "BOOST_CORRECTION": 2,
    "INOCULATE": 3,
    "QUARANTINE": 5,
}

STARTING_BUDGET: int = 20
LATENT_DURATION: int = 3
BOOST_DURATION: int = 3
SATURATION_MERCY_THRESHOLD: float = 0.90

TASK_SEEDS: dict[str, int] = {
    "easy": 42,
    "medium": 137,
    "hard": 512,
}

TASK_CONFIG: dict[str, dict] = {
    "easy": {
        "n_nodes": 20,
        "n_initial_infected": 3,
        "max_steps": 15,
        "graph_type": "erdos_renyi",
        "er_edge_prob": 0.15,
        "base_spread_prob": 0.30,
        "external_seed_interval": None,
    },
    "medium": {
        "n_nodes": 35,
        "n_initial_infected": 5,
        "max_steps": 20,
        "graph_type": "erdos_renyi",
        "er_edge_prob": 0.12,
        "base_spread_prob": 0.30,
        "external_seed_interval": None,
    },
    "hard": {
        "n_nodes": 50,
        "n_initial_infected": 7,
        "max_steps": 25,
        "graph_type": "barabasi_albert",
        "ba_edges": 2,
        "base_spread_prob": 0.30,
        "external_seed_interval": 5,
    },
}


class GraphNode(BaseModel):
    """
    Internal node model used by the simulator.
    """

    node_id: str
    degree: int
    influence_score: float = Field(ge=0.0, le=1.0)
    virality_modifier: float = Field(ge=0.5, le=2.0)
    base_spread_prob: float = Field(ge=0.0, le=1.0)
    effective_spread_prob: float = Field(ge=0.0, le=1.0)

    status: Literal[
        "SUSCEPTIBLE",
        "LATENT",
        "CONFIRMED_INFECTED",
        "INOCULATED",
        "QUARANTINED",
    ] = "SUSCEPTIBLE"

    at_risk: bool = False
    turns_at_risk: int = Field(default=0, ge=0)
    infected_neighbor: Optional[str] = None
    boost_turns_remaining: int = Field(default=0, ge=0)


class CascadeAction(Action):
    action_type: Literal[
        "FACTCHECK",
        "QUARANTINE",
        "INOCULATE",
        "BOOST_CORRECTION",
        "WAIT",
    ]
    target_node_id: Optional[str] = None
    reasoning: str = Field(default="", max_length=500)

    @model_validator(mode="after")
    def validate_target(self) -> "CascadeAction":
        if self.action_type == "WAIT":
            object.__setattr__(self, "target_node_id", None)
        elif not self.target_node_id:
            raise ValueError(
                f"target_node_id is required for '{self.action_type}'. "
                "Only WAIT may omit it."
            )
        return self


class NodeSummary(BaseModel):
    node_id: str
    influence_score: float = Field(ge=0.0, le=1.0)
    status: Literal[
        "SUSCEPTIBLE",
        "AT_RISK",
        "CONFIRMED_INFECTED",
        "INOCULATED",
        "QUARANTINED",
    ]
    boost_active: bool = False
    infected_neighbor: Optional[str] = None
    turns_at_risk: Optional[int] = None

    @model_validator(mode="after")
    def enforce_field_consistency(self) -> "NodeSummary":
        if self.status == "AT_RISK":
            if self.infected_neighbor is None:
                raise ValueError(
                    f"AT_RISK node must have infected_neighbor set. node_id={self.node_id}"
                )
            if self.turns_at_risk is None:
                raise ValueError(
                    f"AT_RISK node must have turns_at_risk set. node_id={self.node_id}"
                )
        else:
            self.infected_neighbor = None
            self.turns_at_risk = None

        if self.status in ("QUARANTINED", "INOCULATED"):
            self.boost_active = False

        return self


class CascadeObservation(Observation):
    top_nodes: list[NodeSummary]
    confirmed_infected: list[NodeSummary]
    at_risk_nodes: list[NodeSummary]

    budget_remaining: int = Field(ge=0)
    step: int = Field(ge=1)
    max_steps: int = Field(ge=1)
    total_nodes: int = Field(ge=1)
    infected_count: int = Field(ge=0)
    inoculated_count: int = Field(ge=0)
    quarantined_count: int = Field(ge=0)

    spread_delta_last_step: int = Field(ge=0)
    last_action_effect: str

    reward: float = Field(default=0.0)
    done: bool = Field(default=False)

    @computed_field
    @property
    def steps_remaining(self) -> int:
        return max(0, self.max_steps - self.step)

    @model_validator(mode="after")
    def deduplicate_at_risk_nodes(self) -> "CascadeObservation":
        top_ids = {n.node_id for n in self.top_nodes}
        filtered = [n for n in self.at_risk_nodes if n.node_id not in top_ids]
        object.__setattr__(self, "at_risk_nodes", filtered)
        return self


class CascadeState(State):
    episode_id: str
    difficulty: Literal["easy", "medium", "hard"]
    step_count: int = Field(ge=0)
    budget: int = Field(ge=0)
    max_steps: int = Field(ge=1)

    susceptible: list[str]
    latent: list[str]
    confirmed_infected: list[str]
    inoculated: list[str]
    quarantined: list[str]

    total_damage_accumulated: float = Field(ge=0.0)
    null_trajectory: list[float]
    graph_node_link_data: dict

    termination_reason: Optional[Literal["eradication", "saturation", "max_steps"]] = (
        None
    )

    @model_validator(mode="after")
    def validate_null_trajectory_length(self) -> "CascadeState":
        expected = self.max_steps + 1
        actual = len(self.null_trajectory)
        if actual != expected:
            raise ValueError(
                f"null_trajectory must have max_steps+1={expected} entries, got {actual}."
            )
        return self
