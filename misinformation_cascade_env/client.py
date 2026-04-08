# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Misinformation Cascade Env Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from .models import CascadeAction, CascadeObservation, CascadeState


class MisinformationCascadeEnv(
    EnvClient[CascadeAction, CascadeObservation, CascadeState]
):
    """
    Client for the Misinformation Cascade Env Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with MisinformationCascadeEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.infected_count)
        ...
        ...     result = client.step(CascadeAction(action_type="WAIT"))
        ...     print(result.observation.budget_remaining)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = MisinformationCascadeEnv.from_docker_image("misinformation_cascade_env-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(CascadeAction(action_type="WAIT"))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: CascadeAction) -> Dict:
        """
        Convert CascadeAction to JSON payload for step message.

        Args:
            action: CascadeAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: Dict) -> StepResult[CascadeObservation]:
        """
        Parse server response into StepResult[CascadeObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with CascadeObservation
        """
        obs_data = payload.get("observation", {})
        observation = CascadeObservation.model_validate(
            {
                **obs_data,
                "done": payload.get("done", obs_data.get("done", False)),
                "reward": payload.get("reward", obs_data.get("reward", 0.0)),
            }
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> CascadeState:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return CascadeState.model_validate(payload)
