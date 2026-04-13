# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
OpenEnv client for the Misinformation Cascade environment.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from openenv.core.env_server.http_client import EnvClient

from misinformation_cascade_env.models import CascadeAction, CascadeObservation


class CascadeEnvClient:
    """HTTP client for the Misinformation Cascade environment."""

    def __init__(
        self,
        base_url: str,
        difficulty: str = "medium",
        seed: Optional[int] = None,
    ):
        from openenv.core.env_server.http_client import EnvClient

        self._client = EnvClient[CascadeAction, CascadeObservation](
            base_url=base_url,
            action_model=CascadeAction,
            observation_model=CascadeObservation,
        )
        self._difficulty = difficulty
        self._seed = seed

    def reset(self, seed: Optional[int] = None, **kwargs):
        return self._client.reset(
            seed=seed or self._seed, difficulty=self._difficulty, **kwargs
        )

    def step(self, action: CascadeAction, **kwargs):
        return self._client.step(action, **kwargs)

    def state(self, **kwargs):
        return self._client.state(**kwargs)

    def close(self):
        return self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


__all__ = ["CascadeEnvClient"]
