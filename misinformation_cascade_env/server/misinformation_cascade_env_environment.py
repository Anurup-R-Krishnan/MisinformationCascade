# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
OpenEnv server adapter for the misinformation cascade simulator.
"""

from __future__ import annotations

from typing import Any, Optional

from openenv.core.env_server.interfaces import Environment

try:
    from ..env import MisinformationCascadeEnv as CascadeSimulator
    from ..models import CascadeAction, CascadeObservation, CascadeState
except ImportError:
    from env import MisinformationCascadeEnv as CascadeSimulator
    from models import CascadeAction, CascadeObservation, CascadeState


class MisinformationCascadeEnvironment(
    Environment[CascadeAction, CascadeObservation, CascadeState]
):
    """
    OpenEnv-compatible environment implementation for the cascade task.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, difficulty: str = "medium", seed: Optional[int] = None):
        self._difficulty = difficulty
        self._seed = seed
        self._sim = CascadeSimulator(difficulty=difficulty, seed=seed)
        self._last_seed: Optional[int] = seed

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> CascadeObservation:
        if "difficulty" in kwargs and kwargs["difficulty"] != self._difficulty:
            self._difficulty = kwargs["difficulty"]
            self._sim = CascadeSimulator(difficulty=self._difficulty, seed=seed)

        # episode_id is accepted for OpenEnv contract parity; simulator generates ids.
        _ = episode_id

        self._last_seed = seed if seed is not None else self._seed
        return self._sim.reset(seed=self._last_seed)

    def step(
        self,
        action: CascadeAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> CascadeObservation:
        _ = timeout_s
        _ = kwargs
        return self._sim.step(action)

    @property
    def state(self) -> CascadeState:
        return CascadeState.model_validate(self._sim.state())
