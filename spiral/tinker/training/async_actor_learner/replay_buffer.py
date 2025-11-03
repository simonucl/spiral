# Copyright 2025 SPIRAL Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Replay buffer for async actor-learner architecture."""

import asyncio
import logging
from collections import deque
from typing import List, Tuple

from tinker_cookbook.rl.types import EnvGroupBuilder, TrajectoryGroup

logger = logging.getLogger(__name__)


class ReplayBufferItem:
    """Single item in the replay buffer."""

    def __init__(
        self,
        trajectory_group: TrajectoryGroup,
        env_group_builder: EnvGroupBuilder,
        step: int,
        age: int,
    ):
        """
        Initialize a replay buffer item.

        Args:
            trajectory_group: Trajectory group from rollout
            env_group_builder: Environment builder used for rollout
            step: Current training step when this was generated
            age: Age of the data (oldest component in FSP matchup, or current step in self-play)
        """
        self.trajectory_group = trajectory_group
        self.env_group_builder = env_group_builder
        self.step = step
        self.age = age


class ReplayBuffer:
    """
    Thread-safe replay buffer for async actor-learner architecture.

    The buffer stores trajectories from actors and provides batches to the learner.
    It enforces staleness constraints by removing data older than max_staleness.
    """

    def __init__(
        self,
        max_staleness: int = 5,
        batch_size: int = 1,
    ):
        """
        Initialize the replay buffer.

        Args:
            max_staleness: Maximum age difference allowed (current_step - data_step)
            batch_size: Number of trajectory groups per batch
        """
        self.max_staleness = max_staleness
        self.batch_size = batch_size
        self.buffer: deque[ReplayBufferItem] = deque()
        self.lock = asyncio.Lock()
        self.data_available = asyncio.Condition(self.lock)
        self.current_learner_step = 0

        logger.info(
            f"Initialized ReplayBuffer with max_staleness={max_staleness}, "
            f"batch_size={batch_size}"
        )

    async def add(
        self,
        trajectory_group: TrajectoryGroup,
        env_group_builder: EnvGroupBuilder,
        step: int,
        age: int,
    ):
        """
        Add a trajectory group to the buffer.

        Args:
            trajectory_group: Trajectory group from rollout
            env_group_builder: Environment builder used for rollout
            step: Current training step when this was generated
            age: Age of the data (oldest component in FSP matchup)
        """
        async with self.data_available:
            item = ReplayBufferItem(trajectory_group, env_group_builder, step, age)
            self.buffer.append(item)
            logger.debug(
                f"Added item to buffer (step={step}, age={age}). "
                f"Buffer size: {len(self.buffer)}"
            )
            # Notify waiting learner that data is available
            self.data_available.notify()

    async def get_batch(
        self,
    ) -> Tuple[List[TrajectoryGroup], List[EnvGroupBuilder]]:
        """
        Get a batch of trajectory groups from the buffer.

        This blocks until enough data is available. Returns data in FIFO order.

        Returns:
            Tuple of (trajectory_groups, env_group_builders)
        """
        async with self.data_available:
            # Wait until we have enough data
            while len(self.buffer) < self.batch_size:
                logger.debug(
                    f"Waiting for data... ({len(self.buffer)}/{self.batch_size})"
                )
                await self.data_available.wait()

            # Pop batch_size items from front (FIFO)
            trajectory_groups = []
            env_group_builders = []

            for _ in range(self.batch_size):
                item = self.buffer.popleft()
                trajectory_groups.append(item.trajectory_group)
                env_group_builders.append(item.env_group_builder)

            logger.debug(
                f"Retrieved batch of {len(trajectory_groups)} items. "
                f"Remaining buffer size: {len(self.buffer)}"
            )

            return trajectory_groups, env_group_builders

    async def cleanup_stale_data(self, current_step: int):
        """
        Remove data that is too old based on staleness threshold.

        Age is calculated as (current_step - data_age), where data_age is the
        oldest component in the matchup for FSP, or the step for self-play.

        Args:
            current_step: Current learner training step
        """
        async with self.lock:
            self.current_learner_step = current_step
            initial_size = len(self.buffer)

            # Filter out stale data
            self.buffer = deque(
                [
                    item
                    for item in self.buffer
                    if (current_step - item.age) <= self.max_staleness
                ]
            )

            removed = initial_size - len(self.buffer)
            if removed > 0:
                logger.info(
                    f"Cleaned {removed} stale items from buffer (step={current_step}). "
                    f"Remaining: {len(self.buffer)}"
                )

    async def size(self) -> int:
        """Get current buffer size."""
        async with self.lock:
            return len(self.buffer)

    async def get_stats(self) -> dict:
        """Get buffer statistics for logging."""
        async with self.lock:
            if len(self.buffer) == 0:
                return {
                    "replay_buffer/size": 0,
                    "replay_buffer/oldest_age": 0,
                    "replay_buffer/newest_age": 0,
                    "replay_buffer/mean_age": 0,
                }

            ages = [item.age for item in self.buffer]
            steps = [item.step for item in self.buffer]

            return {
                "replay_buffer/size": len(self.buffer),
                "replay_buffer/oldest_age": min(ages),
                "replay_buffer/newest_age": max(ages),
                "replay_buffer/mean_age": sum(ages) / len(ages),
                "replay_buffer/oldest_step": min(steps),
                "replay_buffer/newest_step": max(steps),
            }
