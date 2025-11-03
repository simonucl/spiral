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

"""Actor loop for async actor-learner architecture."""

import asyncio
import logging
from typing import Optional

import tinker
from tinker_cookbook.rl.types import RLDataset

from spiral.tinker.training.async_actor_learner.replay_buffer import ReplayBuffer
from spiral.tinker.training.population import PopulationManager
from spiral.tinker.training.rollouts import do_group_rollout_and_filter_constant_reward

logger = logging.getLogger(__name__)


class ActorLoop:
    """
    Actor loop that continuously generates rollouts and adds them to replay buffer.

    The actor runs independently of the learner, using the latest policy available.
    In FSP mode, it samples matchups from the population manager.
    """

    def __init__(
        self,
        cfg,
        service_client: tinker.ServiceClient,
        dataset: RLDataset,
        replay_buffer: ReplayBuffer,
        population_manager: Optional[PopulationManager] = None,
    ):
        """
        Initialize the actor loop.

        Args:
            cfg: Training configuration
            service_client: Tinker service client for creating sampling clients
            dataset: RL dataset for environment builders
            replay_buffer: Replay buffer to add trajectories to
            population_manager: Optional population manager for FSP
        """
        self.cfg = cfg
        self.service_client = service_client
        self.dataset = dataset
        self.replay_buffer = replay_buffer
        self.population_manager = population_manager

        # Latest policy info
        self.current_step = 0
        self.current_sampling_client: Optional[tinker.SamplingClient] = None
        self.update_lock = asyncio.Lock()

        # Shutdown flag
        self.should_stop = False

        logger.info(
            f"Initialized ActorLoop (FSP={'enabled' if population_manager else 'disabled'})"
        )

    async def update_policy(self, step: int, sampling_client: tinker.SamplingClient):
        """
        Update the actor's policy.

        Called by the learner when a new policy is available.

        Args:
            step: Training step for this policy
            sampling_client: New sampling client to use
        """
        async with self.update_lock:
            self.current_step = step
            self.current_sampling_client = sampling_client

            # Update population manager if using FSP
            if self.population_manager is not None:
                self.population_manager.add_checkpoint(
                    step=step, sampling_client=sampling_client, is_current=True
                )

                # Add to historical pool at regular intervals
                if (
                    step >= self.cfg.fsp_start_from
                    and step % self.cfg.fsp_update_interval == 0
                ):
                    self.population_manager.add_checkpoint(
                        step=step, sampling_client=sampling_client, is_current=False
                    )

            logger.info(f"[ACTOR] Updated policy to step {step}")

    async def run(self):
        """
        Main actor loop.

        Continuously generates rollouts and adds them to the replay buffer.
        Runs until should_stop is set to True.
        """
        logger.info("[ACTOR] Starting actor loop")
        rollout_counter = 0

        while not self.should_stop:
            # Wait for initial policy
            async with self.update_lock:
                if self.current_sampling_client is None:
                    logger.debug("[ACTOR] Waiting for initial policy...")
                    await asyncio.sleep(0.1)
                    continue

                current_step = self.current_step
                current_client = self.current_sampling_client

            # Get batch of environments
            # Cycle through dataset in a round-robin fashion
            batch_idx = rollout_counter % len(self.dataset)
            env_group_builders = self.dataset.get_batch(batch_idx)

            # Generate rollouts for this batch
            for env_builder in env_group_builders:
                # Determine which policies to use
                if (
                    self.population_manager is not None
                    and current_step >= self.cfg.fsp_start_from
                ):
                    # FSP mode: sample matchup
                    (
                        step1,
                        step2,
                        client1,
                        client2,
                    ) = self.population_manager.sample_matchup()
                    age = min(step1, step2)  # Oldest component
                    opponent_client = client2
                    logger.debug(
                        f"[ACTOR] FSP matchup: step {step1} vs {step2} (age={age})"
                    )
                else:
                    # Self-play mode
                    age = current_step
                    opponent_client = None
                    logger.debug(f"[ACTOR] Self-play rollout (step={current_step})")

                # Do rollout
                trajectory_group = await do_group_rollout_and_filter_constant_reward(
                    sampling_client=current_client,
                    env_group_builder=env_builder,
                    max_tokens=self.cfg.max_tokens,
                    do_remove_constant_reward_groups=self.cfg.remove_constant_reward_groups,
                    opponent_sampling_client=opponent_client,
                )

                # Add to replay buffer if not filtered
                if trajectory_group is not None:
                    await self.replay_buffer.add(
                        trajectory_group=trajectory_group,
                        env_group_builder=env_builder,
                        step=current_step,
                        age=age,
                    )
                    logger.debug(
                        f"[ACTOR] Added rollout to buffer (step={current_step}, age={age})"
                    )

            rollout_counter += 1

        logger.info("[ACTOR] Actor loop stopped")

    def stop(self):
        """Signal the actor loop to stop."""
        logger.info("[ACTOR] Stopping actor loop...")
        self.should_stop = True
