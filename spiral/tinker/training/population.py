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

"""Population management for fictitious self-play (FSP)."""

import logging
import random
from typing import List, Tuple, Optional

import tinker

logger = logging.getLogger(__name__)


class PopulationManager:
    """
    Manages a population of model checkpoints for fictitious self-play.

    The population is a pool of sampling clients representing different training
    checkpoints. During training, agents are randomly sampled from this pool to
    play against each other, providing diverse opponents and improving robustness.
    """

    def __init__(
        self,
        pool_size: int,
        include_current: bool = True,
    ):
        """
        Initialize the population manager.

        Args:
            pool_size: Maximum number of checkpoints to keep in the pool (excluding current)
            include_current: Whether the current model should always be available for sampling
        """
        self.pool_size = pool_size
        self.include_current = include_current
        self.population: List[Tuple[int, tinker.SamplingClient]] = []  # (step, client)
        self.current_step = 0
        self.current_client: tinker.SamplingClient | None = None

        logger.info(
            f"Initialized PopulationManager with pool_size={pool_size}, "
            f"include_current={include_current}"
        )

    def add_checkpoint(
        self, step: int, sampling_client: tinker.SamplingClient, is_current: bool = False
    ):
        """
        Add a new checkpoint to the population.

        If the pool is full, the oldest checkpoint is evicted (FIFO).

        Args:
            step: Training step number for this checkpoint
            sampling_client: Sampling client for this checkpoint
            is_current: Whether this is the current (latest) model
        """
        if is_current:
            # Update current model reference
            self.current_step = step
            self.current_client = sampling_client
            logger.info(f"[FSP] Updated current model to step {step}")
        else:
            # Add to historical population
            self.population.append((step, sampling_client))
            logger.info(f"[FSP] Added checkpoint from step {step} to population")

            # Evict oldest if pool is full
            if len(self.population) > self.pool_size:
                evicted_step, _ = self.population.pop(0)  # FIFO: remove oldest
                logger.info(
                    f"[FSP] Population full ({len(self.population)}/{self.pool_size}), "
                    f"evicted checkpoint from step {evicted_step}"
                )

    def sample_matchup(self) -> Tuple[int, int, tinker.SamplingClient, tinker.SamplingClient]:
        """
        Sample a matchup where the current policy plays against a randomly sampled opponent.

        In FSP, one player is always the current (latest) policy, and the opponent
        is randomly sampled from the pool (which can include the current policy for self-play).

        Returns:
            Tuple of (current_step, opponent_step, current_client, opponent_client) where:
            - current_step: Current model's training step
            - opponent_step: Opponent's training step (randomly sampled from pool)
            - current_client: Current model's sampling client
            - opponent_client: Opponent's sampling client (randomly sampled from pool)
        """
        if self.current_client is None:
            raise ValueError("No current model available for FSP")

        # Build opponent pool from historical checkpoints
        opponent_pool = list(self.population)

        # Add current model to opponent pool if enabled (for self-play)
        if self.include_current:
            opponent_pool.append((self.current_step, self.current_client))

        if len(opponent_pool) == 0:
            # No opponents available, fall back to self-play
            logger.warning(
                f"[FSP] No opponents in pool, using self-play"
            )
            return (
                self.current_step,
                self.current_step,
                self.current_client,
                self.current_client,
            )

        # Current policy is always player 0
        step1 = self.current_step
        client1 = self.current_client

        # Randomly sample opponent from pool (can be current model if include_current=True)
        opponent_idx = random.randint(0, len(opponent_pool) - 1)
        step2, client2 = opponent_pool[opponent_idx]

        return step1, step2, client1, client2

    def get_population_stats(self) -> dict:
        """
        Get statistics about the current population.

        Returns:
            Dictionary with population statistics for logging
        """
        if not self.population:
            return {
                "fsp/pool_size": 0,
                "fsp/oldest_agent_step": 0,
                "fsp/newest_agent_step": self.current_step,
                "fsp/avg_agent_age": 0,
            }

        steps = [s for s, _ in self.population]
        ages = [self.current_step - s for s in steps]

        stats = {
            "fsp/pool_size": len(self.population),
            "fsp/oldest_agent_step": min(steps),
            "fsp/newest_agent_step": max(steps),
            "fsp/avg_agent_age": sum(ages) / len(ages) if ages else 0,
        }

        if self.include_current and self.current_client is not None:
            stats["fsp/total_agents"] = len(self.population) + 1
            stats["fsp/current_step"] = self.current_step

        return stats

    def __len__(self) -> int:
        """Return total number of agents available for sampling."""
        count = len(self.population)
        if self.include_current and self.current_client is not None:
            count += 1
        return count

    async def restore_from_checkpoint_base_path(
        self,
        service_client: tinker.ServiceClient,
        checkpoint_base_path: str,
        fsp_start_from: int,
        fsp_update_interval: int,
        resume_step: int,
    ):
        """
        Restore population from checkpoint base path when checkpoints.jsonl is unavailable.

        This method constructs checkpoint paths based on the base path pattern and loads
        the appropriate checkpoints into the FSP pool.

        Args:
            service_client: Tinker service client for creating sampling clients
            checkpoint_base_path: Base path template (e.g., "tinker://xxx/sampler_weights/")
            fsp_start_from: Step when FSP was enabled
            fsp_update_interval: Interval at which checkpoints were added to pool
            resume_step: Current training step being resumed from
        """
        logger.info(
            f"[FSP] Restoring population from base path: {checkpoint_base_path}"
        )

        # Determine which steps should be in the pool
        checkpoint_steps = []
        for step in range(fsp_start_from, resume_step, fsp_update_interval):
            if step % fsp_update_interval == 0:
                checkpoint_steps.append(step)

        # Keep only the most recent pool_size checkpoints
        checkpoint_steps = checkpoint_steps[-self.pool_size :]

        logger.info(
            f"[FSP] Loading {len(checkpoint_steps)} checkpoints: {checkpoint_steps}"
        )

        # Load sampling clients for each checkpoint
        for step in checkpoint_steps:
            # Construct checkpoint path (assumes format: base_path + "000180")
            checkpoint_path = f"{checkpoint_base_path}{step:06d}"

            try:
                sampling_client = service_client.create_sampling_client(
                    model_path=checkpoint_path
                )
                self.population.append((step, sampling_client))
                logger.info(
                    f"[FSP] Restored checkpoint from step {step}: {checkpoint_path}"
                )
            except Exception as e:
                logger.error(
                    f"[FSP] Failed to load checkpoint from step {step} at {checkpoint_path}: {e}"
                )

        logger.info(
            f"[FSP] Population restoration complete. Pool size: {len(self.population)}/{self.pool_size}"
        )

        # Log restored population stats
        if self.population:
            steps = [s for s, _ in self.population]
            logger.info(
                f"[FSP] Population steps: {sorted(steps)}, "
                f"oldest={min(steps)}, newest={max(steps)}"
            )
