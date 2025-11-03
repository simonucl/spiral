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

"""Learner loop for async actor-learner architecture."""

import asyncio
import logging
import time
from typing import Optional

import tinker
from tinker_cookbook import checkpoint_utils
from tinker_cookbook.rl import train
from tinker_cookbook.tokenizer_utils import Tokenizer
from tinker_cookbook.utils import ml_log
from tinker_cookbook.utils.misc_utils import timed

from spiral.tinker.eval.evaluator import AsyncEvalRunner
from spiral.tinker.training.async_actor_learner.actor import ActorLoop
from spiral.tinker.training.async_actor_learner.replay_buffer import ReplayBuffer
from spiral.tinker.training.train_step import train_step as spiral_train_step
from spiral.tinker.utils import convert_to_json_serializable

logger = logging.getLogger(__name__)


class LearnerLoop:
    """
    Learner loop that trains on batches from the replay buffer.

    The learner runs independently of the actor, pulling batches when available
    and updating the policy.
    """

    def __init__(
        self,
        cfg,
        training_client: tinker.TrainingClient,
        service_client: tinker.ServiceClient,
        tokenizer: Tokenizer,
        replay_buffer: ReplayBuffer,
        actor_loop: ActorLoop,
        ml_logger: ml_log.Logger,
        evaluators: list,
        eval_runner: Optional[AsyncEvalRunner] = None,
        start_batch: int = 0,
        num_batches: int = 1000,
    ):
        """
        Initialize the learner loop.

        Args:
            cfg: Training configuration
            training_client: Tinker training client
            service_client: Tinker service client
            tokenizer: Tokenizer
            replay_buffer: Replay buffer to pull batches from
            actor_loop: Actor loop to update with new policies
            ml_logger: Logger for metrics
            evaluators: List of evaluators
            eval_runner: Optional async eval runner
            start_batch: Starting batch number
            num_batches: Total number of batches to train
        """
        self.cfg = cfg
        self.training_client = training_client
        self.service_client = service_client
        self.tokenizer = tokenizer
        self.replay_buffer = replay_buffer
        self.actor_loop = actor_loop
        self.ml_logger = ml_logger
        self.evaluators = evaluators
        self.eval_runner = eval_runner
        self.start_batch = start_batch
        self.num_batches = num_batches

        # Current training step
        self.current_batch = start_batch

        # Shutdown flag
        self.should_stop = False

        logger.info(
            f"Initialized LearnerLoop (start_batch={start_batch}, num_batches={num_batches})"
        )

    async def run(self):
        """
        Main learner loop.

        Continuously pulls batches from replay buffer, trains, and updates policy.
        Runs until num_batches is reached or should_stop is set.
        """
        logger.info(
            f"[LEARNER] Starting learner loop (batches {self.start_batch}-{self.num_batches})"
        )

        for i_batch in range(self.start_batch, self.num_batches):
            if self.should_stop:
                logger.info(f"[LEARNER] Stopping at batch {i_batch}")
                break

            self.current_batch = i_batch
            metrics = {
                "progress/batch": i_batch,
                "optim/lr": self.cfg.learning_rate,
                "progress/done_frac": (i_batch + 1) / self.num_batches,
            }
            t_start = time.time()

            # Create new sampling client for this step
            if (i_batch + 1) % self.cfg.save_every == 0:
                sampling_client, _ = await train.save_checkpoint_and_get_sampling_client(
                    self.training_client, i_batch + 1, self.cfg.log_path, self.cfg.save_every
                )
            else:
                sampling_path = (
                    self.training_client.save_weights_for_sampler(name=f"{i_batch + 1:06d}")
                    .result()
                    .path
                )
                sampling_client = self.service_client.create_sampling_client(
                    model_path=sampling_path
                )

            # Update actor with new policy (non-blocking)
            asyncio.create_task(
                self.actor_loop.update_policy(i_batch, sampling_client)
            )

            # Schedule evaluations
            if self.cfg.eval_every > 0 and i_batch % self.cfg.eval_every == 0 and self.eval_runner:
                self.eval_runner.schedule_eval(self.evaluators, sampling_client, i_batch)

            # Get batch from replay buffer (blocks until available)
            with timed("wait_for_batch", metrics):
                logger.info(
                    f"[LEARNER] Waiting for batch {i_batch} from replay buffer..."
                )
                (
                    trajectory_groups,
                    env_group_builders,
                ) = await self.replay_buffer.get_batch()
                logger.info(
                    f"[LEARNER] Got batch {i_batch} with {len(trajectory_groups)} trajectory groups"
                )

            # Train on the batch
            train_step_metrics = await spiral_train_step(
                self.cfg,
                i_batch,
                self.training_client,
                self.tokenizer,
                env_group_builders,
                trajectory_groups,
                sampling_client,
            )

            # Clean up stale data from replay buffer
            with timed("cleanup_stale", metrics):
                await self.replay_buffer.cleanup_stale_data(i_batch)

            # Get replay buffer stats
            buffer_stats = await self.replay_buffer.get_stats()
            metrics.update(buffer_stats)

            # Log metrics
            metrics.update(train_step_metrics)
            metrics["time/total"] = time.time() - t_start
            metrics = convert_to_json_serializable(metrics)
            self.ml_logger.log_metrics(metrics, step=i_batch)

            logger.info(
                f"[LEARNER] Completed batch {i_batch} "
                f"(time={metrics['time/total']:.2f}s, buffer_size={buffer_stats['replay_buffer/size']})"
            )

        # Wait for all background evaluations to complete
        if self.eval_runner:
            await self.eval_runner.wait_all()

        # Save final checkpoint
        if self.start_batch < self.num_batches:
            _ = await checkpoint_utils.save_checkpoint_async(
                training_client=self.training_client,
                name="final",
                log_path=self.cfg.log_path,
                kind="both",
                loop_state={"batch": self.num_batches},
            )
        else:
            logger.info("[LEARNER] Training was already complete; nothing to do")

        logger.info("[LEARNER] Learner loop completed")

    def stop(self):
        """Signal the learner loop to stop."""
        logger.info("[LEARNER] Stopping learner loop...")
        self.should_stop = True
