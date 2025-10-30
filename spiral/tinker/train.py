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

"""Custom training loop for SPIRAL with draw retry."""

import asyncio
import logging
import time

import tinker
from tinker_cookbook import checkpoint_utils
from tinker_cookbook.completers import TinkerTokenCompleter
from tinker_cookbook.rl import train
from tinker_cookbook.rl.types import (EnvGroupBuilder, RLDataset,
                                      TrajectoryGroup)
from tinker_cookbook.tokenizer_utils import Tokenizer
from tinker_cookbook.utils import ml_log
from tinker_cookbook.utils.misc_utils import timed
from tinker_cookbook.utils.trace import scope

from spiral.tinker.utils import convert_to_json_serializable
from tqdm.asyncio import tqdm

from spiral.tinker.env import SpiralTwoPlayerEnvGroupBuilder
from spiral.tinker.rollouts import do_group_rollout_with_draw_retry
from spiral.tinker.train_step import train_step as spiral_train_step

logger = logging.getLogger(__name__)


@scope
async def do_group_rollout_and_filter_constant_reward(
    sampling_client: tinker.SamplingClient,
    env_group_builder: EnvGroupBuilder,
    max_tokens: int,
    do_remove_constant_reward_groups: bool,
) -> TrajectoryGroup | None:
    """
    Do a group rollout with draw retry (if builder supports it) and optionally filter constant rewards.

    Args:
        sampling_client: Sampling client for policy
        env_group_builder: Environment group builder
        max_tokens: Max tokens for generation
        do_remove_constant_reward_groups: Whether to filter constant reward groups

    Returns:
        TrajectoryGroup or None if filtered
    """
    policy = TinkerTokenCompleter(sampling_client, max_tokens=max_tokens)

    # Use custom rollout with draw retry if this is a SPIRAL builder
    if isinstance(env_group_builder, SpiralTwoPlayerEnvGroupBuilder):
        trajectory_group = await do_group_rollout_with_draw_retry(
            env_group_builder, policy
        )
    else:
        # Standard rollout for non-SPIRAL builders
        from tinker_cookbook.rl.rollouts import do_group_rollout

        trajectory_group = await do_group_rollout(env_group_builder, policy)

    # Remove if all trajectories have the same reward
    trajectory_groups = [trajectory_group]
    # if do_remove_constant_reward_groups:
    #     trajectory_groups = remove_constant_reward_groups(trajectory_groups)
    if len(trajectory_groups) == 0:
        return None
    return trajectory_groups[0]


@scope
async def do_sync_training_spiral(
    start_batch: int,
    end_batch: int,
    num_batches: int,
    cfg: train.Config,
    training_client: tinker.TrainingClient,
    service_client: tinker.ServiceClient,
    evaluators: list,
    dataset: RLDataset,
    ml_logger: ml_log.Logger,
    tokenizer: Tokenizer,
):
    """
    Implements fully synchronous on-policy training with SPIRAL's draw retry.

    This is adapted from tinker_cookbook.rl.train.do_sync_training but uses
    our custom rollout function that includes draw retry logic.

    Args:
        start_batch: Starting batch index
        end_batch: Ending batch index
        num_batches: Total number of batches
        cfg: Training configuration
        training_client: Tinker training client
        service_client: Tinker service client
        evaluators: List of evaluators
        dataset: RL dataset
        ml_logger: Logger for metrics
        tokenizer: Tokenizer
    """
    for i_batch in range(start_batch, end_batch):

        if (i_batch + 1) % cfg.save_every == 0:
            sampling_client, _ = await train.save_checkpoint_and_get_sampling_client(
                training_client, i_batch + 1, cfg.log_path, cfg.save_every
            )
        else:
            sampling_path = (
                training_client.save_weights_for_sampler(name=f"{i_batch + 1:06d}")
                .result()
                .path
            )
            sampling_client = service_client.create_sampling_client(
                model_path=sampling_path
            )

        metrics = {
            "progress/batch": i_batch,
            "optim/lr": cfg.learning_rate,
            "progress/done_frac": (i_batch + 1) / num_batches,
        }
        t_start = time.time()

        # Run evaluations
        if cfg.eval_every > 0 and i_batch % cfg.eval_every == 0:
            with timed("run_evals", metrics):
                for evaluator in evaluators:
                    eval_metrics = await evaluator(
                        sampling_client
                    )
                    metrics.update(eval_metrics)

        # Get batch and sample trajectories
        env_group_builders_P = dataset.get_batch(i_batch)
        with timed("sample", metrics):
            trajectory_groups_P: list[TrajectoryGroup] = await tqdm.gather(
                *[
                    asyncio.create_task(
                        do_group_rollout_and_filter_constant_reward(
                            sampling_client,
                            builder,
                            max_tokens=cfg.max_tokens,
                            do_remove_constant_reward_groups=cfg.remove_constant_reward_groups,
                        ),
                        name=f"sample_task_{i}",
                    )
                    for i, builder in enumerate(env_group_builders_P)
                ],
                desc=f"Batch {i_batch} rollouts",
            )

        logger.info(
            f"Training step {i_batch} with {len(trajectory_groups_P)} trajectory groups"
        )

        # Train step - use custom SPIRAL training step for RAE
        train_step_metrics = await spiral_train_step(
            cfg,
            i_batch,
            training_client,
            tokenizer,
            env_group_builders_P,
            trajectory_groups_P,
        )

        # Log metrics
        metrics.update(train_step_metrics)
        metrics["time/total"] = time.time() - t_start
        metrics = convert_to_json_serializable(metrics)
        ml_logger.log_metrics(metrics, step=i_batch)


async def create_spiral_train_loop(cfg: train.Config):
    """
    Main training loop for SPIRAL using Tinker.

    This is adapted from tinker_cookbook.rl.train.main but uses our custom
    sync training function that includes draw retry.

    Args:
        cfg: Training configuration
    """
    ml_logger = ml_log.setup_logging(
        log_dir=cfg.log_path,
        wandb_project=cfg.wandb_project,
        config=cfg,
        wandb_name=cfg.wandb_name,
    )

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("pylatexenc").setLevel(logging.WARNING)

    resume_info = checkpoint_utils.get_last_checkpoint(cfg.log_path)
    if resume_info:
        start_batch = resume_info["batch"]
    else:
        start_batch = 0

    service_client = tinker.ServiceClient(base_url=cfg.base_url)
    training_client = await service_client.create_lora_training_client_async(
        cfg.model_name, rank=cfg.lora_rank
    )

    # Resume from checkpoint if available
    load_state_path: str | None = (
        resume_info["state_path"] if resume_info else cfg.load_checkpoint_path
    )
    if load_state_path:
        future = await training_client.load_state_async(load_state_path)
        _ = await future.result_async()
        logger.info(f"Loaded state from {load_state_path}")

    # Get tokenizer from training client
    tokenizer = training_client.get_tokenizer()

    # Create dataset from thunk
    dataset, maybe_test_dataset = await cfg.dataset_builder()
    # evaluators = [evaluator() for evaluator in cfg.evaluator_builders]
    evaluators = []
    if maybe_test_dataset is not None:
        from tinker_cookbook.rl.metric_util import RLTestSetEvaluator

        evaluators.append(
            RLTestSetEvaluator(maybe_test_dataset, max_tokens=cfg.max_tokens)
        )

    num_batches = len(dataset)
    logger.info(f"Will train on {num_batches} batches")

    # Use our custom sync training with draw retry
    # Note: Async and streaming not yet supported with draw retry
    if cfg.async_config is not None:
        raise NotImplementedError(
            "Async training not yet supported with SPIRAL draw retry"
        )
    elif cfg.stream_minibatch_config is not None:
        raise NotImplementedError(
            "Streaming minibatch not yet supported with SPIRAL draw retry"
        )
    else:
        await do_sync_training_spiral(
            start_batch=start_batch,
            end_batch=num_batches,
            num_batches=num_batches,
            cfg=cfg,
            training_client=training_client,
            service_client=service_client,
            evaluators=evaluators,
            dataset=dataset,
            ml_logger=ml_logger,
            tokenizer=tokenizer,
        )

    # Save final checkpoint
    if start_batch < num_batches:
        _ = await checkpoint_utils.save_checkpoint_async(
            training_client=training_client,
            name="final",
            log_path=cfg.log_path,
            kind="both",
            loop_state={"batch": num_batches},
        )
    else:
        logger.info("Training was already complete; nothing to do")

    # Cleanup
    ml_logger.close()
    logger.info("Training completed successfully")
