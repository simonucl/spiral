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

"""SPIRAL training script using Tinker framework."""

import asyncio
import logging
import os
from datetime import datetime

import chz
from tinker_cookbook import cli_utils
from tinker_cookbook.rl import train

# Import SPIRAL custom train loop with draw retry
from spiral.tinker.dataset import SpiralRLDatasetBuilder
from spiral.tinker.eval.math_test import SpiralMathTestDatasetBuilder
from spiral.tinker.training import create_spiral_train_loop

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

@chz.chz
class SpiralConfig(train.Config):
    """Configuration for SPIRAL training."""

    # Model settings
    model_name: str = "Qwen/Qwen3-8B-Base"
    renderer_name: str = "qwen3"
    lora_rank: int = 32

    # Environment settings
    env_ids: list[str] = chz.field(default_factory=lambda: ["KuhnPoker-v1"]) # Format: "[env1, env2, ...]"
    use_llm_obs_wrappers: list[bool] = chz.field(default_factory=lambda: [True]) # Format: "[true, false, ...]"
    template_overrides: str = ""  # Format: "env1:template1,env2:template2"

    # Training settings
    batch_size: int = 128
    num_train_datapoints: int = 51200  # 400 steps * 128 batch
    num_test_datapoints: int = 128
    learning_rate: float = 1e-4
    max_tokens: int = 8192
    num_substeps: int = 1

    # SPIRAL-specific settings
    filter_draw: bool = False
    max_draw_retries: int = 5
    use_role_baseline: bool = True
    role_baseline_ema_gamma: float = 0.95
    use_intermediate_rewards: bool = True  # Whether to discount earlier turns
    gamma: float = 1.0  # Discount factor for turn-level rewards
    filter_zero_adv: bool = False  # Whether to filter turns with zero advantage

    # Evaluation settings
    eval_env_ids: str = ""  # Comma-separated, defaults to env_ids
    eval_use_llm_obs_wrappers: str = ""  # Comma-separated "true,false,true"
    eval_opponent_names: str = "random"  # Comma-separated
    eval_every: int = 16
    save_every: int = 20
    compute_post_kl: bool = False

    # Math test evaluation settings
    math_test_data_paths: str = ""  # Comma-separated paths (e.g., "data/aime,data/amc")
    enable_math_test_eval: bool = False  # Whether to enable math test evaluation

    # Loss function
    loss_fn: str = "importance_sampling"  # or "ppo"

    # Logging
    wandb_project: str | None = "spiral"
    wandb_name: str | None = None
    log_path: str = ""

    # Tinker service
    base_url: str | None = None

    # Advanced: streaming minibatch (optional)
    use_streaming: bool = False
    num_minibatches: int = 4

    # Fictitious self-play (FSP) settings
    fsp_enabled: bool = False  # Enable fictitious self-play
    fsp_start_from: int = 0  # Step to start FSP (before this, use standard self-play)
    fsp_update_interval: int = 10  # Add checkpoint to pool every N steps
    fsp_pool_size: int = 5  # Maximum number of historical checkpoints in pool
    fsp_include_current: bool = True  # Whether current model is available for sampling

    # Async actor-learner with replay buffer
    use_async_actor_learner: bool = False  # Enable async actor-learner architecture
    replay_buffer_max_staleness: int = 5  # Maximum staleness (steps) for replay buffer data

    # Resume settings
    load_checkpoint_path: str | None = None  # Path to training checkpoint (e.g., "tinker://.../weights/000180")
    fsp_resume_checkpoint_base: str | None = None  # Base path for FSP checkpoints (e.g., "tinker://.../sampler_weights/")


def parse_template_overrides(override_str: str) -> dict[str, str]:
    """Parse template overrides from string format 'env1:template1,env2:template2'."""
    if not override_str:
        return {}

    overrides = {}
    for pair in override_str.split(","):
        if ":" in pair:
            env, template = pair.split(":")
            overrides[env.strip()] = template.strip()
    return overrides


def parse_eval_env_ids(eval_env_str: str, default_env_ids: list[str]) -> list[str]:
    """Parse eval env IDs from comma-separated string."""
    if not eval_env_str:
        return default_env_ids
    return [env.strip() for env in eval_env_str.split(",")]


def parse_eval_llm_obs_wrappers(
    wrapper_str: str, default_wrappers: list[bool]
) -> list[bool]:
    """Parse eval LLM obs wrappers from comma-separated string of 'true'/'false'."""
    if not wrapper_str:
        return default_wrappers
    return [w.strip().lower() == "true" for w in wrapper_str.split(",")]


def parse_opponent_names(opponent_str: str) -> list[str]:
    """Parse opponent names from comma-separated string."""
    return [name.strip() for name in opponent_str.split(",")]


def build_config(cli_config: SpiralConfig) -> train.Config:

    # Parse template overrides
    template_overrides = parse_template_overrides(cli_config.template_overrides)

    # Parse evaluation settings
    eval_env_ids = parse_eval_env_ids(cli_config.eval_env_ids, cli_config.env_ids)
    eval_use_llm_obs_wrappers = parse_eval_llm_obs_wrappers(
        cli_config.eval_use_llm_obs_wrappers, cli_config.use_llm_obs_wrappers
    )
    eval_opponent_names = parse_opponent_names(cli_config.eval_opponent_names)

    # Parse math test data paths
    math_test_data_paths = []
    if cli_config.enable_math_test_eval and cli_config.math_test_data_paths:
        math_test_data_paths = [
            path.strip() for path in cli_config.math_test_data_paths.split(",")
        ]

    # Create run name
    date_and_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
    env_str = "+".join([env.replace("-v0", "").replace("-v1", "").replace("-v2", "") for env in cli_config.env_ids])
    run_name = (
        f"{cli_config.model_name.split('/')[-1]}-{env_str}-"
        f"{cli_config.batch_size}batch-{cli_config.learning_rate}lr-{date_and_time}"
    )

    # Set log path
    if cli_config.log_path:
        log_path = cli_config.log_path
    else:
        log_path = f"/tmp/spiral-tinker/{run_name}"

    # Set wandb name
    if cli_config.wandb_name is not None:
        wandb_name = cli_config.wandb_name
    else:
        wandb_name = run_name

    # Create dataset builder (always game-based for training)
    dataset_builder = SpiralRLDatasetBuilder(
            batch_size=cli_config.batch_size,
            num_train_datapoints=cli_config.num_train_datapoints,
            num_test_datapoints=cli_config.num_test_datapoints,
            model_name=cli_config.model_name,
            renderer_name=cli_config.renderer_name,
            env_ids=cli_config.env_ids,
            use_llm_obs_wrappers=cli_config.use_llm_obs_wrappers,
            template_overrides=template_overrides,
            filter_draw=cli_config.filter_draw,
            max_draw_retries=cli_config.max_draw_retries,
            use_role_baseline=cli_config.use_role_baseline,
            role_baseline_ema_gamma=cli_config.role_baseline_ema_gamma,
            use_intermediate_rewards=cli_config.use_intermediate_rewards,
            gamma=cli_config.gamma,
            eval_env_ids=eval_env_ids,
            eval_use_llm_obs_wrappers=eval_use_llm_obs_wrappers,
            eval_opponent_names=eval_opponent_names,
            base_url=cli_config.base_url,
        )

    # Create streaming config if enabled
    stream_minibatch_config = None
    if cli_config.use_streaming:
        stream_minibatch_config = train.StreamMinibatchConfig(
            groups_per_batch=cli_config.batch_size // 2,  # Each group has 2 envs
            num_minibatches=cli_config.num_minibatches,
        )
        logger.info(
            f"Enabled streaming minibatch with {cli_config.num_minibatches} minibatches"
        )

    # Create evaluator builders
    evaluator_builders = []
    if cli_config.eval_every > 0:
        # Add game evaluator
        evaluator_builders.append(lambda: dataset_builder.create_evaluator())

        # Add math test evaluators (one per dataset)
        if cli_config.enable_math_test_eval and math_test_data_paths:
            math_test_builder = SpiralMathTestDatasetBuilder(
                data_paths=math_test_data_paths,
                model_name_for_tokenizer=cli_config.model_name,
                renderer_name=cli_config.renderer_name,
                max_tokens=cli_config.max_tokens
            )
            # Create separate evaluators for each math dataset
            math_evaluators = math_test_builder.create_evaluators()
            evaluator_builders.extend([lambda e=evaluator: e for evaluator in math_evaluators])
            logger.info(f"Math test evaluation enabled with {len(math_evaluators)} datasets: {math_test_data_paths}")

    config = chz.replace(cli_config,
        dataset_builder=dataset_builder,
        evaluator_builders=evaluator_builders,
        stream_minibatch_config=stream_minibatch_config,
        wandb_name=wandb_name,
        log_path=log_path,
    )
    return config


def main():
    """Main entry point for SPIRAL training."""
    # Parse CLI config
    cli_config = chz.entrypoint(SpiralConfig)

    # Build training config
    config = build_config(cli_config)

    # Print configuration
    logger.info("=" * 80)
    logger.info("SPIRAL Training Configuration")
    logger.info("=" * 80)
    logger.info(f"Model: {config.model_name}")
    logger.info(f"Environments: {cli_config.env_ids}")
    logger.info(f"Type of env_ids: {type(cli_config.env_ids)}")
    logger.info(f"Length of env_ids: {len(cli_config.env_ids)}")
    logger.info(f"Batch size: {cli_config.batch_size}")
    logger.info(f"Learning rate: {cli_config.learning_rate}")
    logger.info(f"Max tokens: {cli_config.max_tokens}")
    logger.info(f"Draw filtering: {cli_config.filter_draw} (max retries: {cli_config.max_draw_retries})")
    logger.info(f"Role baseline (RAE): {cli_config.use_role_baseline}")
    logger.info(f"Loss function: {cli_config.loss_fn}")
    logger.info(f"Streaming minibatch: {cli_config.use_streaming}")
    logger.info(f"Log path: {config.log_path}")
    logger.info("=" * 80)

    # Check log dir
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="ask")

    # Run training with custom SPIRAL loop (includes draw retry)
    asyncio.run(create_spiral_train_loop(config))


if __name__ == "__main__":
    main()
