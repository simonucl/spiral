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

"""Dataset builders for SPIRAL multi-environment training."""

import logging
import random
from typing import Sequence

import chz
import tinker
from tinker_cookbook.completers import TinkerMessageCompleter
from tinker_cookbook.rl.types import (EnvGroupBuilder, RLDataset,
                                      RLDatasetBuilder)

from spiral.agents.random import RandomAgent
from spiral.tinker.renderer import get_spiral_renderer
from spiral.tinker.training.env import SpiralTwoPlayerEnvGroupBuilder

logger = logging.getLogger(__name__)


class SpiralRLDataset(RLDataset):
    """
    Dataset for SPIRAL that supports multiple environments and rotation.
    """

    def __init__(
        self,
        env_group_builders: list[SpiralTwoPlayerEnvGroupBuilder],
        batch_size: int,
        num_datapoints: int,
        shuffle_envs: bool = True,
    ):
        """
        Initialize SPIRAL dataset.

        Args:
            env_group_builders: List of environment group builders (one per env config)
            batch_size: Batch size (must be even, since each game has 2 players)
            num_datapoints: Total number of datapoints for this dataset
            shuffle_envs: Whether to shuffle environments to avoid order bias
        """
        self.env_group_builders = env_group_builders
        self.batch_size = batch_size
        self.num_datapoints = num_datapoints
        self.shuffle_envs = shuffle_envs

        # Validate batch size
        if batch_size % 2 != 0:
            raise ValueError(f"Batch size must be even (got {batch_size})")

        # Shuffle builders to avoid order bias
        if self.shuffle_envs and len(self.env_group_builders) > 1:
            random.shuffle(self.env_group_builders)
            logger.info(f"Shuffled {len(self.env_group_builders)} environment builders")

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        """
        Get a batch of environment group builders.

        Rotates through the available environment builders cyclically.

        Args:
            index: Batch index

        Returns:
            List of EnvGroupBuilders for this batch
        """
        # Number of groups needed (each group creates 2 envs)
        num_groups = self.batch_size // 2

        # Calculate starting datapoint index
        start_datapoint = index * self.batch_size

        # Build batch by rotating through environment builders
        batch_builders = []
        for i in range(num_groups):
            # Calculate which datapoint this is
            datapoint_idx = start_datapoint + (i * 2)

            # Skip if beyond dataset size
            if datapoint_idx >= self.num_datapoints:
                break

            # Select environment builder cyclically
            builder_idx = (datapoint_idx // 2) % len(self.env_group_builders)
            batch_builders.append(self.env_group_builders[builder_idx])

        return batch_builders

    def __len__(self) -> int:
        """Return number of batches."""
        return self.num_datapoints // self.batch_size


@chz.chz
class SpiralRLDatasetBuilder(RLDatasetBuilder):
    """
    Builder for SPIRAL datasets with multi-environment support.
    """

    # Basic settings
    batch_size: int
    num_train_datapoints: int
    num_test_datapoints: int
    model_name: str
    renderer_name: str

    # Environment settings
    env_ids: list[str]
    use_llm_obs_wrappers: list[bool]
    template_overrides: dict[str, str] = chz.field(default_factory=dict)

    # SPIRAL-specific settings
    filter_draw: bool = True
    max_draw_retries: int = 5
    use_role_baseline: bool = True
    role_baseline_ema_gamma: float = 0.95
    use_intermediate_rewards: bool = (
        True  # Whether to distribute final reward to all turns
    )
    gamma: float = 1.0  # Discount factor for intermediate rewards

    # Evaluation settings
    # Use munger to default to training envs if not specified
    eval_env_ids: list[str] | None = chz.field(
        default=None, munger=lambda self, val: val if val is not None else self.env_ids
    )
    eval_use_llm_obs_wrappers: list[bool] | None = chz.field(
        default=None,
        munger=lambda self, val: val if val is not None else self.use_llm_obs_wrappers,
    )
    eval_opponent_names: list[str] = chz.field(
        default_factory=lambda: ["random", "google/gemini-2.0-flash-exp"]
    )
    eval_games_per_matchup: int = 16
    eval_prompt_template: str = "qwen3"

    # Tinker settings
    base_url: str | None = None

    @chz.validate
    def _validate_env_wrapper_length(self):
        """Validate that env_ids and use_llm_obs_wrappers have same length."""
        if len(self.env_ids) != len(self.use_llm_obs_wrappers):
            raise ValueError(
                f"Length mismatch: {len(self.env_ids)} env_ids but "
                f"{len(self.use_llm_obs_wrappers)} use_llm_obs_wrappers"
            )

    def _create_env_group_builder(
        self,
        env_id: str,
        use_llm_obs_wrapper: bool,
        self_play: bool,
        opponent_policy: any = None,
    ) -> SpiralTwoPlayerEnvGroupBuilder:
        """
        Create an environment group builder for a specific environment.

        Args:
            env_id: Environment ID
            use_llm_obs_wrapper: Whether to use LLM observation wrapper
            self_play: Whether this is self-play or vs opponent
            opponent_policy: Optional opponent policy (for testing)

        Returns:
            Configured SpiralTwoPlayerEnvGroupBuilder
        """
        # Get template name (with override if specified)
        template_name = self.template_overrides.get(env_id, self.renderer_name)

        # Create renderer for this environment
        # Note: env_id is not passed to renderer - validation happens in Environment
        renderer = get_spiral_renderer(self.model_name, template_name)

        # Create builder
        builder = SpiralTwoPlayerEnvGroupBuilder(
            env_id=env_id,
            renderer=renderer,
            num_envs=2,
            self_play=self_play,
            opponent_policy=opponent_policy,
            filter_draw=self.filter_draw,
            max_draw_retries=self.max_draw_retries,
            use_role_baseline=self.use_role_baseline,
            role_baseline_ema_gamma=self.role_baseline_ema_gamma,
            use_intermediate_rewards=self.use_intermediate_rewards,
            gamma=self.gamma,
            use_llm_obs_wrapper=use_llm_obs_wrapper,
        )
        return builder

    def create_evaluator(self):
        """Create a GameEvaluator for online evaluation."""
        from spiral.tinker.evaluator import GameEvaluator

        return GameEvaluator(
            eval_env_ids=self.eval_env_ids,
            eval_opponent_names=self.eval_opponent_names,
            model_name=self.model_name,
            eval_use_llm_obs_wrappers=self.eval_use_llm_obs_wrappers,
            eval_games_per_matchup=self.eval_games_per_matchup,
            prompt_template=self.eval_prompt_template,
            model_player_id=0,  # Model always plays as player 0 during eval
        )

    def _create_opponent_policy(self, opponent_name: str, env_id: str):
        """
        Create opponent policy for evaluation.

        Args:
            opponent_name: Name of opponent ("random" or model name for LLM)
            env_id: Environment ID

        Returns:
            Opponent policy (RandomAgent or TinkerMessageCompleter)
        """
        if opponent_name == "random":
            return RandomAgent(env_id)
        else:
            # LLM opponent via TinkerMessageCompleter
            # Get template and renderer
            template_name = self.template_overrides.get(env_id, self.renderer_name)
            renderer = get_spiral_renderer(self.model_name, template_name)

            # Create sampling client
            service_client = tinker.ServiceClient(base_url=self.base_url)
            sampling_client = service_client.create_sampling_client(
                base_model=opponent_name
            )

            # Create message completer
            return TinkerMessageCompleter(
                sampling_client=sampling_client,
                renderer=renderer,
                max_tokens=4096,
                stop_condition=renderer.get_stop_sequences(),
            )

    async def __call__(self) -> tuple[SpiralRLDataset, SpiralRLDataset | None]:
        """
        Build the dataset for training and testing.

        Returns:
            Tuple of (train_dataset, test_dataset)
        """
        # Create training dataset (self-play)
        train_builders = []
        for env_id, use_llm_obs_wrapper in zip(self.env_ids, self.use_llm_obs_wrappers):
            builder = self._create_env_group_builder(
                env_id=env_id,
                use_llm_obs_wrapper=use_llm_obs_wrapper,
                self_play=True,
                opponent_policy=None,
            )
            train_builders.append(builder)

        train_dataset = SpiralRLDataset(
            env_group_builders=train_builders,
            batch_size=self.batch_size,
            num_datapoints=self.num_train_datapoints,
            shuffle_envs=True,
        )

        logger.info(
            f"Created train dataset with {len(train_builders)} environments: "
            f"{[b.env_id for b in train_builders]}"
        )

        # Create test dataset (vs opponents) if requested
        test_dataset = None
        if self.num_test_datapoints > 0:
            test_builders = []
            for env_id, use_llm_obs_wrapper in zip(
                self.eval_env_ids, self.eval_use_llm_obs_wrappers
            ):
                for opponent_name in self.eval_opponent_names:
                    opponent_policy = self._create_opponent_policy(
                        opponent_name, env_id
                    )
                    builder = self._create_env_group_builder(
                        env_id=env_id,
                        use_llm_obs_wrapper=use_llm_obs_wrapper,
                        self_play=False,
                        opponent_policy=opponent_policy,
                    )
                    test_builders.append(builder)

            test_dataset = SpiralRLDataset(
                env_group_builders=test_builders,
                batch_size=self.num_test_datapoints,
                num_datapoints=self.num_test_datapoints,
                shuffle_envs=False,  # Don't shuffle test set
            )

            logger.info(
                f"Created test dataset with {len(test_builders)} environment-opponent pairs"
            )

        return train_dataset, test_dataset
