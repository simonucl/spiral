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

"""Math test evaluation for SPIRAL Tinker implementation."""

import logging
import math
from functools import partial
from typing import Literal, Sequence

import chz
from datasets import Dataset, load_from_disk
from tinker_cookbook import renderers
from tinker_cookbook.recipes.math_rl.math_env import MathEnv, extract_boxed
from tinker_cookbook.recipes.math_rl.math_grading import extract_boxed as extract_boxed_grading
from tinker_cookbook.rl.problem_env import ProblemGroupBuilder
from tinker_cookbook.rl.types import EnvGroupBuilder, RLDataset, RLDatasetBuilder
from tinker_cookbook.tokenizer_utils import get_tokenizer

logger = logging.getLogger(__name__)

class SpiralMathTestEnv(MathEnv):
    """
    Environment for SPIRAL math test problems.
    """

    def __init__(
        self, problem: str, answer: str, renderer: renderers.Renderer, convo_prefix: list[renderers.Message] | None = None):
        super().__init__(problem, answer, renderer, convo_prefix)

    @classmethod
    def question_suffix(cls) -> str:
        return " Please reason step by step, and put your final answer within \\boxed{}."

class SpiralMathTestDataset(RLDataset):
    """
    Test dataset for SPIRAL math problems (AIME, AMC, MATH, Minerva, Olympiad).

    This dataset is designed to be used as `maybe_test_dataset` in tinker training
    for periodic evaluation on held-out math problems.
    """

    def __init__(
        self,
        data_path: str,
        renderer: renderers.Renderer,
    ):
        """
        Initialize SPIRAL math test dataset.

        Args:
            data_path: Path to the dataset directory (e.g., "data/aime")
            renderer: Renderer for formatting prompts
        """
        self.ds = load_from_disk(data_path)
        self.renderer = renderer
        # For test datasets, always use group_size=1 (no parallel envs per problem)
        self.group_size = 1

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        """Get all problems as a single batch (parallel evaluation)."""
        if index > 0:
            return []
        return [
            builder
            for row in self.ds
            if (builder := self._make_env_group_builder(row, self.group_size)) is not None
        ]

    def __len__(self) -> int:
        """Number of batches in the dataset."""
        return 1  # All problems evaluated in parallel

    def _make_env_group_builder(
        self, x: dict[str, str], group_size: int
    ) -> ProblemGroupBuilder | None:
        """Create environment group builder for a single problem."""
        problem = x.get("problem", "")
        answer = x.get("answer", "")

        if not (problem and answer):
            logger.warning(f"Missing problem or answer: {x}")
            return None

        # Convert answer to string if it's a number (float/int)
        if isinstance(answer, (int, float)):
            answer = str(answer)

        return ProblemGroupBuilder(
            env_thunk=partial(
                SpiralMathTestEnv, problem, answer, self.renderer, convo_prefix=None
            ),
            num_envs=group_size,
        )


@chz.chz
class SpiralMathTestDatasetBuilder(RLDatasetBuilder):
    """
    Builder for creating SPIRAL math test datasets.

    Can load from multiple dataset splits (aime, amc, math, minerva, olympiad_bench).
    """

    data_paths: list[str] | str  # Can be single path or list of paths
    model_name_for_tokenizer: str
    renderer_name: str

    async def __call__(self) -> tuple[None, RLDataset]:
        """
        Build math test dataset(s).

        Returns:
            Tuple of (None, test_dataset) since we only provide test data,
            not training data (training uses self-play games).
        """

        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)

        # Handle single path or multiple paths
        paths = [self.data_paths] if isinstance(self.data_paths, str) else self.data_paths

        if len(paths) == 1:
            # Single dataset
            test_dataset = SpiralMathTestDataset(
                data_path=paths[0],
                renderer=renderer,
            )
        else:
            # Multiple datasets - concatenate them
            test_dataset = ConcatenatedMathTestDataset(
                data_paths=paths,
                renderer=renderer,
            )

        # Return (None, test_dataset) - no training dataset, only test
        return (None, test_dataset)


class ConcatenatedMathTestDataset(RLDataset):
    """
    Concatenates multiple SPIRAL math test datasets.

    This allows evaluating on multiple benchmarks (e.g., AIME + AMC + MATH).
    """

    def __init__(
        self,
        data_paths: list[str],
        renderer: renderers.Renderer,
    ):
        """
        Initialize concatenated test dataset.

        Args:
            data_paths: List of dataset directory paths
            renderer: Renderer for formatting prompts
        """
        self.datasets = [
            SpiralMathTestDataset(
                data_path=path,
                renderer=renderer,
            )
            for path in data_paths
        ]

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        """Get all problems from all datasets (parallel evaluation)."""
        if index > 0:
            return []
        builders = []
        for dataset in self.datasets:
            builders.extend(dataset.get_batch(0))
        return builders

    def __len__(self) -> int:
        """Total number of batches across all datasets."""
        return 1  # All problems evaluated in parallel
