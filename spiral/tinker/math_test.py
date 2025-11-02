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
from tinker_cookbook.rl.metric_util import RLTestSetEvaluator, compute_trajectory_metrics
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tqdm.asyncio import tqdm
import tinker
from tinker_cookbook.completers import TinkerTokenCompleter
from tinker_cookbook.rl.rollouts import do_group_rollout

logger = logging.getLogger(__name__)

logger.setLevel(logging.DEBUG)

class SpiralMathTestEnv(MathEnv):
    """
    Environment for SPIRAL math test problems.

    Always works with a list of acceptable answers internally.
    Single answers are converted to single-element lists.
    """

    def __init__(
        self,
        problem: str,
        answer_candidates: list[str],  # Always a list now
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None
    ):
        """
        Initialize math environment with list of acceptable answers.

        Args:
            problem: The math problem text
            answer_candidates: List of acceptable answers (any match = correct)
            renderer: Renderer for formatting prompts
            convo_prefix: Optional conversation prefix
        """
        # Store all acceptable answers
        self.answer_candidates = answer_candidates

        # Pass first answer to parent for compatibility
        answer_str = answer_candidates[0] if answer_candidates else ""
        super().__init__(problem, answer_str, renderer, convo_prefix)

    @classmethod
    def question_suffix(cls) -> str:
        return " Please reason step by step, and put your final answer within \\boxed{}."

    def grade_answer(self, answer: str) -> bool:
        """
        Grade an answer against all acceptable answers.

        Returns True if the given answer matches ANY of the acceptable answers.
        """
        from tinker_cookbook.recipes.math_rl.math_env import safe_grade

        # Try matching against each acceptable answer
        for gold_answer in self.answer_candidates:
            if safe_grade(answer, gold_answer, self.grader, self.timeout):
                return True

        return False

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

        # Normalize answer to list of strings (unified representation)
        answer_candidates = self._normalize_to_list(answer)

        return ProblemGroupBuilder(
            env_thunk=partial(
                SpiralMathTestEnv, problem, answer_candidates, self.renderer, convo_prefix=None
            ),
            num_envs=group_size,
        )

    @staticmethod
    def _normalize_to_list(answer) -> list[str]:
        """
        Normalize answer to list of string candidates.

        Handles:
        - Single values (str, int, float) -> [str(value)]
        - Lists -> [str(item) for item in list]
        - Nested lists -> flattened list of strings
        """
        if isinstance(answer, list):
            # Flatten nested lists and convert all to strings
            result = [str(item) for item in answer]
        else:
            result = [str(answer)]
        return result

class MathTestEvaluator(RLTestSetEvaluator):
    """
    Evaluator for SPIRAL math test problems.
    """

    def __init__(self, dataset: SpiralMathTestDataset, max_tokens: int, name: str | None = None):
        super().__init__(dataset, max_tokens, name)

    async def __call__(self, sampling_client: tinker.SamplingClient) -> dict[str, float]:
        policy = TinkerTokenCompleter(sampling_client, max_tokens=self.max_tokens)
        trajectory_groups_P = await tqdm.gather(
            *[do_group_rollout(builder, policy) for builder in self.env_group_builders_P],
            desc=f"Evaluating {self.name}",
        )
        taglist_P = [builder.logging_tags() for builder in self.env_group_builders_P]
        metrics = compute_trajectory_metrics(trajectory_groups_P, taglist_P)
        if self.name is not None:
            metrics = {f"{self.name}/{k}": v for k, v in metrics.items()}
        return metrics

@chz.chz
class SpiralMathTestDatasetBuilder(RLDatasetBuilder):
    """
    Builder for creating SPIRAL math test evaluators.

    Creates separate evaluators for each math dataset path instead of concatenating.
    This allows separate tracking of metrics per benchmark (AIME, AMC, MATH, etc.).
    """

    data_paths: list[str] | str  # Can be single path or list of paths
    model_name_for_tokenizer: str
    renderer_name: str
    max_tokens: int

    async def __call__(self) -> tuple[None, None]:
        """
        Build math test evaluators.

        Note: This returns (None, None) because we don't provide datasets.
        Instead, use create_evaluators() to get separate evaluators for each benchmark.

        Returns:
            Tuple of (None, None)
        """
        # Math test evaluation now happens via separate evaluators, not datasets
        return (None, None)

    def create_evaluators(self) -> list:
        """
        Create separate evaluators for each math dataset path.

        Returns:
            List of MathTestEvaluator objects, one per dataset path
        """
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)

        # Handle single path or multiple paths
        paths = [self.data_paths] if isinstance(self.data_paths, str) else self.data_paths

        evaluators = []
        for path in paths:
            # Extract dataset name from path (e.g., "data/aime" -> "aime")
            dataset_name = path.rstrip('/').split('/')[-1]

            # Create dataset
            test_dataset = SpiralMathTestDataset(
                data_path=path,
                renderer=renderer,
            )

            evaluator = MathTestEvaluator(
                dataset=test_dataset,
                max_tokens=self.max_tokens,  # Default, can be overridden
                name=f"eval/math/{dataset_name}"  # e.g., "eval/math/aime"
            )
            evaluators.append(evaluator)

        return evaluators
