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

"""Tinker integration for SPIRAL self-play training."""

from spiral.tinker.dataset import SpiralRLDatasetBuilder
from spiral.tinker.eval import GameEvaluator
from spiral.tinker.eval.math_test import (
    SpiralMathTestDataset,
    SpiralMathTestDatasetBuilder,
)
from spiral.tinker.renderer import SpiralRenderer, get_spiral_renderer
from spiral.tinker.training import (
    PopulationManager,
    SpiralTwoPlayerEnv,
    SpiralTwoPlayerEnvGroupBuilder,
    create_spiral_train_loop,
    do_group_rollout,
    do_group_rollout_with_draw_retry,
    do_single_rollout,
    train_step,
)
from spiral.tinker.training.env import (
    ILLEGAL_MOVE_REWARD,
    INVALID_ACTION,
    TwoPlayerCoordinator,
)
from spiral.tinker.utils import compute_trajectory_metrics

__all__ = [
    # Dataset
    "SpiralRLDatasetBuilder",
    "SpiralMathTestDataset",
    "SpiralMathTestDatasetBuilder",
    # Environment
    "TwoPlayerCoordinator",
    "SpiralTwoPlayerEnv",
    "SpiralTwoPlayerEnvGroupBuilder",
    "ILLEGAL_MOVE_REWARD",
    "INVALID_ACTION",
    # Renderer
    "SpiralRenderer",
    "get_spiral_renderer",
    # Rollouts
    "do_single_rollout",
    "do_group_rollout",
    "do_group_rollout_with_draw_retry",
    # Training
    "create_spiral_train_loop",
    "train_step",
    "PopulationManager",
    # Evaluation
    "GameEvaluator",
    # Utils
    "compute_trajectory_metrics",
]
