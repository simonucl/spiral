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

"""Training components for SPIRAL."""

from spiral.tinker.training.env import (
    SpiralTwoPlayerEnv,
    SpiralTwoPlayerEnvGroupBuilder,
)
from spiral.tinker.training.population import PopulationManager
from spiral.tinker.training.rollouts import (
    do_group_rollout,
    do_group_rollout_and_filter_constant_reward,
    do_group_rollout_with_draw_retry,
    do_single_rollout,
)
from spiral.tinker.training.train import create_spiral_train_loop
from spiral.tinker.training.train_step import train_step

__all__ = [
    "create_spiral_train_loop",
    "train_step",
    "SpiralTwoPlayerEnv",
    "SpiralTwoPlayerEnvGroupBuilder",
    "PopulationManager",
    "do_group_rollout",
    "do_group_rollout_and_filter_constant_reward",
    "do_group_rollout_with_draw_retry",
    "do_single_rollout",
]
