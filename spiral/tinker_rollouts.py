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

"""Rollout utilities for SPIRAL with draw filtering."""

import logging

from tinker_cookbook.completers import TokenCompleter
from tinker_cookbook.rl.rollouts import do_group_rollout
from tinker_cookbook.rl.types import TrajectoryGroup

from spiral.tinker_env import SpiralTwoPlayerEnvGroupBuilder

logger = logging.getLogger(__name__)


async def do_group_rollout_with_draw_retry(
    env_group_builder: SpiralTwoPlayerEnvGroupBuilder,
    policy: TokenCompleter,
) -> TrajectoryGroup | None:
    """
    Perform a group rollout with draw retry logic.

    If the game ends in a draw (all rewards == 0) and filter_draw is enabled,
    retry the game up to max_draw_retries times.

    Args:
        env_group_builder: Environment group builder (with draw retry settings)
        policy: Policy to use for rollout

    Returns:
        TrajectoryGroup with non-draw game, or draw game after max retries
        Returns None if all trajectories have constant rewards (filtered out)
    """
    max_retries = env_group_builder.max_draw_retries if env_group_builder.filter_draw else 0

    for retry in range(max_retries + 1):
        # Perform rollout
        traj_group = await do_group_rollout(env_group_builder, policy)

        # Check for draw (all rewards == 0)
        total_rewards = traj_group.get_total_rewards()
        is_draw = all(r == 0 for r in total_rewards)

        # If not a draw, or draw filtering is disabled, return immediately
        if not is_draw or not env_group_builder.filter_draw:
            if retry > 0:
                logger.info(
                    f"[{env_group_builder.env_id}] Non-draw game after {retry} retries"
                )
            return traj_group

        # If we've reached max retries, accept the draw
        if retry >= max_retries:
            logger.info(
                f"[{env_group_builder.env_id}] Draw detected but max retries ({max_retries}) "
                f"reached, accepting draw with rewards: {total_rewards}"
            )
            return traj_group

        # Otherwise, log and retry
        logger.info(
            f"[{env_group_builder.env_id}] Draw detected (rewards: {total_rewards}), "
            f"retry {retry + 1}/{max_retries}"
        )

    # Should never reach here, but return last trajectory group just in case
    return traj_group
