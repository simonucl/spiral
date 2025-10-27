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

import asyncio
import logging
from typing import Any, Dict, List, Sequence

import weave
from tinker import ModelInput
from tinker_cookbook.completers import TokenCompleter
from tinker_cookbook.rl.types import (Env, EnvGroupBuilder, Observation,
                                      Trajectory, TrajectoryGroup, Transition)
from tinker_cookbook.tokenizer_utils import get_tokenizer

from spiral.tinker.env import SpiralTwoPlayerEnvGroupBuilder

logger = logging.getLogger(__name__)


def check_valid_observation(ob: Observation) -> bool:
    """Check if observation is valid (non-empty ModelInput)."""
    return isinstance(ob, ModelInput) and len(ob.chunks) > 0


def _postprocess_observation(trajectory: Trajectory) -> Dict[str, List[Any]]:
    """
    Postprocess trajectory for weave logging.
    Decodes observations and actions to human-readable format.
    """
    model_name = "Qwen/Qwen3-8B"
    tokenizer = get_tokenizer(model_name)
    results: Dict[str, List[Any]] = {
        "observation": [],
        "action": [],
        "reward": [],
        "episode_done": [],
        "metrics": [],
    }
    if len(trajectory.transitions) == 0:
        return results
    for transition in trajectory.transitions:
        results["observation"].append(tokenizer.decode(transition.ob.to_ints()))
        results["action"].append(tokenizer.decode(transition.ac.tokens))
        results["reward"].append(transition.reward)
        results["episode_done"].append(transition.episode_done)
        results["metrics"].append(transition.metrics)
    return results


@weave.op(postprocess_output=_postprocess_observation)
async def do_single_rollout(policy: TokenCompleter, env: Env) -> Trajectory:
    """
    Perform a single rollout with the given policy and environment.

    Args:
        policy: Policy to use for action selection
        env: Environment to interact with

    Returns:
        Trajectory containing all transitions
    """
    transitions = []
    ob, stop_condition = await env.initial_observation()
    while True:
        if not check_valid_observation(ob):
            break
        ac_with_logprobs = await policy(ob, stop_condition)
        step_result = await env.step(ac_with_logprobs.tokens)
        transition = Transition(
            ob=ob,
            ac=ac_with_logprobs,
            reward=step_result.reward,
            episode_done=step_result.episode_done,
            metrics=step_result.metrics,
        )
        transitions.append(transition)
        ob = step_result.next_observation
        stop_condition = step_result.next_stop_condition
        if step_result.episode_done:
            break
    return Trajectory(transitions=transitions, final_ob=ob)


async def do_group_rollout(
    env_group_builder: EnvGroupBuilder, policy: TokenCompleter
) -> TrajectoryGroup:
    """
    Perform a group rollout with the given policy and environment group.

    Args:
        env_group_builder: Builder for creating environment group
        policy: Policy to use for action selection

    Returns:
        TrajectoryGroup containing all trajectories and computed rewards
    """
    envs_G: Sequence[Env] = await env_group_builder.make_envs()
    logger.debug(f"Created {len(envs_G)} envs")
    trajectories_G = await asyncio.gather(
        *[do_single_rollout(policy, env) for env in envs_G]
    )
    rewards_and_metrics_G = await env_group_builder.compute_group_rewards(
        trajectories_G
    )
    rewards_G, metrics_G = zip(*rewards_and_metrics_G, strict=True)
    logger.debug(f"Metrics: {metrics_G}")
    return TrajectoryGroup(trajectories_G, list(rewards_G), list(metrics_G))


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
    max_retries = (
        env_group_builder.max_draw_retries if env_group_builder.filter_draw else 0
    )

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
