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
from typing import Any, Dict, List, Optional, Sequence

import tinker
import weave
from tinker import ModelInput
from tinker_cookbook.completers import TokenCompleter, TinkerTokenCompleter
from tinker_cookbook.rl.types import (
    Env,
    EnvGroupBuilder,
    Observation,
    Trajectory,
    TrajectoryGroup,
    Transition,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils.trace import scope

from spiral.tinker.training.env import SpiralTwoPlayerEnvGroupBuilder

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

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
    env_group_builder: EnvGroupBuilder,
    policy: TokenCompleter | list[TokenCompleter],
) -> TrajectoryGroup:
    """
    Perform a group rollout with given policy/policies.

    Args:
        env_group_builder: Builder for creating environment group
        policy: Single policy (self-play) or list of policies (FSP)

    Returns:
        TrajectoryGroup containing all trajectories and computed rewards
    """
    envs_G: Sequence[Env] = await env_group_builder.make_envs()
    logger.debug(f"Created {len(envs_G)} envs")

    # Normalize to list for uniform handling
    policies = policy if isinstance(policy, list) else [policy]
    trajectories_G = await asyncio.gather(
        *[do_single_rollout(policies[i % len(policies)], env) for i, env in enumerate(envs_G)]
    )

    rewards_and_metrics_G = await env_group_builder.compute_group_rewards(trajectories_G)
    rewards_G, metrics_G = zip(*rewards_and_metrics_G, strict=True)
    logger.debug(f"Metrics: {metrics_G}")
    return TrajectoryGroup(trajectories_G, list(rewards_G), list(metrics_G))


async def do_group_rollout_with_draw_retry(
    env_group_builder: SpiralTwoPlayerEnvGroupBuilder,
    policy: TokenCompleter,
    opponent_policy: TokenCompleter | None = None,
) -> TrajectoryGroup | None:
    """
    Perform a group rollout with draw retry logic.

    If the game ends in a draw (all rewards == 0) and filter_draw is enabled,
    retry the game up to max_draw_retries times.

    Args:
        env_group_builder: Environment group builder (with draw retry settings)
        policy: Policy to use for rollout (player 0 in FSP mode)
        opponent_policy: Optional opponent policy (player 1 in FSP mode)

    Returns:
        TrajectoryGroup with non-draw game, or draw game after max retries
        Returns None if all trajectories have constant rewards (filtered out)
    """
    max_retries = (
        env_group_builder.max_draw_retries if env_group_builder.filter_draw else 0
    )

    for retry in range(max_retries + 1):
        # Perform rollout (pass list of policies for FSP, single policy for self-play)
        policies = [policy, opponent_policy] if opponent_policy else policy
        traj_group = await do_group_rollout(env_group_builder, policies)

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


@scope
async def do_group_rollout_and_filter_constant_reward(
    sampling_client: tinker.SamplingClient,
    env_group_builder: EnvGroupBuilder,
    max_tokens: int,
    do_remove_constant_reward_groups: bool,
    opponent_sampling_client: Optional[tinker.SamplingClient] = None,
) -> TrajectoryGroup | None:
    """
    Do a group rollout with draw retry (if builder supports it) and optionally filter constant rewards.

    Args:
        sampling_client: Sampling client for policy
        env_group_builder: Environment group builder
        max_tokens: Max tokens for generation
        do_remove_constant_reward_groups: Whether to filter constant reward groups
        opponent_sampling_client: Optional opponent sampling client for FSP mode

    Returns:
        TrajectoryGroup or None if filtered
    """
    policy = TinkerTokenCompleter(sampling_client, max_tokens=max_tokens)
    opponent_policy = None
    if opponent_sampling_client is not None:
        opponent_policy = TinkerTokenCompleter(opponent_sampling_client, max_tokens=max_tokens)

    # Use custom rollout with draw retry if this is a SPIRAL builder
    if isinstance(env_group_builder, SpiralTwoPlayerEnvGroupBuilder):
        trajectory_group = await do_group_rollout_with_draw_retry(
            env_group_builder, policy, opponent_policy=opponent_policy
        )
    else:
        # Standard rollout for non-SPIRAL builders
        policies = [policy, opponent_policy] if opponent_policy else policy
        trajectory_group = await do_group_rollout(env_group_builder, policies)

    # Remove if all trajectories have the same reward
    trajectory_groups = [trajectory_group]
    # if do_remove_constant_reward_groups:
    #     trajectory_groups = remove_constant_reward_groups(trajectory_groups)
    if len(trajectory_groups) == 0:
        return None
    return trajectory_groups[0]
