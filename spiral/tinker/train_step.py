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

"""Training step implementation for SPIRAL with per-turn advantage computation."""

import logging
import time
from typing import Any, List, Sequence

import numpy as np
import tinker
import torch
from tinker import AdamParams, TensorData, types
from tinker_cookbook.rl.metrics import compute_post_kl
from tinker_cookbook.rl.types import (EnvGroupBuilder, TrajectoryGroup,
                                      Transition)
from tinker_cookbook.tokenizer_utils import Tokenizer
from tinker_cookbook.utils.misc_utils import timed

from spiral.tinker.utils import compute_trajectory_metrics

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


async def train_step(
    cfg: Any,
    i_batch: int,
    training_client: tinker.TrainingClient,
    tokenizer: Tokenizer,
    env_group_builders_P: Sequence[EnvGroupBuilder],
    trajectory_groups_P: list[TrajectoryGroup],
    sampling_client: tinker.SamplingClient | None = None,
) -> dict[str, Any]:
    """
    Custom training step for SPIRAL with RAE-based advantage computation.

    This replaces Tinker's default training step to give us full control over
    per-turn advantage computation with discounting, matching OAT's logic:

    For each turn in a player's trajectory:
        - If use_intermediate_rewards: advantage = player_reward * (gamma ** steps_from_end)
        - Else: advantage = player_reward (no discounting)

    Where player_reward is the RAE-adjusted reward from compute_group_rewards().

    Args:
        cfg: Training configuration with fields:
            - gamma: Discount factor for turn-level rewards (default: 1.0)
            - use_intermediate_rewards: Whether to discount earlier turns (default: True)
            - batch_size: Number of turns to sample for training
            - loss_fn: Loss function name (e.g., "importance_sampling")
            - learning_rate: Learning rate for optimizer
            - filter_zero_adv: Whether to filter turns with zero advantage (optional)
        i_batch: Current batch/step number
        training_client: Tinker training client
        tokenizer: Tokenizer for decoding (for logging)
        env_group_builders_P: List of environment group builders (for logging tags)
        trajectory_groups_P: List of trajectory groups from rollouts

    Returns:
        Dictionary of training metrics
    """
    metrics = {}
    t_start = time.time()

    # Step 0: Update role baselines once per batch (before computing advantages)
    # Group trajectory groups by env_id and update baselines per environment per role
    with timed("update_baselines", metrics):
        from collections import defaultdict
        traj_groups_by_env = defaultdict(list)
        builder_by_env = {}

        for i, traj_group in enumerate(trajectory_groups_P):
            if i < len(env_group_builders_P):
                builder = env_group_builders_P[i]
                env_id = getattr(builder, 'env_id', None)
                if env_id:
                    traj_groups_by_env[env_id].append(traj_group)
                    builder_by_env[env_id] = builder  # Keep reference to any builder for this env

        # Update baselines for each environment
        for env_id, env_traj_groups in traj_groups_by_env.items():
            builder = builder_by_env[env_id]
            new_baselines = builder.update_role_baselines(env_traj_groups)
            if new_baselines:
                # Log the updated baselines
                for player_id, baseline in new_baselines.items():
                    metrics[f"train/{env_id}/player_{player_id}_baseline_updated"] = baseline

    # Step 1: Collect all turns with their advantages
    # Each trajectory group contains 2 trajectories (one per player)
    transitions: List[Transition] = []

    with timed("compute_turn_advantages", metrics):
        for traj_group in trajectory_groups_P:
            player_rewards = traj_group.get_total_rewards()
            assert len(player_rewards) == 2, "Expected 2 player rewards"
            for player_reward, trajectory in zip(
                player_rewards, traj_group.trajectories_G
            ):
                num_turns = len(trajectory.transitions)
                for turn_idx, transition in enumerate(trajectory.transitions):
                    transition.reward = player_reward * (
                        cfg.gamma ** (num_turns - turn_idx - 1)
                    )

                    # Filter zero rewards if configured
                    if (
                        hasattr(cfg, "filter_zero_adv")
                        and cfg.filter_zero_adv
                        and transition.reward == 0
                    ):
                        continue

                    transitions.append(transition)

    logger.info(f"Total transitions: {len(transitions)}")

    # Step 2: Normalize advantages by total number of transitions
    # Instead of subsampling, we use all transitions but normalize advantages
    # to keep the effective batch size consistent
    total_transitions = len(transitions)
    advantage_scale = cfg.batch_size / total_transitions if total_transitions > 0 else 1.0
    logger.info(f"Advantage scale factor: {advantage_scale:.4f} (batch_size={cfg.batch_size}, total_transitions={total_transitions})")

    # Step 3: Prepare training datums from transitions
    with timed("prepare_datums", metrics):
        training_datums = []
        for transition in transitions:
            ob_tokens = transition.ob.to_ints()
            ac_tokens = transition.ac.tokens
            ob_len = len(ob_tokens) - 1  # -1 due to shifting
            tokens = ob_tokens + ac_tokens
            input_tokens = tokens[:-1]
            target_tokens = tokens[1:]

            # Logprobs: 0 for observation tokens, actual logprobs for action tokens
            all_logprobs = [0.0] * ob_len + transition.ac.logprobs

            # Advantages: 0 for observation tokens, scaled advantage value for action tokens
            # Scale by advantage_scale to normalize for varying total_transitions
            scaled_advantage = transition.reward * advantage_scale
            all_advantages = [0.0] * ob_len + [scaled_advantage] * (
                len(input_tokens) - ob_len
            )

            assert (
                len(input_tokens)
                == len(target_tokens)
                == len(all_logprobs)
                == len(all_advantages)
            ), (
                f"Length mismatch: input={len(input_tokens)}, "
                f"target={len(target_tokens)}, logprobs={len(all_logprobs)}, "
                f"advantages={len(all_advantages)}"
            )

            datum = types.Datum(
                model_input=types.ModelInput.from_ints(tokens=input_tokens),
                loss_fn_inputs={
                    "target_tokens": TensorData.from_torch(torch.tensor(target_tokens)),
                    "logprobs": TensorData.from_torch(torch.tensor(all_logprobs)),
                    "advantages": TensorData.from_torch(torch.tensor(all_advantages)),
                },
            )
            training_datums.append(datum)

    logger.info(f"Prepared {len(training_datums)} training datums")

    # Step 4: Training step (forward-backward + optimizer step)
    with timed("train", metrics):
        fwd_bwd_future = await training_client.forward_backward_async(
            training_datums, loss_fn=cfg.loss_fn
        )
        optim_step_future = await training_client.optim_step_async(
            adam_params=AdamParams(
                learning_rate=cfg.learning_rate, beta1=0.9, beta2=0.95, eps=1e-8
            )
        )
        fwd_bwd_result = await fwd_bwd_future.result_async()
        _ = await optim_step_future.result_async()

    # Step 5: Compute metrics
    with timed("compute_metrics", metrics):
        # Compute policy entropy and sampler-learner KL difference
        act_token_logprobs = []
        act_token_diffs = []

        for i in range(min(len(transitions), len(fwd_bwd_result.loss_fn_outputs))):
            transition = transitions[i]
            train_output = fwd_bwd_result.loss_fn_outputs[i]

            # Get action logprobs from transition
            sample_logprobs = torch.tensor(transition.ac.maybe_logprobs)
            act_token_logprobs.extend(transition.ac.maybe_logprobs)

            # Get training logprobs for action tokens
            train_logprobs = train_output["logprobs"].to_torch()[
                -len(transition.ac.maybe_logprobs) :
            ]

            act_token_diffs.append(sample_logprobs - train_logprobs)

        if len(act_token_diffs) > 0:
            act_token_diffs = torch.cat(act_token_diffs)
            kl_sample_train_v1 = act_token_diffs.mean().item()
            kl_sample_train_v2 = 0.5 * (act_token_diffs**2).mean().item()

            metrics["sampler/token_entropy"] = (
                -torch.tensor(act_token_logprobs).mean().item()
            )
            metrics["train/kl_sample_train_v1"] = kl_sample_train_v1
            metrics["train/kl_sample_train_v2"] = kl_sample_train_v2

        # Add loss metrics from forward-backward
        metrics.update(**{f"train/{k}": v for k, v in fwd_bwd_result.metrics.items()})

        # Add basic statistics
        metrics["train/num_turns"] = len(transitions)
        metrics["train/num_trajectory_groups"] = len(trajectory_groups_P)
        metrics["train/advantage_scale"] = advantage_scale

        # Compute average advantage and reward statistics (before scaling)
        advantages = [transition.reward for transition in transitions]
        if len(advantages) > 0:
            metrics["train/mean_advantage"] = np.mean(advantages)
            metrics["train/std_advantage"] = np.std(advantages)
            metrics["train/max_advantage"] = np.max(advantages)
            metrics["train/min_advantage"] = np.min(advantages)

            # Also track scaled advantages
            scaled_advantages = [adv * advantage_scale for adv in advantages]
            metrics["train/mean_scaled_advantage"] = np.mean(scaled_advantages)
            metrics["train/std_scaled_advantage"] = np.std(scaled_advantages)

        # Compute comprehensive trajectory metrics using utility function
        trajectory_metrics = compute_trajectory_metrics(trajectory_groups_P, prefix="train")
        metrics.update(trajectory_metrics)

        # Compute win/loss/draw rates per game and overall from training trajectories
        # Group by env_id
        from collections import defaultdict
        env_outcomes = defaultdict(lambda: {"win": 0, "loss": 0, "draw": 0})
        overall_outcomes = {"win": 0, "loss": 0, "draw": 0}

        for i, traj_group in enumerate(trajectory_groups_P):
            total_rewards = traj_group.get_total_rewards()

            # Determine outcome (from player 0's perspective)
            if len(total_rewards) >= 2:
                if total_rewards[0] > total_rewards[1]:
                    outcome = "win"
                elif total_rewards[0] < total_rewards[1]:
                    outcome = "loss"
                else:
                    outcome = "draw"

                # Get env_id from the corresponding builder
                env_id = None
                if i < len(env_group_builders_P) and hasattr(env_group_builders_P[i], 'env_id'):
                    env_id = env_group_builders_P[i].env_id

                # Update counts
                if env_id:
                    env_outcomes[env_id][outcome] += 1
                overall_outcomes[outcome] += 1

        # Compute rates per game
        for env_id, outcomes in env_outcomes.items():
            total = sum(outcomes.values())
            if total > 0:
                metrics[f"train/{env_id}/win_rate"] = outcomes["win"] / total
                metrics[f"train/{env_id}/loss_rate"] = outcomes["loss"] / total
                metrics[f"train/{env_id}/draw_rate"] = outcomes["draw"] / total

        # Compute overall rates
        total_overall = sum(overall_outcomes.values())
        if total_overall > 0:
            metrics["train/overall/win_rate"] = overall_outcomes["win"] / total_overall
            metrics["train/overall/loss_rate"] = overall_outcomes["loss"] / total_overall
            metrics["train/overall/draw_rate"] = overall_outcomes["draw"] / total_overall

    metrics["time/train_total"] = time.time() - t_start

    # Optional: Compute post-update KL divergence
    if cfg.compute_post_kl and sampling_client is not None:
        with timed("compute_post_kl", metrics):
            post_kl_metrics = await compute_post_kl(training_datums, sampling_client)
            metrics.update(post_kl_metrics)

    return metrics
