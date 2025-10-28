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

"""Utility functions for Tinker-based SPIRAL training."""

from typing import Any, Dict


def compute_trajectory_metrics(trajectory_groups, prefix: str = "train") -> Dict[str, Any]:
    """
    Compute comprehensive metrics from trajectory groups for training/evaluation.

    This function computes:
    - Per-trajectory statistics (turns per game)
    - Per-player statistics (rewards, win rates, invalid actions)
    - Game outcome statistics (draws, decisive games)
    - Role baseline statistics (from RAE)

    Args:
        trajectory_groups: List of TrajectoryGroup objects from rollouts
        prefix: Metric name prefix (e.g., "train" or "eval")

    Returns:
        Dictionary of computed metrics with keys like:
        - "{prefix}/mean_turns_per_trajectory"
        - "{prefix}/player_{id}/mean_reward"
        - "{prefix}/player_{id}/win_rate"
        - "{prefix}/player_{id}/invalid_count"
        - "{prefix}/total_invalid_rate"
        - "{prefix}/draw_rate"
        etc.
    """
    from collections import defaultdict
    import numpy as np

    metrics = {}

    # Compute per-trajectory statistics (turns per game)
    game_lengths = []
    for traj_group in trajectory_groups:
        for trajectory in traj_group.trajectories_G:
            # Get game length from last transition metrics
            if len(trajectory.transitions) > 0:
                last_metrics = trajectory.transitions[-1].metrics
                game_length = last_metrics.get("game_length", len(trajectory.transitions))
                if game_length > 0:  # Only count if game_length was set (game ended)
                    game_lengths.append(game_length)

    if len(game_lengths) > 0:
        metrics[f"{prefix}/mean_turns_per_trajectory"] = np.mean(game_lengths)
        metrics[f"{prefix}/std_turns_per_trajectory"] = np.std(game_lengths)
        metrics[f"{prefix}/max_turns_per_trajectory"] = np.max(game_lengths)
        metrics[f"{prefix}/min_turns_per_trajectory"] = np.min(game_lengths)

    # Compute per-role statistics (generalized for any number of players)
    player_rewards = defaultdict(list)
    player_invalid_count = defaultdict(int)
    player_turn_count = defaultdict(int)
    player_baselines = defaultdict(list)
    draw_count = 0
    decisive_count = 0

    for traj_group in trajectory_groups:
        total_rewards = traj_group.get_total_rewards()

        # Track draw vs decisive games (all players get 0 = draw)
        if len(total_rewards) > 0:
            if all(r == 0 for r in total_rewards):
                draw_count += 1
            else:
                decisive_count += 1

        # Collect per-player statistics
        for i, (reward, trajectory) in enumerate(
            zip(total_rewards, traj_group.trajectories_G)
        ):
            # Determine player_id from trajectory transitions
            if len(trajectory.transitions) > 0:
                player_id = trajectory.transitions[0].metrics.get("player_id", i)

                player_rewards[player_id].append(reward)
                player_turn_count[player_id] += len(trajectory.transitions)

                # Count invalid actions
                for transition in trajectory.transitions:
                    if transition.metrics.get("invalid_action", 0) == 1:
                        player_invalid_count[player_id] += 1

        # Collect baseline metrics
        for traj_metrics in traj_group.metrics_G:
            role = traj_metrics.get("role")
            baseline = traj_metrics.get("baseline")
            if baseline is not None and role is not None:
                player_baselines[role].append(baseline)

    # Compute metrics for each player
    for player_id in sorted(player_rewards.keys()):
        rewards = player_rewards[player_id]
        if len(rewards) > 0:
            metrics[f"{prefix}/player_{player_id}/mean_reward"] = np.mean(rewards)
            metrics[f"{prefix}/player_{player_id}/win_rate"] = np.mean(
                [r > 0 for r in rewards]
            )
            metrics[f"{prefix}/player_{player_id}/invalid_count"] = player_invalid_count[
                player_id
            ]

            if player_turn_count[player_id] > 0:
                metrics[f"{prefix}/player_{player_id}/invalid_rate"] = (
                    player_invalid_count[player_id] / player_turn_count[player_id]
                )

        # Add baseline metrics if available
        if len(player_baselines[player_id]) > 0:
            metrics[f"{prefix}/player_{player_id}_baseline"] = np.mean(
                player_baselines[player_id]
            )

    # Overall invalid action metrics
    total_invalid_count = sum(player_invalid_count.values())
    total_turn_count = sum(player_turn_count.values())
    metrics[f"{prefix}/total_invalid_count"] = total_invalid_count
    if total_turn_count > 0:
        metrics[f"{prefix}/total_invalid_rate"] = total_invalid_count / total_turn_count

    # Game outcome metrics
    total_games = draw_count + decisive_count
    if total_games > 0:
        metrics[f"{prefix}/draw_count"] = draw_count
        metrics[f"{prefix}/draw_rate"] = draw_count / total_games
        metrics[f"{prefix}/decisive_count"] = decisive_count

    return metrics

def convert_to_json_serializable(obj: Any) -> Any:
    """Convert numpy types and other non-serializable types to JSON-serializable types."""
    import numpy as np
    
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    return obj