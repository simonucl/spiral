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

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict

import numpy as np
import wandb
from tinker_cookbook.utils.ml_log import (
    JsonLogger,
    Logger,
    MultiplexLogger,
    NeptuneLogger,
    PrettyPrintLogger,
    TrackioLogger,
    WandbLogger,
    configure_logging_module,
    dump_config,
)

def setup_logging(
    log_dir: str,
    wandb_project: str | None = None,
    wandb_name: str | None = None,
    config: Any | None = None,
    do_configure_logging_module: bool = True,
    reinit: str = "create_new",
) -> Logger:
    """
    Set up logging infrastructure with multiple backends.

    Args:
        log_dir: Directory for logs
        wandb_project: W&B project name (if None, W&B logging is skipped)
        wandb_name: W&B run name
        config: Configuration object to log
        do_configure_logging_module: Whether to configure the logging module

    Returns:
        MultiplexLogger that combines all enabled loggers
    """
    # Create log directory
    log_dir_path = Path(log_dir).expanduser()
    log_dir_path.mkdir(parents=True, exist_ok=True)

    # Initialize loggers
    loggers = []

    # Always add JSON logger
    loggers.append(JsonLogger(log_dir_path))

    # Always add pretty print logger
    loggers.append(PrettyPrintLogger())

    # Add W&B logger if available and configured
    if wandb_project:
        loggers.append(
            WandbLoggerWithReinit(
                project=wandb_project,
                config=config,
                log_dir=log_dir_path,
                wandb_name=wandb_name,
                reinit=reinit,
            )
        )

    # Create multiplex logger
    ml_logger = MultiplexLogger(loggers)

    # Log initial configuration
    if config is not None:
        ml_logger.log_hparams(config)

    if do_configure_logging_module:
        configure_logging_module(str(log_dir_path / "logs.log"))

    return ml_logger

class WandbLoggerWithReinit(WandbLogger):
    def __init__(
        self,
        project: str | None = None,
        config: Any | None = None,
        log_dir: str | Path | None = None,
        wandb_name: str | None = None,
        reinit: str = "default",
    ):
        # Initialize wandb run
        assert wandb is not None  # For type checker
        self.run = wandb.init(
            project=project,
            config=dump_config(config) if config else None,
            dir=str(log_dir) if log_dir else None,
            name=wandb_name,
            reinit=reinit,
        )

    def log_hparams(self, config: Any) -> None:
        """Log hyperparameters to wandb."""
        if self.run and wandb is not None:
            self.run.config.update(dump_config(config), allow_val_change=True)

    def log_metrics(self, metrics: Dict[str, Any], step: int | None = None) -> None:
        """Log metrics to wandb."""
        if self.run and wandb is not None:
            self.run.log(metrics, step=step)

    def close(self) -> None:
        """Close wandb run."""
        if self.run and wandb is not None:
            self.run.finish()

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

    # Compute per-role statistics (generalized for any number of players)
    player_rewards = defaultdict(list)
    player_invalid_count = defaultdict(int)
    player_turn_count = defaultdict(int)
    player_baselines = defaultdict(list)
    player_gen_lengths = defaultdict(list)  # Track generation lengths per player
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

                # Count invalid actions and track generation lengths
                for transition in trajectory.transitions:
                    if transition.metrics.get("invalid_action", 0) == 1:
                        player_invalid_count[player_id] += 1
                    # Track generation length (number of tokens generated)
                    gen_length = len(transition.ac.tokens)
                    player_gen_lengths[player_id].append(gen_length)

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

            # Add generation length metrics
            if len(player_gen_lengths[player_id]) > 0:
                metrics[f"{prefix}/player_{player_id}/mean_gen_length"] = np.mean(
                    player_gen_lengths[player_id]
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

    # Overall generation length metrics
    all_gen_lengths = [length for lengths in player_gen_lengths.values() for length in lengths]
    if len(all_gen_lengths) > 0:
        metrics[f"{prefix}/mean_gen_length"] = np.mean(all_gen_lengths)

    # Game outcome metrics
    total_games = draw_count + decisive_count
    if total_games > 0:
        metrics[f"{prefix}/draw_count"] = draw_count
        metrics[f"{prefix}/draw_rate"] = draw_count / total_games
        metrics[f"{prefix}/decisive_count"] = decisive_count

    return metrics

def convert_to_json_serializable(obj: Any) -> Any:
    """Convert numpy types and other non-serializable types to JSON-serializable types."""
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