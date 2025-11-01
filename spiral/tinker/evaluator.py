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

"""Game evaluation for SPIRAL Tinker implementation."""

import asyncio
import logging
import random
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional
import weave
import numpy as np

import textarena as ta
import tinker
from tinker_cookbook.completers import TinkerMessageCompleter
from tinker_cookbook.eval.evaluators import SamplingClientEvaluator
from tinker_cookbook.utils import ml_log
from tinker_cookbook.utils.misc_utils import dict_mean
from tqdm.asyncio import tqdm

from spiral.agents.random import RandomAgent
from spiral.agents.utils import get_valid_action_parser
from spiral.envs import make_env
from spiral.tinker.renderer import INVALID_ACTION, get_spiral_renderer
from spiral.tinker.utils import convert_to_json_serializable
from spiral.utils import extract_boxed_answer

logger = logging.getLogger(__name__)


class AsyncEvalRunner:
    """Manages asynchronous evaluation runs that log to a separate wandb run."""

    def __init__(self, eval_logger: Optional[ml_log.Logger] = None):
        """
        Initialize async evaluation runner.

        Args:
            eval_logger: Logger for evaluation metrics (separate from training)
        """
        self.eval_logger = eval_logger
        self.running_tasks = []  # Track background evaluation tasks

    async def run_eval_async(
        self,
        evaluators: list,
        sampling_client: tinker.SamplingClient,
        step: int,
    ):
        """
        Run evaluations asynchronously and log to eval wandb run.

        Args:
            evaluators: List of evaluator objects to run
            sampling_client: Tinker sampling client for inference
            step: Training step number
        """
        try:
            logger.info(f"[EVAL] Starting async evaluation at step {step}")
            t_start = time.time()

            eval_metrics = {}
            for evaluator in evaluators:
                metrics = await evaluator(sampling_client)
                eval_metrics.update(metrics)

            eval_metrics["eval/time"] = time.time() - t_start
            eval_metrics = convert_to_json_serializable(eval_metrics)

            # Log to separate eval wandb run
            if self.eval_logger:
                self.eval_logger.log_metrics(eval_metrics, step=step)

            logger.info(
                f"[EVAL] Completed async evaluation at step {step} "
                f"in {eval_metrics['eval/time']:.2f}s"
            )

        except Exception as e:
            logger.error(
                f"[EVAL] Error in async evaluation at step {step}: {e}",
                exc_info=True,
            )

    def schedule_eval(
        self,
        evaluators: list,
        sampling_client: tinker.SamplingClient,
        step: int,
    ):
        """
        Schedule an evaluation to run in the background.

        Args:
            evaluators: List of evaluator objects to run
            sampling_client: Tinker sampling client for inference
            step: Training step number
        """
        # Create background task
        task = asyncio.create_task(
            self.run_eval_async(evaluators, sampling_client, step),
            name=f"eval_step_{step}",
        )
        self.running_tasks.append((step, task))
        logger.info(f"[EVAL] Scheduled async evaluation for step {step}")

    async def wait_all(self):
        """Wait for all running evaluation tasks to complete."""
        if not self.running_tasks:
            return

        logger.info(
            f"[EVAL] Waiting for {len(self.running_tasks)} "
            "background evaluations to complete..."
        )
        for step, task in self.running_tasks:
            await task

        self.running_tasks.clear()
        logger.info("[EVAL] All background evaluations completed")


class GameEvaluator(SamplingClientEvaluator):
    """
    Evaluator for game playing against fixed opponents (random, LLM-based).

    Similar to OAT's SelfPlayActor.run_eval_episode() but adapted for Tinker.
    Inherits from SamplingClientEvaluator for tinker-cookbook compatibility.
    """

    def __init__(
        self,
        eval_env_ids: List[str],
        eval_opponent_names: List[str],
        model_name: str,
        eval_use_llm_obs_wrappers: List[bool] | None = None,
        eval_games_per_matchup: int = 16,
        prompt_template: str = "qwen3",
        model_player_id: int = 0,
    ):
        """
        Initialize the game evaluator.

        Args:
            eval_env_ids: List of environment IDs to evaluate on
            eval_opponent_names: List of opponent types ("random" or LLM model names)
            model_name: Model name for creating the renderer
            eval_use_llm_obs_wrappers: Per-env observation wrapper config
            eval_games_per_matchup: Number of games per (env, opponent) matchup
            prompt_template: Template name for formatting observations
            model_player_id: Which player ID the model plays as (0 or 1)
        """
        self.eval_env_ids = eval_env_ids
        self.eval_opponent_names = eval_opponent_names
        self.model_name = model_name
        self.eval_use_llm_obs_wrappers = eval_use_llm_obs_wrappers or [
            True
        ] * len(eval_env_ids)
        self.eval_games_per_matchup = eval_games_per_matchup
        self.prompt_template = prompt_template
        self.model_player_id = model_player_id

        # Create env_id -> use_llm_obs_wrapper mapping
        self.env_to_llm_obs_wrapper = dict(
            zip(self.eval_env_ids, self.eval_use_llm_obs_wrappers)
        )

        # Initialize LLM opponents (non-random)
        self.llm_opponents = {}
        for opponent_name in eval_opponent_names:
            if opponent_name != "random":
                self.llm_opponents[opponent_name] = ta.agents.OpenRouterAgent(
                    opponent_name
                )

    async def __call__(
        self, sampling_client: tinker.SamplingClient, max_tokens: int = 2048
    ) -> Dict[str, Any]:
        """
        Run evaluation against all opponents on all environments.

        Args:
            sampling_client: Tinker sampling client for model inference
            max_tokens: Max tokens for generation

        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Starting game evaluation...")
        t_start = time.time()

        # Create renderer and policy from sampling client
        renderer = get_spiral_renderer(self.model_name, self.prompt_template)
        policy = TinkerMessageCompleter(
            sampling_client=sampling_client,
            renderer=renderer,
            max_tokens=max_tokens,
        )

        # Generate all eval runs
        eval_runs = []
        for env_id in self.eval_env_ids:
            for opponent_name in self.eval_opponent_names:
                # Skip if random opponent not supported for this env
                if opponent_name == "random":
                    try:
                        RandomAgent(env_id)
                    except NotImplementedError:
                        logger.warning(
                            f"Random opponent not supported for {env_id}, skipping"
                        )
                        continue

                for game_idx in range(self.eval_games_per_matchup):
                    eval_runs.append((env_id, opponent_name, game_idx))

        # Run all evaluation games in parallel
        results = await tqdm.gather(
            *[
                self._run_single_game(policy, env_id, opponent_name, game_idx)
                for env_id, opponent_name, game_idx in eval_runs
            ],
            desc="Evaluating games",
        )

        # Aggregate metrics
        metrics = self._aggregate_metrics(results)
        metrics["eval/game_eval_time"] = time.time() - t_start
        metrics["eval/total_games"] = len(results)

        logger.info(f"Game evaluation completed in {metrics['eval/game_eval_time']:.2f}s")
        return metrics

    async def _run_single_game(
        self,
        policy: TinkerMessageCompleter,
        env_id: str,
        opponent_name: str,
        game_idx: int,
    ) -> Dict[str, Any]:
        """Run a single evaluation game."""
        model_pid = self.model_player_id
        opponent_pid = 1 - model_pid

        # Create opponent
        if opponent_name == "random":
            opponent = RandomAgent(env_id)
        else:
            opponent = self.llm_opponents[opponent_name]

        # Create environment
        use_llm_obs_wrapper = self.env_to_llm_obs_wrapper[env_id]
        env = make_env(env_id, use_llm_obs_wrapper)
        env.reset(num_players=2, seed=int(time.time_ns()) + game_idx)
        env.state.error_allowance = 0

        # Play game
        turn_counter = 0
        done = False
        invalid_action_player = None
        model_gen_lengths = []  # Track generation lengths for model actions

        while not done:
            pid, observation = env.get_observation()

            # Get action from appropriate agent
            if pid == model_pid:
                action, gen_length = await self._model_act(policy, observation, env_id)
                model_gen_lengths.append(gen_length)
            else:
                action = opponent(observation)

            # Take step
            done, info = env.step(action)

            if action == INVALID_ACTION:
                done = True
                invalid_action_player = pid

            turn_counter += 1

        # Get final rewards
        if invalid_action_player is not None:
            rewards = {0: 0.5, 1: 0.5}
            rewards[invalid_action_player] = -1.5
        else:
            rewards = env.close()

        # Determine outcome from model's perspective
        if rewards[model_pid] > rewards[opponent_pid]:
            outcome = "win"
        elif rewards[model_pid] < rewards[opponent_pid]:
            outcome = "loss"
        else:
            outcome = "draw"

        return {
            "env_id": env_id,
            "opponent_name": opponent_name,
            "outcome": outcome,
            "invalid_move": invalid_action_player is not None,
            "invalid_move_by_model": invalid_action_player == model_pid,
            "num_turns": turn_counter,
            "model_reward": rewards[model_pid],
            "opponent_reward": rewards[opponent_pid],
            "model_pid": model_pid,
            "model_gen_lengths": model_gen_lengths,  # List of generation lengths
        }

    @weave.op()
    async def _model_act(
        self, policy: TinkerMessageCompleter, observation: str, env_id: str
    ) -> tuple[str, int, str]:
        """
        Get model action for a given observation.

        Returns:
            Tuple of (action_string, generation_length)
        """
        # Format observation with template
        renderer = get_spiral_renderer(self.model_name, self.prompt_template)
        model_input = renderer.build_generation_prompt(
            [{"role": "user", "content": observation}],
        )

        # Sample directly to get token information
        response = await policy.sampling_client.sample_async(
            model_input,
            num_samples=1,
            sampling_params=tinker.SamplingParams(
                temperature=1.0,
                max_tokens=policy.max_tokens,
                stop=policy.stop_condition,
            ),
        )

        # Get token count
        gen_length = len(response.sequences[0].tokens)

        # Parse the response
        parsed_message, _success = policy.renderer.parse_response(response.sequences[0].tokens)
        extracted_action = parsed_message["content"]

        if extracted_action is None:
            return INVALID_ACTION, gen_length

        # Validate action against action space
        try:
            action_parser = get_valid_action_parser(env_id)

            if env_id in ["DontSayIt-v0", "SimpleNegotiation-v1"]:
                # Free-form chat, no validation
                return extracted_action, gen_length
            elif env_id == "SimpleNegotiation-v2":
                # Regex patterns
                patterns = action_parser(observation)
                for pattern in patterns:
                    if pattern.match(extracted_action):
                        return extracted_action, gen_length
                return INVALID_ACTION, gen_length
            else:
                # Standard validation
                valid_actions = action_parser(observation)
                if extracted_action in valid_actions:
                    return extracted_action, gen_length
                return INVALID_ACTION, gen_length
        except NotImplementedError:
            # No action parser for this env, accept as-is
            return extracted_action, gen_length
        except Exception as e:
            logger.warning(f"Error validating action: {e}")
            return extracted_action, gen_length

    def _compute_group_metrics(
        self, results: List[Dict[str, Any]], prefix: str
    ) -> Dict[str, Any]:
        """Compute metrics for a group of results."""
        if not results:
            return {}

        total = len(results)
        metrics = {}

        # Outcome rates
        outcomes = {"win": 0, "loss": 0, "draw": 0}
        for r in results:
            outcomes[r["outcome"]] += 1

        metrics.update({
            f"{prefix}/win_rate": outcomes["win"] / total,
            f"{prefix}/loss_rate": outcomes["loss"] / total,
            f"{prefix}/draw_rate": outcomes["draw"] / total,
            f"{prefix}/num_games": total,
        })

        # Invalid action rates
        invalid_moves = sum(1 for r in results if r["invalid_move"])
        invalid_by_model = sum(1 for r in results if r["invalid_move_by_model"])
        metrics[f"{prefix}/invalid_move_rate"] = invalid_moves / total
        metrics[f"{prefix}/model_invalid_rate"] = invalid_by_model / total

        # Average continuous metrics using dict_mean
        continuous_metrics = [
            {"num_turns": r["num_turns"], "model_reward": r["model_reward"]}
            for r in results
        ]
        avg_metrics = dict_mean(continuous_metrics)
        metrics[f"{prefix}/avg_turns"] = avg_metrics["num_turns"]
        metrics[f"{prefix}/avg_reward"] = avg_metrics["model_reward"]

        # Generation length metrics
        all_gen_lengths = [
            length for r in results for length in r["model_gen_lengths"]
        ]
        if all_gen_lengths:
            metrics[f"{prefix}/mean_gen_length"] = np.mean(all_gen_lengths)

        return metrics

    def _aggregate_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results into metrics."""
        metrics = {}

        # Group by env_id only (for per-game metrics)
        grouped_by_env = defaultdict(list)
        for result in results:
            grouped_by_env[result["env_id"]].append(result)

        # Group by env_id and opponent_name (for per-matchup metrics)
        grouped_by_matchup = defaultdict(list)
        for result in results:
            key = (result["env_id"], result["opponent_name"])
            grouped_by_matchup[key].append(result)

        # Compute metrics per game (across all opponents for that game)
        for env_id, env_results in grouped_by_env.items():
            metrics.update(self._compute_group_metrics(env_results, f"eval/{env_id}"))

        # Compute metrics per matchup (env + opponent)
        for (env_id, opponent_name), matchup_results in grouped_by_matchup.items():
            metrics.update(
                self._compute_group_metrics(
                    matchup_results, f"eval/{env_id}/{opponent_name}"
                )
            )

        # Compute overall metrics
        if results:
            metrics.update(self._compute_group_metrics(results, "eval/overall"))

        return metrics
