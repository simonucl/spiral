import concurrent.futures
import logging
import os
import sys
import time
from typing import List, Optional

import numpy as np
import textarena as ta
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import requests
from tqdm import tqdm
from utils import CSVManagerData

from evals.game.inference import VLLMInferenceClient, VLLMServerManager
from evals.game.utils import CSVManagerData
from spiral.agents.random import RandomAgent
from spiral.envs import make_env

USE_LLM_WRAPPER = {
    "DontSayIt-v1": True,
    "KuhnPoker-v1": True,
    "LiarsDice-v1": True,
    "TwentyQuestions-v0": True,
    "SimpleNegotiation-v1": True,
    "SimpleNegotiation-v2": True,
    "Poker-v0": True,
    "SimpleBlindAuction-v0": True,
    "TicTacToe-v0": False,
    "Snake-v0": False,
    "ConnectFour-v0": False,
    "SimpleTak-v0": False,
    "Nim-v0": False,
}

IS_CHAT = {
    "DontSayIt-v1": True,
    "KuhnPoker-v1": False,
    "LiarsDice-v1": False,
    "TwentyQuestions-v0": True,
    "SimpleNegotiation-v1": True,
    "SimpleNegotiation-v2": True,
    "Poker-v0": False,
    "SimpleBlindAuction-v0": True,
    "TicTacToe-v0": False,
    "Snake-v0": False,
    "ConnectFour-v0": False,
    "SimpleTak-v0": False,
    "Nim-v0": False,
}


def get_openrouter_model_ids():
    response = requests.get(f"https://openrouter.ai/api/v1/models")
    models = response.json()
    return [model["id"] for model in models["data"]]


class VLLMCollector:
    def __init__(
        self,
        env_ids: List[str],
        checkpoint_paths: List[str],
        output_dir: str,
        max_new_tokens: int,
        max_workers: int,
        temperature: float = 0.7,
        top_p: float = 0.9,
        vllm_max_num_seq: int = 64,
        tensor_parallel_size: int = 1,
        gpus: Optional[List[int]] = None,
        base_port: int = 8000,
    ):
        self.env_ids = env_ids
        self.output_dir = output_dir
        self.tensor_parallel_size = tensor_parallel_size
        self.base_port = base_port
        self.max_workers = max_workers
        self.vllm_max_num_seq = vllm_max_num_seq

        # Support multiple checkpoints
        self.checkpoint_paths = (
            [checkpoint_paths]
            if isinstance(checkpoint_paths, str)
            else checkpoint_paths
        )
        self.primary_checkpoint = self.checkpoint_paths[
            0
        ]  # Use first checkpoint as primary
        self.checkpoint_paths = list(set(self.checkpoint_paths))  # prevent duplicates

        # generation parameters
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        self.gpus = list(range(torch.cuda.device_count())) if gpus is None else gpus

        self.logger = logging.getLogger(__name__)
        self.server_managers = (
            {}
        )  # Dictionary to store server managers by checkpoint path
        self.vllm_clients = {}  # Dictionary to store clients by checkpoint path

    def _make_env(self, env_id: str) -> ta.Env:
        use_llm_obs_wrapper = USE_LLM_WRAPPER.get(env_id, False)
        env = make_env(env_id, use_llm_obs_wrapper=use_llm_obs_wrapper)
        return env

    def __enter__(self):
        """Context manager entry: Setup vLLM clients for each model."""
        # Calculate GPUs per model
        if len(self.checkpoint_paths) > 1:
            gpus_per_model = len(self.gpus) // len(self.checkpoint_paths)
            if gpus_per_model < self.tensor_parallel_size:
                self.logger.warning(
                    f"Not enough GPUs for tensor parallelism across all models. Reducing tensor_parallel_size to {gpus_per_model}"
                )
                self.tensor_parallel_size = max(1, gpus_per_model)

        # Start servers for each checkpoint
        for i, checkpoint_path in enumerate(self.checkpoint_paths):
            # Assign GPUs to each model
            if len(self.checkpoint_paths) > 1:
                model_gpus = self.gpus[i * gpus_per_model : (i + 1) * gpus_per_model]
            else:
                model_gpus = self.gpus

            # Create server manager for this checkpoint
            server_manager = VLLMServerManager(
                model_path=checkpoint_path,
                max_seq_len=self.max_new_tokens * 2,
                gpus=model_gpus,
                tensor_parallel_size=self.tensor_parallel_size,
                base_port=self.base_port
                + i * 100,  # Use different port ranges for each model
                max_num_seq=self.vllm_max_num_seq,
            )
            server_manager.start_servers()
            self.server_managers[checkpoint_path] = server_manager

            # Initialize clients for this checkpoint
            self.vllm_clients[checkpoint_path] = []
            for j in range(server_manager.num_servers):
                client = server_manager.get_client(j)
                self.vllm_clients[checkpoint_path].append(client)

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for checkpoint_path, server_manager in self.server_managers.items():
            server_manager.stop_servers()
        self.vllm_clients = {}
        self.server_managers = {}

    def _get_vllm_client(
        self, episode_id: int, checkpoint_path: str
    ) -> VLLMInferenceClient:
        client_idx = episode_id % len(self.vllm_clients[checkpoint_path])
        return self.vllm_clients[checkpoint_path][client_idx]

    def evaluate(
        self,
        num_episodes: int,
        model_name: str,
        opponent_names: List[str] = ["google/gemini-2.0-flash-001"],
        env_ids: Optional[List[str]] = None,
    ) -> None:
        # Start CSVManagerData as a context manager
        with CSVManagerData(self.output_dir, episode_type="eval") as csv_manager:

            def run_episode(
                episode_id: int, env_id: str, episode_type: str, opponent_name: str
            ) -> None:
                try:
                    # get episode starting time
                    t0 = time.time()

                    # Get client for target model
                    if model_name == "random":
                        target_client = RandomAgent(env_id=env_id)
                    elif model_name in self.checkpoint_paths:
                        target_client = self._get_vllm_client(episode_id, model_name)
                    else:
                        target_client = ta.agents.OpenRouterAgent(model_name)

                    if opponent_name == "random":
                        opponent_client = RandomAgent(env_id=env_id)
                    elif opponent_name in self.checkpoint_paths:
                        opponent_client = self._get_vllm_client(
                            episode_id, opponent_name
                        )
                    else:
                        opponent_client = ta.agents.OpenRouterAgent(opponent_name)

                    # assign player roles
                    agent_idx = int(np.random.uniform() < 0.5)
                    agents = {agent_idx: target_client, 1 - agent_idx: opponent_client}
                    models = {agent_idx: model_name, 1 - agent_idx: opponent_name}
                    # Create & wrap environment
                    env = self._make_env(env_id)
                    env_id = env.env_id
                    env.reset(num_players=2)

                    episode_data = []
                    step_count = 0
                    done = False

                    while not done:
                        player_id, observation = env.get_observation()

                        # route to correct model
                        model = agents[player_id]
                        if isinstance(model, VLLMInferenceClient):
                            # Use vLLM for inference
                            formatted_observation, reasoning, action = (
                                model.generate_text(
                                    prompt=observation,
                                    max_new_tokens=self.max_new_tokens,
                                    temperature=self.temperature,
                                    top_p=self.top_p,
                                    is_chat=IS_CHAT.get(env_id, False),
                                )
                            )
                        else:
                            formatted_observation, reasoning, action = (
                                None,
                                None,
                                model(observation),
                            )

                        episode_data.append(
                            {
                                "episode_id": episode_id,
                                "env_id": env_id,
                                "model_name": models[player_id],
                                "player_id": player_id,
                                "observation": observation,
                                "formatted_observation": formatted_observation,
                                "reasoning": reasoning,
                                "action": action,
                                "step": step_count,
                            }
                        )

                        # done, info = env.step(action=action)
                        done, info = env.step(
                            action=f"[{action}]" if action[0] != "[" else action
                        )
                        step_count += 1

                    # Episode done
                    rewards = env.close()

                    # Add rewards to episode data and send to CSV manager
                    for step_data in episode_data:
                        csv_manager.add_episode(
                            [
                                step_data["episode_id"],
                                step_data["env_id"],
                                step_data["model_name"],
                                step_data["player_id"],
                                step_data["observation"],
                                step_data["formatted_observation"],
                                step_data["reasoning"],
                                step_data["action"],
                                step_data["step"],
                                len(episode_data),
                                rewards[step_data["player_id"]],
                            ]
                        )

                    # Write relevant data to the CSV logging manager
                    t1 = time.time()
                    if rewards[agent_idx] > rewards[1 - agent_idx]:
                        model_outcome = "win"
                    elif rewards[agent_idx] < rewards[1 - agent_idx]:
                        model_outcome = "loss"
                    else:
                        model_outcome = "draw"

                    completion_status = (
                        "invalid"
                        if (rewards[0] == -1 and rewards[1] == 0)
                        or (rewards[1] == -1 and rewards[0] == 0)
                        else "complete"
                    )
                    csv_manager.add_episode_information(
                        [
                            episode_type,
                            episode_id,
                            env_id,
                            models[agent_idx],
                            models[1 - agent_idx],
                            t0,
                            t1,
                            t1 - t0,
                            step_count,
                            rewards[agent_idx],
                            rewards[1 - agent_idx],
                            completion_status,
                            model_outcome,
                        ]
                    )
                except Exception as e:
                    print(f"Episode collection failed with exception {e}")

            # Parallel collection
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.max_workers
            ) as executor:
                futures = []
                for env_id in env_ids:
                    for opponent_name in opponent_names:
                        for i in range(num_episodes):
                            futures.append(
                                executor.submit(
                                    run_episode, i, env_id, "eval", opponent_name
                                )
                            )

                # futures = [executor.submit(run_episode, i) for i in range(num_episodes)]
                progress_bar = tqdm(total=len(futures), desc="Collecting episodes")
                for future in concurrent.futures.as_completed(futures):
                    future.result()  # wait for the result, but nothing to do with it here
                    progress_bar.update(1)
                progress_bar.close()
