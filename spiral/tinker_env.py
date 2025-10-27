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

"""TextArena environment for SPIRAL self-play training with Tinker."""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, ClassVar, Sequence

import textarena as ta
from tinker import ModelInput, types
from tinker_cookbook.completers import StopCondition
from tinker_cookbook.rl.types import (
    Action,
    Env,
    EnvGroupBuilder,
    Metrics,
    Observation,
    StepResult,
    Trajectory,
)

from spiral.agents.random import RandomAgent
from spiral.agents.utils import get_valid_action_parser
from spiral.envs import make_env
from spiral.tinker_renderer import INVALID_ACTION, SpiralRenderer
from spiral.utils import EMA

logger = logging.getLogger(__name__)

ILLEGAL_MOVE_REWARD = -1.5


class TwoPlayerCoordinator:
    """
    Coordinates a single two player game between two players.
    Adapted from tinker-cookbook/recipes/multiplayer_rl/text_arena/env.py
    """

    def __init__(self, shared_env: ta.Env):
        self.shared_env = shared_env  # Should already be reset
        self.condition = asyncio.Condition()
        self.illegal_player_id: int | None = None
        self.rewards: dict[int, float] = {}

    @property
    def state(self) -> ta.State:
        return self.shared_env.state  # type: ignore

    @property
    def current_player_id(self) -> int:
        """Get the current player ID from the environment state."""
        return self.state.current_player_id

    @property
    def game_done(self) -> bool:
        """Check if the game is done. Either the game state is done, or some player made an illegal move"""
        return self.state.done or self.illegal_player_id is not None

    async def wait_across_env(self, player_id: int) -> None:
        """Wait until the opponent has finished their turn"""
        async with self.condition:
            await self.condition.wait_for(
                lambda: self.current_player_id == player_id or self.game_done
            )

    async def make_move(self, player_id: int, move: str) -> None:
        """Make a move and notify waiting players."""
        async with self.condition:
            # Ensure it's actually this player's turn
            before_player_id = self.current_player_id

            if not self.game_done and (self.current_player_id != player_id):
                raise ValueError(
                    f"Not player {player_id}'s turn (current: {self.current_player_id})"
                )

            done, _ = self.shared_env.step(move)

            if done:
                self.rewards = self.shared_env.close()
            else:
                # we will know that the move is illegal if the next player's id has not changed after the move
                if self.current_player_id == before_player_id:
                    self.illegal_player_id = before_player_id

            # Notify all waiting players about the state change
            self.condition.notify_all()


@dataclass
class SpiralTwoPlayerEnv(Env):
    """Two player TextArena environment for SPIRAL."""

    player_id: int  # 0 or 1
    env_id: str
    coordinator: TwoPlayerCoordinator
    self_play: bool
    renderer: SpiralRenderer
    opponent_policy: Any | None  # MessageCompleter or RandomAgent

    def __post_init__(self):
        assert self.self_play == (
            self.opponent_policy is None
        ), "If self_play is True, opponent_policy must be None"

        # Initialize action parser for validation
        try:
            self.action_parser = get_valid_action_parser(self.env_id)
        except NotImplementedError:
            logger.info(
                f"No action parser for {self.env_id}, will skip action validation"
            )
            self.action_parser = None

    @property
    def stop_condition(self) -> StopCondition:
        return self.renderer.get_stop_sequences()

    async def wait_for_turn(self) -> None:
        """If the game is not done, wait until the opponent's to finish playing their turn"""
        if not self.coordinator.game_done:
            if self.self_play:
                await self.coordinator.wait_across_env(self.player_id)
            else:
                await self.opponent_step()

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        # Check if it's actually our turn by looking at the shared env
        if not self.coordinator.game_done:
            current_player_id, _ = self.coordinator.shared_env.get_observation()
            if current_player_id != self.player_id:
                await self.wait_for_turn()
        obs = self.get_observation()
        return obs, self.stop_condition

    async def opponent_step(self) -> None:
        """When not self_play, the opponent policy takes a step on the shared environment"""
        assert self.opponent_policy is not None
        opponent_player_id, opponent_observation_str = (
            self.coordinator.shared_env.get_observation()
        )
        assert isinstance(opponent_player_id, int) and isinstance(
            opponent_observation_str, str
        )
        assert opponent_player_id == 1 - self.player_id, (
            f"Opponent player ID should be 1 - [the id of the policy player], "
            f"{opponent_player_id=}, {self.player_id=}"
        )

        # Handle different opponent types
        if isinstance(self.opponent_policy, RandomAgent):
            # Random agent - just call it directly
            opponent_action_text = self.opponent_policy(opponent_observation_str)
        else:
            # MessageCompleter (e.g., TinkerMessageCompleter for LLM opponents)
            opponent_convo = [{"role": "user", "content": opponent_observation_str}]
            opponent_response = await self.opponent_policy(opponent_convo)
            opponent_action_text = opponent_response["content"]

        await self.coordinator.make_move(opponent_player_id, opponent_action_text)

    async def step(self, action: Action) -> StepResult:
        """Take a step in the environment."""
        if self.coordinator.game_done:
            return self.get_done_step()
        assert (
            self.coordinator.current_player_id == self.player_id
        ), "Not the current player's turn"

        # Parse action from tokens (extraction only, no validation)
        action_message, _ = self.renderer.parse_response(action)
        action_text = action_message["content"]

        # Validate action against current game state
        if action_text != INVALID_ACTION:
            action_text = self._validate_action(action_text)

        # Check for invalid action
        if action_text == INVALID_ACTION:
            # Mark this player as making an illegal move and notify opponent
            async with self.coordinator.condition:
                self.coordinator.illegal_player_id = self.player_id
                self.coordinator.condition.notify_all()
            return StepResult(
                reward=ILLEGAL_MOVE_REWARD,
                episode_done=True,
                next_observation=ModelInput.empty(),
                next_stop_condition=self.stop_condition,
                metrics={"invalid_action": 1},
            )

        # Make move
        await self.coordinator.make_move(self.player_id, action_text)

        # Wait for opponent turn (if game continues)
        await self.wait_for_turn()

        return StepResult(
            reward=self.compute_reward(),
            episode_done=self.coordinator.game_done,
            next_observation=self.get_observation(),
            next_stop_condition=self.stop_condition,
            metrics={"invalid_action": 0},
        )

    def _validate_action(self, extracted_action: str) -> str:
        """
        Validate extracted action against current game state.

        Args:
            extracted_action: Action extracted from \\boxed{}

        Returns:
            Validated action string or INVALID_ACTION
        """
        # If no action parser, accept as-is
        if self.action_parser is None:
            return extracted_action

        # Get current observation from shared environment
        _, observation_str = self.coordinator.shared_env.get_observation()

        # Special handling for different environment types
        if self.env_id in ["DontSayIt-v0", "SimpleNegotiation-v1"]:
            # These environments accept free-form chat, no validation needed
            return extracted_action

        elif self.env_id == "SimpleNegotiation-v2":
            # This uses regex patterns for validation
            patterns = self.action_parser(observation_str)
            for pattern in patterns:
                if pattern.match(extracted_action):
                    return extracted_action
            logger.warning(
                f"Action '{extracted_action}' doesn't match any pattern for {self.env_id}"
            )
            return INVALID_ACTION

        else:
            # Standard validation: check if action is in valid action list
            try:
                valid_actions = self.action_parser(observation_str)
                if extracted_action in valid_actions:
                    return extracted_action
                else:
                    logger.warning(
                        f"Action '{extracted_action}' not in valid actions for {self.env_id}, "
                        f"valid actions: {valid_actions}"
                    )
                    return INVALID_ACTION
            except Exception as e:
                logger.warning(
                    f"Error validating action '{extracted_action}': {e}, accepting as-is"
                )
                return extracted_action

    def get_done_step(self) -> StepResult:
        """Return a done step result with computed reward."""
        return StepResult(
            reward=self.compute_reward(),
            episode_done=True,
            next_observation=types.ModelInput.empty(),
            next_stop_condition=self.stop_condition,
            metrics={},
        )

    def compute_reward(self) -> float:
        """Compute reward for this player."""
        # Illegal move penalty for this player
        if self.coordinator.illegal_player_id == self.player_id:
            return ILLEGAL_MOVE_REWARD

        # Opponent made illegal move - this player wins
        if self.coordinator.illegal_player_id is not None:
            return 0.5

        # Game rewards
        if self.coordinator.rewards:
            return self.coordinator.rewards[self.player_id]

        return 0.0

    def get_observation(self) -> types.ModelInput:
        """Get current observation for this player."""
        current_player_id, observation_str = self.coordinator.shared_env.get_observation()

        if not self.coordinator.game_done:
            assert isinstance(current_player_id, int) and isinstance(observation_str, str)
            assert current_player_id == self.player_id, (
                f"Observation should be for the current player, obs: {observation_str}, "
                f"current_player_id: {current_player_id}, player_id: {self.player_id}"
            )

        # Use renderer to build generation prompt
        # If game is done, observation_str might be stale, but that's ok
        # since step() will return immediately with get_done_step()
        return self.renderer.build_generation_prompt(
            [{"role": "user", "content": observation_str}]
        )


@dataclass
class SpiralTwoPlayerEnvGroupBuilder(EnvGroupBuilder):
    """Builder for groups of two player TextArena environments for SPIRAL."""

    env_id: str
    renderer: SpiralRenderer
    num_envs: int = 2
    self_play: bool = True
    opponent_policy: Any | None = None  # MessageCompleter or RandomAgent

    # SPIRAL-specific settings
    filter_draw: bool = True
    max_draw_retries: int = 5
    use_role_baseline: bool = True
    role_baseline_ema_gamma: float = 0.95
    use_llm_obs_wrapper: bool = True
    use_intermediate_rewards: bool = True  # Whether to distribute final reward to all turns
    gamma: float = 1.0  # Discount factor for intermediate rewards

    # Role baselines (shared across all instances for same env)
    # We use a class variable to share state
    _role_baseline_emas: ClassVar[dict] = {}

    num_players: ClassVar[int] = 2

    def __post_init__(self):
        """Initialize role baselines if needed."""
        if self.use_role_baseline:
            # Initialize EMA for each role if not already done
            if self.env_id not in self._role_baseline_emas:
                self._role_baseline_emas[self.env_id] = {
                    0: EMA(self.role_baseline_ema_gamma),
                    1: EMA(self.role_baseline_ema_gamma),
                }
                logger.info(
                    f"Initialized role baselines for {self.env_id} with gamma={self.role_baseline_ema_gamma}"
                )

    async def make_envs(self) -> Sequence[Env]:
        """Create a group of environments sharing the same TextArena game."""
        if self.num_envs % 2 != 0:
            raise ValueError("this env requires an even number of environments (players)")

        def _construct_coordinator() -> TwoPlayerCoordinator:
            """
            During training, the coordinator performs necessary blocking/synchronization,
            so that the policies can take turns to make moves on the shared environment,
            across different Environment objects.
            """
            shared_env = make_env(self.env_id, self.use_llm_obs_wrapper)
            shared_env.reset(num_players=self.num_players)
            # Set error allowance to 0 for strict action validation
            shared_env.state.error_allowance = 0
            return TwoPlayerCoordinator(shared_env=shared_env)

        envs = []
        for _ in range(self.num_envs // 2):
            if self.self_play:
                coordinator = _construct_coordinator()
                # if self_play, then we need to share the same coordinator across all environments
                coordinators = [coordinator for _ in range(self.num_players)]
            else:
                # if not self_play, we can just create a different coordinator for each environment
                coordinators = [_construct_coordinator() for _ in range(self.num_players)]

            envs += [
                SpiralTwoPlayerEnv(
                    player_id=i,
                    coordinator=coordinators[i],
                    renderer=self.renderer,
                    self_play=self.self_play,
                    opponent_policy=self.opponent_policy,
                    env_id=self.env_id,
                )
                for i in range(2)
            ]
        return envs

    async def compute_group_rewards(
        self, trajectory_group: list[Trajectory]
    ) -> list[tuple[float, Metrics]]:
        """
        Compute final rewards for each trajectory in the group.
        Implements Role-conditioned Advantage Estimation (RAE) by subtracting
        role-specific baselines from the rewards.

        Note: Intermediate reward discounting (gamma < 1.0) is not yet implemented.
        With the default gamma=1.0, sparse rewards at episode end are equivalent
        to distributing the full final reward to all turns.

        Args:
            trajectory_group: List of trajectories for the group, per role (player 0, player 1)

        Returns:
            List of (adjusted_reward, metrics) tuples
        """
        results = []

        for i, traj in enumerate(trajectory_group):
            # Get raw reward from trajectory (sum of per-step rewards)
            raw_reward = sum(transition.reward for transition in traj.transitions)

            # Determine player role (even index = player 0, odd = player 1)
            player_id = i % 2

            # Apply role baseline if enabled
            if self.use_role_baseline:
                # Get baseline before updating (to be unbiased)
                baseline = self._role_baseline_emas[self.env_id][player_id].get()

                # Update role baseline EMA with current reward
                self._role_baseline_emas[self.env_id][player_id].update(raw_reward)

                # Subtract baseline from reward (RAE)
                adjusted_reward = raw_reward - baseline

                metrics = {
                    "role": player_id,
                    "raw_reward": raw_reward,
                    "baseline": baseline,
                    "adjusted_reward": adjusted_reward,
                }
            else:
                adjusted_reward = raw_reward
                metrics = {"role": player_id, "raw_reward": raw_reward}

            results.append((adjusted_reward, metrics))

        return results

    def logging_tags(self) -> list[str]:
        """Return tags for logging/metric aggregation."""
        return [self.env_id]
