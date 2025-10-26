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

"""Renderer for SPIRAL that adapts template system to Tinker's interface."""

import logging
from typing import Callable, Optional

import tinker
from tinker_cookbook.renderers import Message, Renderer, Role
from tinker_cookbook.tokenizer_utils import Tokenizer

from spiral.agents.utils import get_valid_action_parser
from spiral.template import TEMPLATE_FACTORY
from spiral.utils import extract_boxed_answer

logger = logging.getLogger(__name__)

INVALID_ACTION = "[｜INVALID_ACTION｜]"


class SpiralRenderer(Renderer):
    """
    Renderer for SPIRAL that uses the template system from spiral/template.py
    and validates actions against environment action spaces.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        template_name: str,
        env_id: str,
    ):
        """
        Initialize the SPIRAL renderer.

        Args:
            tokenizer: Tokenizer for encoding/decoding tokens
            template_name: Name of template from TEMPLATE_FACTORY
            env_id: Environment ID for action validation
        """
        super().__init__(tokenizer)
        self.template_name = template_name
        self.env_id = env_id

        # Get template function
        if template_name not in TEMPLATE_FACTORY:
            raise ValueError(
                f"Unknown template: {template_name}. "
                f"Available: {list(TEMPLATE_FACTORY.keys())}"
            )
        self.template_fn: Callable[[str, Optional[str]], str] = TEMPLATE_FACTORY[
            template_name
        ]

        # Get action parser for validation
        try:
            self.action_parser = get_valid_action_parser(env_id)
        except NotImplementedError:
            logger.warning(
                f"No action parser for {env_id}, will skip action validation"
            )
            self.action_parser = None

    def build_generation_prompt(
        self,
        messages: list[Message],
        role: Role = "assistant",
        prefill: str | None = None,
    ) -> tinker.ModelInput:
        """
        Build a generation prompt from messages.

        For SPIRAL, we expect messages to be in the format:
        [{"role": "user", "content": "<observation>"}]

        Args:
            messages: List of messages (should contain one user message with observation)
            role: Role to generate for (default: "assistant")
            prefill: Optional prefill text

        Returns:
            ModelInput ready for generation
        """
        # Extract the observation from the last user message
        if not messages:
            raise ValueError("Need at least one message")

        last_message = messages[-1]
        if last_message["role"] != "user":
            raise ValueError(f"Expected user message, got {last_message['role']}")

        observation = last_message["content"]

        # Apply template
        formatted_prompt = self.template_fn(observation, system_prompt=None)

        # Add prefill if provided
        if prefill is not None:
            formatted_prompt += prefill

        # Encode to tokens
        tokens = self.tokenizer.encode(formatted_prompt, add_special_tokens=True)

        return tinker.ModelInput(tokens=tokens)

    def get_stop_sequences(self) -> list[str] | list[int]:
        """
        Get stop sequences for generation.

        For TextArena games, we use "]\n" as the stop sequence since actions
        are enclosed in square brackets.

        Returns:
            List of stop sequences (as strings)
        """
        return ["]\n"]

    def parse_response(self, response: list[int]) -> tuple[Message, bool]:
        """
        Parse a response from the model into a Message and done flag.

        Extracts action from \\boxed{} and validates against action space.

        Args:
            response: List of token IDs from the model

        Returns:
            Tuple of (Message with extracted action, is_done flag)
        """
        # Decode tokens to text
        response_text = self.tokenizer.decode(response)

        # Extract action from \boxed{}
        extracted_action = extract_boxed_answer(response_text)

        # Validate action
        if extracted_action is None:
            # No boxed content found
            logger.warning(f"No \\boxed{{}} found in response: {response_text[:100]}")
            action_text = INVALID_ACTION
        else:
            # Validate against action space if parser is available
            action_text = self._validate_action(extracted_action, response_text)

        # Create message
        message: Message = {"role": "assistant", "content": action_text}

        # For SPIRAL, we're always done after one turn
        is_done = True

        return message, is_done

    def _validate_action(self, extracted_action: str, full_response: str) -> str:
        """
        Validate extracted action against environment action space.

        Args:
            extracted_action: Action extracted from \\boxed{}
            full_response: Full response text (contains observation for validation)

        Returns:
            Validated action string or INVALID_ACTION
        """
        # If no action parser, accept as-is
        if self.action_parser is None:
            return extracted_action

        # Special handling for different environment types
        if self.env_id in ["DontSayIt-v0", "SimpleNegotiation-v1"]:
            # These environments accept free-form chat, no validation needed
            return extracted_action

        elif self.env_id == "SimpleNegotiation-v2":
            # This uses regex patterns for validation
            import re

            patterns = self.action_parser(full_response)
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
                # Extract observation from response to get valid actions
                # This is a bit hacky - we need the observation to know valid actions
                # In practice, the observation is in the prompt, not the response
                # So we'll just accept the action if it looks valid
                valid_actions = self.action_parser(full_response)
                if extracted_action in valid_actions:
                    return extracted_action
                else:
                    logger.warning(
                        f"Action '{extracted_action}' not in valid actions for {self.env_id}"
                    )
                    return INVALID_ACTION
            except Exception as e:
                logger.warning(
                    f"Error validating action '{extracted_action}': {e}, accepting as-is"
                )
                return extracted_action

    def build_supervised_example(
        self, messages: list[Message], train_on_what: str = "last_assistant_message"
    ) -> tuple:
        """
        Build supervised training example.
        Not needed for RL, but required by Renderer interface.
        """
        raise NotImplementedError("SPIRAL uses RL, not supervised learning")


def get_spiral_renderer(
    model_name: str, template_name: str, env_id: str
) -> SpiralRenderer:
    """
    Helper function to create a SPIRAL renderer.

    Args:
        model_name: Model name for tokenizer
        template_name: Template name from TEMPLATE_FACTORY
        env_id: Environment ID

    Returns:
        Configured SpiralRenderer
    """
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    tokenizer = get_tokenizer(model_name)
    return SpiralRenderer(tokenizer, template_name, env_id)
