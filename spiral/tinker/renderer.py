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
import re
from typing import Callable, Optional

import weave
import tinker
from tinker_cookbook.renderers import Message, Renderer, Role
from tinker_cookbook.tokenizer_utils import Tokenizer

from spiral.template import TEMPLATE_FACTORY
from spiral.utils import extract_boxed_answer

logger = logging.getLogger(__name__)

INVALID_ACTION = "[｜INVALID_ACTION｜]"


class SpiralRenderer(Renderer):
    """
    Renderer for SPIRAL that uses the template system from spiral/template.py.

    This renderer is responsible for:
    - Formatting observations into prompts using templates
    - Parsing model responses to extract actions from \\boxed{}

    Action validation is handled by the Environment, not the Renderer.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        template_name: str,
    ):
        """
        Initialize the SPIRAL renderer.

        Args:
            tokenizer: Tokenizer for encoding/decoding tokens
            template_name: Name of template from TEMPLATE_FACTORY
        """
        super().__init__(tokenizer)
        self.template_name = template_name

        # Get template function
        if template_name not in TEMPLATE_FACTORY:
            raise ValueError(
                f"Unknown template: {template_name}. "
                f"Available: {list(TEMPLATE_FACTORY.keys())}"
            )
        self.template_fn: Callable[[str, Optional[str]], str] = TEMPLATE_FACTORY[
            template_name
        ]

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

        return tinker.ModelInput.from_ints(tokens)

    def get_stop_sequences(self) -> list[str] | list[int]:
        """
        Get stop sequences for generation.

        For TextArena games, we use "]\n" as the stop sequence since actions
        are enclosed in square brackets.

        Returns:
            List of stop sequences (as strings)
        """
        return []

    @weave.op()
    def parse_response(self, response: list[int]) -> tuple[Message, bool]:
        """
        Parse a response from the model into a Message and done flag.

        Extracts action from \\boxed{} notation. Does NOT validate the action
        against the game state - that's the Environment's responsibility.

        Args:
            response: List of token IDs from the model

        Returns:
            Tuple of (Message with extracted action, is_done flag)
        """
        # Decode tokens to text
        response_text = self.tokenizer.decode(response)

        # Extract action from \boxed{}
        extracted_action = extract_boxed_answer(response_text)

        if extracted_action is None:
            # No boxed content found
            logger.debug(f"No \\boxed{{}} found in response: {response_text[-100:]}")
            action_text = INVALID_ACTION
        else:
            # 1. Convert \boxed{} format to [content] format if found in the action
            formatted_action = re.sub(
                r"\\boxed\{([^}]*)\}",  # Match \boxed{...} capturing everything up to the matching }
                r"[\1]",  # Replace with brackets around the captured content
                extracted_action,
            )

            # 2. If there are no brackets but we should have them, add them
            if "[" not in formatted_action and "]" not in formatted_action:
                # Check if this is a short action that likely needs brackets
                words = formatted_action.split()
                if (
                    len(words) <= 5
                ):  # Heuristic for a short action that might need brackets
                    formatted_action = f"[{formatted_action}]"

            # 3. Additional cleaning to ensure valid formatting
            # Remove any extra newlines, tabs, or multiple spaces
            formatted_action = re.sub(r"\s+", " ", formatted_action).strip()

            action_text = formatted_action
        # Create message
        message: Message = {"role": "assistant", "content": action_text}

        # For SPIRAL, we're always done after one turn
        is_done = True

        return message, is_done

    def build_supervised_example(
        self, messages: list[Message], train_on_what: str = "last_assistant_message"
    ) -> tuple:
        """
        Build supervised training example.
        Not needed for RL, but required by Renderer interface.
        """
        raise NotImplementedError("SPIRAL uses RL, not supervised learning")


def get_spiral_renderer(model_name: str, template_name: str) -> SpiralRenderer:
    """
    Helper function to create a SPIRAL renderer.

    Args:
        model_name: Model name for tokenizer
        template_name: Template name from TEMPLATE_FACTORY

    Returns:
        Configured SpiralRenderer
    """
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    tokenizer = get_tokenizer(model_name)
    return SpiralRenderer(tokenizer, template_name)
