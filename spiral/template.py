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

from typing import Optional


def apply_qwen3_template(observation: str, system_prompt: Optional[str] = None) -> str:
    del system_prompt
    return (
        f"<|im_start|>user\nYou are playing a two-player zero-sum game. Make valid actions to win.\nObservation: {observation}"
        "\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def apply_qwen3_general_template(
    question: str, system_prompt: Optional[str] = None
) -> str:
    del system_prompt
    return (
        f"<|im_start|>user\nQuestion: {question}"
        "\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def apply_octothinker_template(observation: str, system_prompt: Optional[str] = None) -> str:
    """OctoThinker template for game-based tasks."""
    del system_prompt
    return (
        f"A conversation between User and Assistant. The User presents the observation of a zero-sum game, and the Assistant makes a valid action in order to win. "
        f"The Assistant first thinks about the reasoning process in the mind and then provides the action. "
        f"User: You must put your answer inside \\boxed{{}} "
        f"and your final answer will be extracted automatically by the \\boxed{{}} tag.\n"
        f"Observation: {observation}\n"
        f"Assistant:"
    )


def apply_octothinker_general_template(observation: str, system_prompt: Optional[str] = None) -> str:
    """OctoThinker template for general reasoning tasks."""
    del system_prompt
    return (
        f"A conversation between User and Assistant. The user asks a question, and "
        f"the Assistant solves it. The assistant first thinks about the reasoning process in the mind and "
        f"then provides the user with the answer. User: You must put your answer inside \\boxed{{}} "
        f"and your final answer will be extracted automatically by the \\boxed{{}} tag.\n"
        f"Question: {observation}\n"
        f"Assistant:"
    )


def apply_octothinker_enforce_thinking_template(observation: str, system_prompt: Optional[str] = None) -> str:
    """OctoThinker template for games with enforced <think> tags."""
    del system_prompt
    return (
        f"A conversation between User and Assistant. The User presents the observation of a zero-sum game, and the Assistant makes a valid action in order to win. "
        f"The Assistant first thinks about the reasoning process in the mind and then provides the action. "
        f"User: The reasoning process must be enclosed within <think> </think> tags. "
        f"Then you must put your answer inside \\boxed{{}} "
        f"and your final answer will be extracted automatically by the \\boxed{{}} tag.\n"
        f"Observation: {observation}\n"
        f"Assistant: <think>"
    )


def apply_octothinker_enforce_thinking_general_template(observation: str, system_prompt: Optional[str] = None) -> str:
    """OctoThinker template for general reasoning with enforced <think> tags."""
    del system_prompt
    return (
        f"A conversation between User and Assistant. The user asks a question, and "
        f"the Assistant solves it. The assistant first thinks about the reasoning process in the mind and "
        f"then provides the user with the answer. User: The reasoning process must be enclosed within <think> </think> tags. "
        f"Then you must put your answer inside \\boxed{{}} "
        f"and your final answer will be extracted automatically by the \\boxed{{}} tag.\n"
        f"Question: {observation}\n"
        f"Assistant: <think>"
    )


TEMPLATE_FACTORY = {
    "qwen3": apply_qwen3_template,
    "qwen3_general": apply_qwen3_general_template,
    "octothinker": apply_octothinker_template,
    "octothinker_general": apply_octothinker_general_template,
    "octothinker_enforce_thinking": apply_octothinker_enforce_thinking_template,
    "octothinker_enforce_thinking_general": apply_octothinker_enforce_thinking_general_template,
}
