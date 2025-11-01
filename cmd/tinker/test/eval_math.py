#!/usr/bin/env python3
"""Minimal math evaluation script for Qwen/Qwen3-8B-Base."""

import asyncio
import os

import tinker
from datasets import load_from_disk
from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.recipes.math_rl.math_grading import extract_boxed


async def main():
    # Configuration
    model_name = "Qwen/Qwen3-8B-Base"
    renderer_name = "qwen3_instruct"
    data_path = "data/aime"  # Change to your data path

    # Load dataset
    print(f"Loading dataset from {data_path}...")
    ds = load_from_disk(data_path)
    print(f"Loaded {len(ds)} problems")

    # Setup tokenizer and renderer
    tokenizer = get_tokenizer(model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer)

    # Create Tinker client
    print(f"Connecting to model {model_name}...")
    async with tinker.SamplingClient.from_model(model_name) as client:

        # Evaluate each problem
        correct = 0
        total = 0

        for i, row in enumerate(ds):
            problem = row.get("problem", "")
            answer = row.get("answer", "")

            if not (problem and answer):
                continue

            # Format prompt
            question = problem + " Please reason step by step, and put your final answer within \\boxed{}."
            messages = [{"role": "user", "content": question}]
            model_input = renderer.build_generation_prompt(messages)

            # Generate response
            response = await client.sample_async(
                model_input,
                num_samples=1,
                sampling_params=tinker.SamplingParams(
                    temperature=0.0,
                    max_tokens=2048,
                ),
            )

            # Parse response
            parsed_message, _ = renderer.parse_response(response.sequences[0].tokens)
            model_answer = parsed_message["content"]

            # Grade answer
            extracted_answer = extract_boxed(model_answer)
            correct_answer = str(answer) if isinstance(answer, (int, float)) else answer

            is_correct = extracted_answer == correct_answer
            if is_correct:
                correct += 1
            total += 1

            print(f"[{i+1}/{len(ds)}] Correct: {correct}/{total} ({100*correct/total:.1f}%)")

        # Print final results
        print("\n" + "="*50)
        print(f"Final Results:")
        print(f"Total: {total}")
        print(f"Correct: {correct}")
        print(f"Accuracy: {100*correct/total:.2f}%")
        print("="*50)


if __name__ == "__main__":
    asyncio.run(main())
