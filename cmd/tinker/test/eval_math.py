#!/usr/bin/env python3
"""Minimal math evaluation script for Qwen/Qwen3-8B-Base."""

import asyncio

import tinker
from datasets import load_from_disk
from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.recipes.math_rl.math_grading import extract_boxed
from tqdm.asyncio import tqdm
import wandb

wandb.init(project="spiral", name="qwen3-8b-math-eval")

async def evaluate_single_problem(
    sampling_client: tinker.SamplingClient,
    renderer: renderers.Renderer,
    problem: str,
    answer: str,
) -> bool:
    """Evaluate a single math problem asynchronously."""
    # Format prompt
    question = problem + " Please reason step by step, and put your final answer within \\boxed{}."
    messages = [{"role": "user", "content": question}]
    model_input = renderer.build_generation_prompt(messages)

    # Generate response
    response = await sampling_client.sample_async(
        model_input,
        num_samples=1,
        sampling_params=tinker.SamplingParams(
            temperature=0.0,
            max_tokens=16384,
        ),
    )

    # Parse response
    parsed_message, _ = renderer.parse_response(response.sequences[0].tokens)
    model_answer = parsed_message["content"]

    # Grade answer
    try:
        extracted_answer = extract_boxed(model_answer)
    except ValueError:
        return False
    correct_answer = str(answer) if isinstance(answer, (int, float)) else answer

    return extracted_answer == correct_answer


async def main():
    # Configuration
    model_name = "Qwen/Qwen3-8B"
    renderer_name = "qwen3"
    data_path = "data/aime"  # Change to your data path
    base_url = None  # Set to your Tinker service URL if needed

    # Load dataset
    print(f"Loading dataset from {data_path}...")
    ds = load_from_disk(data_path)
    print(f"Loaded {len(ds)} problems")

    # Setup tokenizer and renderer
    tokenizer = get_tokenizer(model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer)

    # Create Tinker service client and sampling client
    print(f"Connecting to model {model_name}...")
    service_client = tinker.ServiceClient(base_url=base_url)
    sampling_client = service_client.create_sampling_client(base_model=model_name)

    # Evaluate all problems in parallel
    print("Evaluating problems...")
    tasks = []
    for row in ds:
        problem = row.get("problem", "")
        answer = row.get("answer", "")
        if problem and answer:
            tasks.append(
                evaluate_single_problem(sampling_client, renderer, problem, answer)
            )

    # Run all evaluations in parallel with progress bar
    results = await tqdm.gather(*tasks, desc="Evaluating")

    # Calculate accuracy
    correct = sum(results)
    total = len(results)

    # Print final results
    print("\n" + "="*50)
    print(f"Final Results:")
    print(f"Total: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {100*correct/total:.2f}%")
    print("="*50)


if __name__ == "__main__":
    asyncio.run(main())
