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
# MODEL_NAME = "Qwen/Qwen3-30B-A3B-Base"
MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"
RENDERER_NAME = "qwen3_instruct"
DATA_PATHS = ["data/aime", "data/amc", "data/olympiad_bench", "data/math", "data/minerva"]
BASE_URL = None

wandb.init(project="spiral", name=f"{MODEL_NAME.replace('/', '-')}-eval-math")

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


async def evaluate_data_path(
    data_path: str,
    sampling_client: tinker.SamplingClient,
    renderer: renderers.Renderer,
) -> tuple[str, int, int]:
    """Evaluate a single data path and return (path_name, correct, total)."""
    # Load dataset
    print(f"Loading dataset from {data_path}...")
    ds = load_from_disk(data_path)
    print(f"Loaded {len(ds)} problems from {data_path}")

    # Evaluate all problems in parallel
    print(f"Evaluating problems from {data_path}...")
    tasks = []
    for row in ds:
        problem = row.get("problem", "")
        answer = row.get("answer", "")
        if problem and answer:
            tasks.append(
                evaluate_single_problem(sampling_client, renderer, problem, answer)
            )

    # Run all evaluations in parallel with progress bar
    results = await tqdm.gather(*tasks, desc=f"Evaluating {data_path}")

    # Calculate accuracy
    correct = sum(results)
    total = len(results)
    
    return (data_path, correct, total)


async def main():
    # Configuration
    # model_name = "Qwen/Qwen3-8B"
    # model_name = "Qwen/Qwen3-4B-Instruct-2507"
    model_name = MODEL_NAME
    renderer_name = RENDERER_NAME
    data_paths = DATA_PATHS
    base_url = BASE_URL

    # Setup tokenizer and renderer
    tokenizer = get_tokenizer(model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer)

    # Create Tinker service client and sampling client
    print(f"Connecting to model {model_name}...")
    service_client = tinker.ServiceClient(base_url=base_url)
    sampling_client = service_client.create_sampling_client(base_model=model_name)

    # Evaluate all data paths in parallel
    print("Evaluating all data paths in parallel...")
    tasks = [
        evaluate_data_path(path, sampling_client, renderer)
        for path in data_paths
    ]
    results = await asyncio.gather(*tasks)

    # Aggregate results
    total_correct = 0
    total_problems = 0
    
    print("\n" + "="*50)
    print("Results by dataset:")
    print("="*50)
    for data_path, correct, total in results:
        accuracy = 100 * correct / total if total > 0 else 0
        print(f"{data_path}: {correct}/{total} ({accuracy:.2f}%)")
        total_correct += correct
        total_problems += total

    # Print final aggregated results
    print("\n" + "="*50)
    print(f"Final Results (Aggregated):")
    print(f"Total: {total_problems}")
    print(f"Correct: {total_correct}")
    print(f"Accuracy: {100*total_correct/total_problems:.2f}%")
    print("="*50)


if __name__ == "__main__":
    asyncio.run(main())
