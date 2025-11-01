#!/usr/bin/env python3
"""Minimal game evaluation script for Qwen/Qwen3-8B-Base."""

import asyncio

import tinker

from spiral.tinker.evaluator import GameEvaluator


async def main():
    # Configuration
    # model_name = "Qwen/Qwen3-8B-Base"
    model_name = "Qwen/Qwen3-8B"
    prompt_template = "qwen3"
    base_url = None  # Set to your Tinker service URL if needed

    # Game evaluation settings
    eval_env_ids = ["TicTacToe-v0", "KuhnPoker-v1", "SimpleNegotiation-v2"]
    eval_use_llm_obs_wrappers = [False, True, True]
    eval_opponent_names = ["random", "google/gemini-2.0-flash-001"]
    eval_games_per_matchup = 16
    max_tokens = 16384

    print(f"Evaluating {model_name} on games...")
    print(f"Environments: {eval_env_ids}")
    print(f"Opponents: {eval_opponent_names}")
    print(f"Games per matchup: {eval_games_per_matchup}")
    print()

    # Create evaluator
    evaluator = GameEvaluator(
        eval_env_ids=eval_env_ids,
        eval_opponent_names=eval_opponent_names,
        model_name=model_name,
        eval_use_llm_obs_wrappers=eval_use_llm_obs_wrappers,
        eval_games_per_matchup=eval_games_per_matchup,
        prompt_template=prompt_template,
        model_player_id=0,
    )

    # Create Tinker service client and sampling client
    service_client = tinker.ServiceClient(base_url=base_url)
    sampling_client = service_client.create_sampling_client(base_model=model_name)

    # Run evaluation (already runs games in parallel internally)
    metrics = await evaluator(sampling_client, max_tokens=max_tokens)

    # Print results
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)

    # Overall metrics
    print("\nOverall:")
    for key in ["win_rate", "loss_rate", "draw_rate", "model_invalid_rate", "avg_turns"]:
        metric_key = f"eval/overall/{key}"
        if metric_key in metrics:
            print(f"  {key}: {metrics[metric_key]:.3f}")

    # Per-environment metrics
    for env_id in eval_env_ids:
        print(f"\n{env_id}:")
        for key in ["win_rate", "loss_rate", "draw_rate", "model_invalid_rate", "avg_turns"]:
            metric_key = f"eval/{env_id}/{key}"
            if metric_key in metrics:
                print(f"  {key}: {metrics[metric_key]:.3f}")

        # Per-opponent breakdown for this environment
        for opponent_name in eval_opponent_names:
            matchup_prefix = f"eval/{env_id}/{opponent_name}"
            if f"{matchup_prefix}/win_rate" in metrics:
                print(f"  vs {opponent_name}:")
                print(f"    win_rate: {metrics[f'{matchup_prefix}/win_rate']:.3f}")
                print(f"    num_games: {metrics[f'{matchup_prefix}/num_games']}")

    print("\n" + "="*70)
    print(f"Total games: {metrics['eval/total_games']}")
    print(f"Evaluation time: {metrics['eval/game_eval_time']:.2f}s")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())
