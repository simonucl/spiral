#!/bin/bash
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
export TINKER_API_KEY=tml-3S8L6EnHoGBANUAh5TSH45uzICC9jnxp03VTHpJVKzBL5lzYoFsgfilJc3SEykRmAAAAA
export WANDB_API_KEY=99c1cfcf5ab402b2d7df6da383d1645fe6da06b6
export WEAVE_PRINT_CALL_LINK=false
export OPENROUTER_API_KEY=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

python train_spiral_tinker.py \
    model_name="openai/gpt-oss-20b" \
    renderer_name=gpt_oss_medium_reasoning \
    lora_rank=32 \
    env_ids='TicTacToe-v0,KuhnPoker-v1,SimpleNegotiation-v2' \
    use_llm_obs_wrappers='False,True,True' \
    batch_size=128 \
    num_train_datapoints=51200 \
    num_test_datapoints=128 \
    learning_rate=1e-4 \
    max_tokens=16384 \
    num_substeps=1 \
    filter_draw=False \
    use_role_baseline=True \
    role_baseline_ema_gamma=0.95 \
    eval_env_ids='TicTacToe-v0,KuhnPoker-v1,SimpleNegotiation-v2' \
    eval_use_llm_obs_wrappers='False,True,True' \
    eval_opponent_names='google/gemini-2.0-flash-001' \
    eval_every=16 \
    enable_math_test_eval=True \
    math_test_data_paths="data/aime,data/amc,data/olympiad_bench,data/math,data/minerva" \
    save_every=20 \
    loss_fn=importance_sampling \
    wandb_project=spiral \
    wandb_name=gpt-oss-20b-train \
    use_streaming=false
