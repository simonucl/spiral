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

# User-specific (Change me) =========
export username=${USER:-spiral}

# Common =========
export LD_LIBRARY_PATH=$(python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))"):$LD_LIBRARY_PATH
export NCCL_CUMEM_ENABLE=0
export LP_DEBUG=1
export LP_LOG_LEVEL=DEBUG

# Check if OpenRouter API key is set (optional - only needed for external opponent evaluation)
if [ -z "$OPENROUTER_API_KEY" ]; then
    echo "Warning: OPENROUTER_API_KEY is not set"
    echo "External opponent evaluation will be disabled"
    echo "Set OPENROUTER_API_KEY if you want to evaluate against external models"
fi

# OctoThinker training with self-play configuration
python train_spiral.py \
    --use_role_baseline \
    --fixed_opponent google/gemini-2.0-flash-lite-001 \
    --env_ids KuhnPoker-v1 TicTacToe-v0 \
    --use_llm_obs_wrappers True False \
    --eval_env_ids TicTacToe-v0 KuhnPoker-v1 \
    --eval_use_llm_obs_wrappers False True \
    --eval_split all \
    --gamma 1 \
    --gpus 8 \
    --gradient-checkpointing \
    --num_samples 1 \
    --rollout_batch_size 128 \
    --dump_game_state_every 1 \
    --num_envs 1 \
    --rollout_batch_size_per_device 16 \
    --pi_buffer_maxlen_per_device 16 \
    --pretrain simonycl/octothinker-3b-hybrid-zero-cold-start-step-5 \
    --prompt_template octothinker \
    --enable_prefix_caching \
    --eval_prompt_template octothinker_general \
    --collocate \
    --vllm_sleep \
    --vllm_gpu_ratio 0.65 \
    --rnd-seed \
    --learning_rate 0.000001 \
    --lr_scheduler constant \
    --lr_warmup_ratio 0 \
    --num_ppo_epochs 2 \
    --train_batch_size 128 \
    --train_batch_size_per_device 1 \
    --beta 0 \
    --max_model_len 16384 \
    --generate_max_length 8192 \
    --max_context_length 32768 \
    --temperature 1.0 \
    --top_p 1 \
    --eval_steps 16 \
    --save_steps 32 \
    --eval_games 16 \
    --eval_temperature 0.6 \
    --eval_top_p 0.95 \
    --eval_generate_max_length 8192 \
    --max_train 65000 \
    --max_save_num 30 \
    --use-wb \
    --wb-run-name ${username}-spiral-octothinker-3b-multi-8k \
    --wb_project oat-self-play