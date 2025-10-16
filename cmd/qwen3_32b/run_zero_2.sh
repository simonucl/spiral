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
export DS_SKIP_CUDA_CHECK=1

# OctoThinker training with self-play configuration
python train_spiral.py \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --use_role_baseline \
    --env_ids TicTacToe-v0 KuhnPoker-v1 SimpleNegotiation-v2 \
    --use_llm_obs_wrappers False True True \
    --eval_env_ids TicTacToe-v0 KuhnPoker-v1 SimpleNegotiation-v2 \
    --eval_use_llm_obs_wrappers False True True \
    --eval_split all \
    --gamma 1 \
    --gpus 8 \
    --num_samples 1 \
    --rollout_batch_size 128 \
    --dump_game_state_every 1 \
    --num_envs 1 \
    --rollout_batch_size_per_device 16 \
    --pi_buffer_maxlen_per_device 16 \
    --pretrain Qwen/Qwen3-32B \
    --prompt_template qwen3 \
    --enable_prefix_caching \
    --gradient-checkpointing \
    --eval_prompt_template qwen3_general \
    --collocate \
    --vllm_sleep \
    --vllm_gpu_ratio 0.65 \
    --num_gpus_per_actor 4 \
    --rnd-seed \
    --learning_rate 0.0001 \
    --lr_scheduler constant \
    --lr_warmup_ratio 0 \
    --num_ppo_epochs 2 \
    --train_batch_size 128 \
    --train_batch_size_per_device 1 \
    --beta 0 \
    --max_model_len 10000 \
    --generate_max_length 8192 \
    --max_context_length 16384 \
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
    --debug \
    --use-wb \
    --wb-run-name ${username}-spiral-qwen3-32b-multi-env \
    --wb_project oat-self-play