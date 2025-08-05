# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SPIRAL is a research framework for training language models through **multi-turn, zero-sum self-play games**. Models learn sophisticated reasoning by playing competitive games against continuously improving versions of themselves, eliminating the need for human supervision.

## Architecture

**Core Framework:** Actor-learner architecture using OAT (🌾 Oat) framework for distributed RL
- **SelfPlayActor**: Manages vectorized game environments, handles action extraction/validation
- **SelfPlayLearner**: PPO with custom evaluation, role-conditioned updates
- **Game Environments**: 5 two-player zero-sum games from TextArena (KuhnPoker, TicTacToe, SimpleNegotiation, LiarsDice, TruthAndDeception)

**Key Technologies:** Python 3.10, PyTorch, VLLM, OAT Framework, TextArena, Qwen3 models

## Development Commands

**Training:**
```bash
bash run.sh                    # Main training script
python train_spiral.py        # Direct training with custom args
```

**Key Parameters:**
- `--env_id KuhnPoker-v1` - Primary training environment
- `--pretrain Qwen/Qwen3-4B-Base` - Base model
- `--gpus 8` - Multi-GPU setup
- `--rollout_batch_size 128` - Game collection batch size

**Code Quality:**
```bash
make format        # Format core code only
make format_all    # Format all code including evals
make lint          # Run linting
make check-docstyle # Check documentation style
make checks        # Run all checks
make test          # Run pytest tests
make clean         # Remove build artifacts
```

**Evaluation:**
```bash
# Game evaluation (requires OPENROUTER_API_KEY)
bash evals/game/batch_run.sh

# Benchmark evaluation
cd evals/benchmarks && bash batch_run.sh
```

## Code Structure

**Core Components:**
- `spiral/components.py` - Core collectors and oracles
- `spiral/template.py` - Qwen3-specific prompt templates
- `spiral/agents/` - Agent implementations
- `spiral/envs/` - Game environment wrappers
- `train_spiral.py` - Main training entry point

**Evaluation Pipeline:**
- `evals/game/` - Game-based evaluations
- `evals/benchmarks/` - Math reasoning benchmarks (GSM8K, MATH, AIME)
- Three evaluation dimensions: in-domain games, out-of-domain transfer, general reasoning

**Environment Registration:** Custom games registered in `spiral/envs/__init__.py`

## Key Implementation Details

**Self-Play Training:** Models play both roles in zero-sum games, learning from competitive pressure
**Role-Conditioned PPO:** Separate advantage estimation for different game roles
**Hierarchical Evaluation:** Nested evaluation across games and benchmarks during training
**Action Parsing:** Custom LLM output parsing with validation and retry logic