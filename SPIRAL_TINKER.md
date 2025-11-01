# SPIRAL Tinker TODO

Migration from OAT to Tinker framework - tracking completed work and development roadmap.

## Completed ✅

- [x] Environment integration with TextArena (async, two-player coordination)
- [x] Role-conditioned Advantage Estimation (RAE) with per-role EMA baselines
- [x] Draw filtering with configurable retry logic
- [x] Multi-environment training with rotation
- [x] Action validation with invalid action penalties (-1.5 / +0.5)
- [x] Custom training step with per-turn discounting (`train_step()`)
- [x] Comprehensive metrics tracking (per-player, per-trajectory, game outcomes)
- [x] Module organization (`spiral/tinker/` with utils, train_step, etc.)
- [x] Architecture fixes (validation in env, turn counting, metadata tracking)
- [x] **Evaluation code**: Online eval against random/LLM opponents (`GameEvaluator`)

## TODO

### High Priority 🔴

- [ ] **LogTree support**: Integrate visualization from https://github.com/thinking-machines-lab/tinker-cookbook/pull/39

### Medium Priority 🟡

- [ ] **Enhanced async rollout**: Parallel rollout across multiple games
- [ ] **Trajectory reuse (async RL)**: Off-policy training with importance sampling

### Experimental 🔬

- [ ] **OAT comparison**: Benchmark speed, GPU utilization, convergence, win rates
- [ ] **Population-based LoRA RL**: Multi-adapter training with cross-play evaluation

## Quick Reference

**Training with evaluation**:
```bash
python train_spiral_tinker.py \
    model_name="Qwen/Qwen3-8B-Base" \
    env_ids='["TicTacToe-v0", "KuhnPoker-v1"]' \
    use_llm_obs_wrappers='[false, true]' \
    eval_every=16 \
    eval_opponent_names="random,google/gemini-2.0-flash-exp"
```

**Key Files**:
- `spiral/tinker/train_step.py` - Custom training step with RAE
- `spiral/tinker/env.py` - Two-player environment
- `spiral/tinker/evaluator.py` - Game evaluation against opponents
- `spiral/tinker/utils.py` - Metrics computation
- `CUSTOM_TRAINING.md` - Detailed implementation docs
