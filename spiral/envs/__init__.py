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

import logging

import textarena as ta
from textarena.envs.registration import register

# Kuhn Poker (two-player)
register(
    id="KuhnPoker-v1",
    entry_point="spiral.envs.KuhnPoker.env:KuhnPokerEnv",
    ante=1,
    max_rounds=5,
)

# Simple Negotiation (two-player)
register(
    id="SimpleNegotiation-v1",
    entry_point="spiral.envs.SimpleNegotiation.env:SimpleNegotiationEnv",
    max_turns=10,
)

# Liar's Dice (two-player)
register(
    id="LiarsDice-v1",
    entry_point="spiral.envs.LiarsDice.env:LiarsDiceEnv",
)

# Truth and Deception (two-player)
register(
    id="TruthAndDeception-v1",
    entry_point="spiral.envs.TruthAndDeception.env:TruthAndDeceptionEnv",
    max_turns=6,
)

# Pig Dice (two-player)
register(
    id="PigDice-v1",
    entry_point="spiral.envs.PigDice.env:PigDiceEnv",
    winning_score=50,
    max_turns=50,
)

# WordChains (two-player)
register(
    id="WordChains-v1",
    entry_point="spiral.envs.WordChains.env:WordChainsEnv",
    max_tries=2,
)

# SpellingBee (two-player)
register(
    id="SpellingBee-v1",
    entry_point="spiral.envs.SpellingBee.env:SpellingBeeEnv",
    num_letters=7,
)

# SimpleBlindAuction (two-player)
register(
    id="SimpleBlindAuction-v1",
    entry_point="spiral.envs.SimpleBlindAuction.env:SimpleBlindAuctionEnv",
)

def make_env(env_id: str, use_llm_obs_wrapper: bool):
    env = ta.make(env_id)
    if use_llm_obs_wrapper:
        logging.info(f"[ENV] Using LLMObservationWrapper for {env_id}")
        env = ta.wrappers.LLMObservationWrapper(env=env)
    else:
        logging.info(f"[ENV] Using FirstLastObservationWrapper for {env_id}")
        env = ta.wrappers.FirstLastObservationWrapper(env=env)
    return env


def make_vec_env(env_id: str, num_envs: int, **kw_args):
    return [make_env(env_id, **kw_args) for _ in range(num_envs)]
