"""
WordChains Game Environment with Zero-Sum Try Limit System

This environment implements a competitive word chains game where players take turns
providing English words that form a chain based on specific rules.

GAME RULES:
1. Players alternate providing English words in square brackets: [word]
2. Each word must start with the last letter of the previous word
3. Each word must be exactly one letter longer than the previous word
4. No word can be repeated during the game
5. Starting word is randomly selected (≤5 letters to keep games finite)

ZERO-SUM TRY LIMIT SYSTEM:
- Each player has a limited number of tries (default: 2) for invalid moves
- Invalid moves include: wrong format, wrong length, wrong starting letter, 
  non-English words, or previously used words

WIN/DRAW CONDITIONS:
1. No Valid Moves: If a player has no possible valid moves → opponent wins immediately
2. Exceeded Try Limit (First Round): If a player exceeds max_tries before both players 
   have made at least one valid move → DRAW (prevents unfair losses due to confusion)
3. Exceeded Try Limit (After First Round): If a player exceeds max_tries after both 
   players have made valid moves → OPPONENT WINS
4. Invalid Move Within Limits: Player gets feedback and loses one try

This system prevents:
- Infinite games waiting for invalid moves
- Exploitation by spamming invalid moves
- Unfair losses during learning phase
- Models learning to avoid playing by giving invalid format

EXAMPLE GAME FLOW:
Start: [cat] (3 letters)
Player 0: [turn] (4 letters, starts with 't') ✓
Player 1: [nurse] (5 letters, starts with 'n') ✓  
Player 0: [elephant] (8 letters, starts with 'e') ✗ (wrong length, should be 6)
Player 0: [earthy] (6 letters, starts with 'e') ✓
...continues until no valid moves exist or try limit exceeded
"""

import re, random 
from typing import Any, Dict, Optional, Tuple 

import nltk 
from nltk.corpus import words 
nltk.download("words")

import textarena as ta 
from textarena.envs.WordChains.renderer import create_board_str
from textarena.utils.word_lists import EnglishDictionary


class WordChainsEnv(ta.Env):
    """ Environment for playing the Word Chains game with increasing word length requirement and try limit system """
    
    def __init__(self, max_tries: int = 2):
        """  Initialize the Word Chains game environment """
        self.max_tries = max_tries
        
        # Ensure NLTK words are loaded
        self.word_list = list((set(word.lower() for word in words.words())))
        
        # only consider words shorter than 6 characters
        self.word_list = [word for word in self.word_list if len(word) <= 5]
        
        # Initialize dictionaries for US and UK English
        self.dictionary = EnglishDictionary(keep_proper_nouns=False, include_nltk=True)

    def get_board_str(self):
        return create_board_str(game_state=self.state.game_state)

    def reset(self, num_players: int, seed: Optional[int]=None):
        """ Reset the game to its initial state """
        self.state = ta.State(num_players=2, min_players=2, max_players=2)

        # Pick a starting word of minimum length
        starting_word = random.choice(self.word_list) 
        
        # Reset state
        game_state={
            "current_word": starting_word,
            "used_words": set([starting_word]),
            "required_start_letter": starting_word[-1].lower(),
            "required_length": len(starting_word) + 1,  # Next word must be one letter longer
            "tries_per_player": {0: 0, 1: 0},  # Track tries for each player
            "players_with_valid_moves": set()  # Track which players have made valid moves
        }
        self.state.reset(seed=seed, game_state=game_state, player_prompt_function=self._generate_player_prompt)

    def _generate_player_prompt(self, player_id: int, game_state: Dict[str, Any]) -> str:
        """ Generate the initial prompt for a player """
        tries_used = game_state['tries_per_player'][player_id]
        remaining_tries = self.max_tries - tries_used
        
        prompt = (
            f"You are Player {player_id} in the Word Chains Game.\n"
            "Players take turns to provide valid English words that:\n"
            "1. Start with the last letter of the previous word\n"
            "2. Must be exactly one letter longer than the previous word\n"
            "3. Cannot be a word that was previously used\n\n"
            f"Try Limit System: You have {remaining_tries} tries remaining (used {tries_used}/{self.max_tries}).\n"
            "- If you exceed max tries in the first round (before both players make valid moves): DRAW\n"
            "- If you exceed max tries after the first round: OPPONENT WINS\n\n"
            "Please wrap your word in square brackets, e.g., '[apple]', '[monkey]', etc.\n"
            f"The starting word is [{game_state['current_word']}].\n"
            f"Your word must start with '{game_state['required_start_letter']}' and "
            f"be exactly {game_state['required_length']} letters long.\n"
        )
        return prompt

    def step(self, action: str) -> Tuple[bool, ta.Info]:
        """ Process the player's move with try limit system logic """
        # Add action to log and observation
        self.state.add_observation(from_id=self.state.current_player_id, to_id=-1, message=action, for_logging=True)

        # Extract the word from the action
        word_match = re.search(r"\[(\w+)\]", action)
        is_invalid = False
        current_player = self.state.current_player_id
        
        if not word_match:
            # Invalid action format
            is_invalid = True
            reason = f"Player {current_player} did not provide a word in the valid format [word]."
        else:
            word = word_match.group(1).lower()
            required_length = self.state.game_state["required_length"]
            
            # Check if the word has the correct length
            if len(word) != required_length:
                is_invalid = True
                reason = f"The word must be exactly {required_length} letters long. '{word}' has {len(word)} characters."
            # Check if the word starts with the required letter
            elif not word.startswith(self.state.game_state["required_start_letter"]):
                is_invalid = True
                reason = f"The word must start with '{self.state.game_state['required_start_letter']}'."
            # Check if the word is a valid English word
            elif not self.dictionary.is_english_word(word):
                is_invalid = True
                reason = f"'{word}' is not a valid English word."
            # Check if the word has already been used
            elif word in self.state.game_state["used_words"]:
                is_invalid = True
                reason = f"The word '{word}' has already been used."
            # The move is valid: update the game state
            else:
                # Mark this player as having made a valid move
                self.state.game_state["players_with_valid_moves"].add(current_player)
                
                # Update game state
                self.state.game_state["used_words"].add(word)
                self.state.game_state["current_word"] = word
                self.state.game_state["required_start_letter"] = word[-1].lower()
                self.state.game_state["required_length"] = len(word) + 1

                # Broadcast a message about the valid move
                message = (
                    f"Player {current_player} played: [{word}]\n"
                    f"Next word must:\n"
                    f"1. Start with '{word[-1].lower()}'\n"
                    f"2. Be exactly {len(word) + 1} letters long"
                )
                self.state.add_observation(from_id=ta.GAME_ID, to_id=-1, message=message)

        # Handle invalid moves with try limit system
        if is_invalid:
            # Increment try count for current player
            self.state.game_state["tries_per_player"][current_player] += 1
            tries = self.state.game_state["tries_per_player"][current_player]
            
            # Check if player exceeded max tries
            if tries > self.max_tries:
                # Determine if first round is complete (both players made valid moves)
                first_round_complete = len(self.state.game_state["players_with_valid_moves"]) == 2
                
                if first_round_complete:
                    # After first round: opponent wins
                    opponent = 1 - current_player
                    self.state.set_winners(
                        player_ids=[opponent], 
                        reason=f"Player {current_player} exceeded {self.max_tries} tries. Player {opponent} wins!"
                    )
                else:
                    # During first round: draw
                    self.state.set_draw(
                        reason=f"Player {current_player} exceeded {self.max_tries} tries during first round. Game ends in draw."
                    )
            else:
                # Player still has tries remaining
                remaining_tries = self.max_tries - tries
                reason_with_tries = f"{reason} Try {tries}/{self.max_tries}. {remaining_tries} tries remaining."
                self.state.set_invalid_move(player_id=current_player, reason=reason_with_tries)

        return self.state.step()