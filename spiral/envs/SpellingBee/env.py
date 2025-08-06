
"""
Spelling Bee Game Environment with Zero-Sum Try Limit System

This environment implements a competitive spelling bee game where players take turns
providing English words using only allowed letters.

GAME RULES:
1. Players alternate providing English words in square brackets: [word]
2. Each word must be at least as long as the previous word
3. Each word must use only the allowed letters
4. No word can be repeated during the game
5. All words must be valid English words

ZERO-SUM TRY LIMIT SYSTEM:
- Each player has a limited number of tries (default: 2) for invalid moves
- Invalid moves include: wrong format, too short, using forbidden letters,
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
"""

import re, random, string, numpy
from typing import Optional, Tuple, List, Dict, Any

import textarena as ta
from textarena.envs.SpellingBee.renderer import create_board_str
from textarena.utils.word_lists import EnglishDictionary



class SpellingBeeEnv(ta.Env):
    """ Environment for the Spelling Bee Game with increasing word length and try limit system. """
    def __init__(self, num_letters: int, max_tries: int = 2):
        """
        Initialize the Spelling Bee Game environment.

        Args:
            num_letters (int): Number of unique allowed letters.
            max_tries (int): Maximum number of invalid moves per player.
        """
        super().__init__()
        self.num_letters = num_letters
        self.max_tries = max_tries
        self.dictionary = EnglishDictionary(keep_proper_nouns=False, include_nltk=True)


    def get_board_str(self):
        return create_board_str(game_state=self.state.game_state)

    def _check_word(self, word: str) -> bool:
        return self.dictionary.is_english_word(word)

    def reset(self, num_players: int = 2, seed: Optional[int]=None):
        """ Reset the Spelling Bee game to its initial state. """
        # Initialize game state variables
        self.state = ta.State(num_players=num_players, min_players=2, max_players=2)

        game_state = {
            "allowed_letters": self._generate_allowed_letters(),
            "word_history": [],
            "tries_per_player": {0: 0, 1: 0},  # Track tries for each player
            "players_with_valid_moves": set()  # Track which players have made valid moves
        }
        self.state.reset(seed=seed, game_state=game_state, player_prompt_function=self._generate_player_prompt)

    def _generate_allowed_letters(self) -> set:
        """
        Generate a random set of unique lowercase letters with a frequency-weighted distribution.
        """
        if self.num_letters > 26:
            raise ValueError("num_letters cannot exceed 26.")

        # Frequency of letters in the English language (rough estimates)
        letter_frequencies = {
            'a': 8.17, 'b': 1.49, 'c': 2.78, 'd': 4.25, 'e': 12.70, 'f': 2.23,
            'g': 2.02, 'h': 6.09, 'i': 7.00, 'j': 0.15, 'k': 0.77, 'l': 4.03,
            'm': 2.41, 'n': 6.75, 'o': 7.51, 'p': 1.93, 'q': 0.10, 'r': 5.99,
            's': 6.33, 't': 9.06, 'u': 2.76, 'v': 0.98, 'w': 2.36, 'x': 0.15,
            'y': 1.97, 'z': 0.07
        }

        letters = list(letter_frequencies.keys())
        weights = list(letter_frequencies.values())

        # Convert weights to probabilities that sum to 1.
        total_weight = sum(weights)
        probs = [w / total_weight for w in weights]

        # Use numpy.random.choice to sample without replacement
        allowed = numpy.random.choice(letters, size=self.num_letters, replace=False, p=probs)
        return set(allowed)


    def _generate_player_prompt(self, player_id: int, game_state: Dict[int, Any]) -> str:
        """ Generate the initial prompt for a player. """
        tries_used = game_state['tries_per_player'][player_id]
        remaining_tries = self.max_tries - tries_used
        
        prompt = (
            f"You are Player {player_id} in the Spelling Bee Game.\n"
            f"Allowed Letters: {''.join(sorted(game_state['allowed_letters']))}\n"
            "Players take turns to provide valid English words that:\n"
            "1. Use only the allowed letters\n"
            "2. Must be at least as long as the previous word\n"
            "3. Cannot be a word that was previously used\n\n"
            f"Try Limit System: You have {remaining_tries} tries remaining (used {tries_used}/{self.max_tries}).\n"
            "- If you exceed max tries in the first round (before both players make valid moves): DRAW\n"
            "- If you exceed max tries after the first round: OPPONENT WINS\n\n"
            "Please wrap your word in square brackets, e.g., '[apple]', '[example]', etc.\n"
        )
        return prompt


    def step(self, action: str) -> Tuple[bool, ta.Info]:
        """ Process the player's action with try limit system logic """
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
            
            # Check if the word is shorter than the last word
            if len(self.state.game_state["word_history"]) != 0 and len(word) < len(self.state.game_state["word_history"][-1]):
                is_invalid = True
                reason = f"The word must be at least {len(self.state.game_state['word_history'][-1])} letters long. '{word}' has {len(word)} characters."
            # Check if the word has already been used
            elif word in self.state.game_state["word_history"]:
                is_invalid = True
                reason = f"The word '{word}' has already been used."
            # Check if the word is a valid English word
            elif not self._check_word(word):
                is_invalid = True
                reason = f"'{word}' is not a valid English word."
            # Check if the word uses only allowed letters
            elif not set(word).issubset(self.state.game_state["allowed_letters"]):
                is_invalid = True
                illegal_chars = set(word) - self.state.game_state["allowed_letters"]
                reason = f"The word '{word}' contains illegal characters: {', '.join(illegal_chars)}."
            # The move is valid: update the game state
            else:
                # Mark this player as having made a valid move
                self.state.game_state["players_with_valid_moves"].add(current_player)
                
                # Update game state
                self.state.game_state["word_history"].append(word)

                # Broadcast a message about the valid move
                message = (
                    f"Player {current_player} played: [{word}]\n"
                    f"Next word must:\n"
                    f"1. Use only allowed letters: {''.join(sorted(self.state.game_state['allowed_letters']))}\n"
                    f"2. Be at least {len(word)} letters long"
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
