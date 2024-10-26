from jass.game.game_util import convert_one_hot_encoded_cards_to_int_encoded_list
from jass.game.game_observation import GameObservation
from jass.game.const import PUSH
from jass.game.rule_schieber import RuleSchieber
from jass.agents.agent import Agent
from dl4g.utils import calculate_trump_selection_score
import numpy as np


class MyAgent(Agent):
    def __init__(self):
        super().__init__()
        # we need a rule object to determine the valid cards
        self._rule = RuleSchieber()

    def action_trump(self, obs: GameObservation) -> int:
        """
        Determine trump action for the given observation
        Args:
            obs: the game observation, it must be in a state for trump selection

        Returns:
            selected trump as encoded in jass.game.const or jass.game.const.PUSH
        """
        scores = np.zeros(4)
        cards = convert_one_hot_encoded_cards_to_int_encoded_list(obs.hand)
        scores = [calculate_trump_selection_score(cards, color) for color in range(4)]
        if obs.forehand == 0:
            return np.argmax(scores)
        else:
            if max(scores) > 68:
                return np.argmax(scores)
            else:
                return PUSH

    def action_play_card(self, obs: GameObservation) -> int:
        """
        Determine the card to play.

        Args:
            obs: the game observation

        Returns:
            the card to play, int encoded as defined in jass.game.const
        """
        valid_cards = self._rule.get_valid_cards_from_obs(obs)

        # we use the global random number generator here
        return np.random.choice(np.flatnonzero(valid_cards))
