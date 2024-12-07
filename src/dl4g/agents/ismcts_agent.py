from dl4g.agents.base_agent import BaseAgent, RandomAgent
from jass.game.game_util import convert_one_hot_encoded_cards_to_int_encoded_list
from jass.game.game_observation import GameObservation
from jass.game.const import PUSH, color_of_card, partner_player
from dl4g.utils import calculate_trump_selection_score, log_to_file, calc_current_winner
import numpy as np
from jass.game.game_state_util import state_from_observation
from dl4g.ismcts import ismcts


class ISMCTSAgent(BaseAgent):
    def __init__(self):
        super().__init__()

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
            return int(np.argmax(scores))
        else:
            if max(scores) > 68:
                return int(np.argmax(scores))
            else:
                return int(PUSH)

    def action_play_card(self, obs: GameObservation) -> int:
        """
        Determine the card to play.

        Args:
            obs: the game observation

        Returns:
            the card to play, int encoded as defined in jass.game.const
        """

        # Convert obs to state
        hand = obs.hand
        player = obs.player
        hands = np.full(shape=[4, 36], dtype=np.int32, fill_value=-1)
        hands[player] = hand

        state = state_from_observation(obs, hands)
        best_action = ismcts(state, self._rule, 1000, 1.0)
        breakpoint()
