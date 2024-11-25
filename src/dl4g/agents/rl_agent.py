from dl4g.agents.base_agent import BaseAgent, RandomAgent
from jass.game.game_util import convert_one_hot_encoded_cards_to_int_encoded_list
from jass.game.game_observation import GameObservation
from jass.game.const import PUSH, color_of_card, partner_player
from dl4g.utils import calculate_trump_selection_score, log_to_file, calc_current_winner
import numpy as np
import torch
from dl4g.rl_ppo import encode_state
from torch import nn


class RLAgent(BaseAgent):
    def __init__(self, model: nn.Module, weights: str):
        self.model = model
        self.model.load_state_dict(torch.load(weights))
        self.model.eval()

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
        state = torch.tensor(encode_state(obs), dtype=torch.float32).unsqueeze(0)
        action_mask = self._rule.get_valid_cards_from_obs(obs)
        probs, _ = self.model(state, torch.tensor(action_mask))
        action = torch.multinomial(probs, 1).item()

        return int(action)
