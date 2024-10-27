from jass.game.game_observation import GameObservation
from jass.game.rule_schieber import RuleSchieber
from jass.agents.agent import Agent
import numpy as np


class BaseAgent(Agent):
    def __init__(self):
        super().__init__()
        # we need a rule object to determine the valid cards
        self._rule = RuleSchieber()


class RandomAgent(BaseAgent):
    def __init__(self):
        super().__init__()

    def action_trump(self, obs: GameObservation) -> int:
        return np.random.choice([0, 1, 2, 3])

    def action_play_card(self, obs: GameObservation) -> int:
        valid_cards = self._rule.get_valid_cards_from_obs(obs)
        return np.random.choice(np.flatnonzero(valid_cards))
