from dl4g.agents.base_agent import BaseAgent, RandomAgent
from jass.game.game_util import convert_one_hot_encoded_cards_to_int_encoded_list
from jass.game.game_observation import GameObservation
from jass.game.const import PUSH, partner_player, card_values, color_of_card
from dl4g.utils import calculate_trump_selection_score, log_to_file, calc_current_winner
import numpy as np


class HeuristicAgent(BaseAgent):
    def __init__(self):
        super().__init__()

    def action_trump(self, obs: GameObservation) -> int:

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

        all_cards = convert_one_hot_encoded_cards_to_int_encoded_list(obs.hand)
        valid_cards = np.flatnonzero(self._rule.get_valid_cards_from_obs(obs))
        trump = obs.trump

        if obs.nr_cards_in_trick == 0:
            colors = color_of_card[all_cards]
            max_color = np.argmax(np.bincount(colors))
            color_valid_cards = [
                card for card in valid_cards if color_of_card[card] == max_color
            ]
            if len(color_valid_cards) > 0:
                return np.random.choice(color_valid_cards)
            return np.random.choice(valid_cards)

        else:
            # Check who is currently winning the trick
            current_winner = calc_current_winner(
                obs.current_trick, obs.trick_first_player[obs.nr_tricks], obs.trump
            )

            teammate = partner_player[obs.player]

            # Split valid cards into winning and losing cards
            winning_cards = []
            losing_cards = []
            simulated_trick = obs.current_trick.copy()
            for card in valid_cards:
                simulated_trick[obs.nr_cards_in_trick] = card
                simulated_winner = calc_current_winner(
                    simulated_trick,
                    obs.trick_first_player[obs.nr_tricks],
                    obs.trump,
                )

                if simulated_winner == obs.player:
                    winning_cards.append(card)
                else:
                    losing_cards.append(card)

            if current_winner == teammate:
                if len(losing_cards) == 0:
                    played_card = np.random.choice(winning_cards)
                else:
                    played_card = np.random.choice(losing_cards)
            else:
                if len(winning_cards) == 0:
                    played_card = losing_cards[
                        np.argmin([card_values[trump][card] for card in losing_cards])
                    ]
                elif len(winning_cards) > 1:
                    if obs.nr_cards_in_trick == 3:
                        played_card = winning_cards[
                            np.argmax(
                                [card_values[trump][card] for card in winning_cards]
                            )
                        ]
                    else:
                        played_card = np.random.choice(winning_cards)
                else:
                    played_card = winning_cards[0]

        return played_card
