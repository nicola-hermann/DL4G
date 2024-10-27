import numpy as np

from jass.game.const import (
    color_of_card,
    lower_trump,
    UNE_UFE,
    OBE_ABE,
    card_strings,
    trump_strings_short,
)

# score if the color is trump
TRUMP_SCORE = [15, 10, 7, 25, 6, 19, 5, 5, 5]
# score if the color is not trump
NO_TRUMP_SCORE = [9, 7, 5, 2, 1, 0, 0, 0, 0]
# score if obenabe is selected (all colors)
OBENABE_SCORE = [14, 10, 8, 7, 5, 0, 5, 0, 0]
# score if uneufe is selected (all colors)
UNEUFE_SCORE = [0, 2, 1, 1, 5, 5, 7, 9, 11]


def calculate_trump_selection_score(cards: list, trump: int) -> int:
    if trump <= 3:
        trump_range_start = trump * 9
        trump_range_end = trump_range_start + 9

        score = 0
        for card in cards:
            if card >= trump_range_start and card < trump_range_end:
                score += TRUMP_SCORE[card - trump_range_start]
            else:
                score += NO_TRUMP_SCORE[card % 9]
        return score
    elif trump == 4:
        score = 0
        for card in cards:
            score += OBENABE_SCORE[card % 9]
        return score
    elif trump == 5:
        score = 0
        for card in cards:
            score += UNEUFE_SCORE[card % 9]
        return score
    else:
        raise ValueError("Trump must be in range 0-5")


def log_to_file(obs, played_card, all_cards, valid_cards, losing_cards, winning_cards):
    all_cards_str = [str(card_strings[card]) for card in all_cards]
    valid_cards_str = [str(card_strings[card]) for card in valid_cards]
    if len(losing_cards) == 0:
        losing_cards_str = ["_"]
    else:
        losing_cards_str = [str(card_strings[card]) for card in losing_cards]
    if len(winning_cards) == 0:
        winning_cards_str = ["_"]
    else:
        winning_cards_str = [str(card_strings[card]) for card in winning_cards]
    played_card_str = str(card_strings[played_card])
    current_trick_str = [
        str(card_strings[card]) if card != -1 else "_" for card in obs.current_trick
    ]
    with open("log.txt", "a") as f:
        f.write(
            f"Player: {str(obs.player)},Trump: {trump_strings_short[obs.trump]}, Cards: {all_cards_str}, Valid: {valid_cards_str}, Current Trick: {current_trick_str}, Played Card: {played_card_str}, Losing Cards: {losing_cards_str}, Winning Cards: {winning_cards_str}\n"
        )


def calc_current_winner(trick: np.ndarray, first_player: int, trump: int = -1) -> int:
    color_of_first_card = color_of_card[trick[0]]
    if trump == UNE_UFE:
        # lowest card of first color wins
        winner = 0
        lowest_card = trick[0]
        for i in range(1, 4):
            if trick[i] == -1:
                continue
            # (lower card values have a higher card index)
            if (
                color_of_card[trick[i]] == color_of_first_card
                and trick[i] > lowest_card
            ):
                lowest_card = trick[i]
                winner = i
    elif trump == OBE_ABE:
        # highest card of first color wins
        winner = 0
        highest_card = trick[0]
        for i in range(1, 4):
            if trick[i] == -1:
                continue
            if (
                color_of_card[trick[i]] == color_of_first_card
                and trick[i] < highest_card
            ):
                highest_card = trick[i]
                winner = i
    elif color_of_first_card == trump:
        # trump mode and first card is trump: highest trump wins
        winner = 0
        highest_card = trick[0]
        for i in range(1, 4):
            if trick[i] == -1:
                continue
            # lower_trump[i,j] checks if j is a lower trump than i
            if color_of_card[trick[i]] == trump and lower_trump[trick[i], highest_card]:
                highest_card = trick[i]
                winner = i
    else:
        # trump mode, but different color played on first move, so we have to check for higher cards until
        # a trump is played, and then for the highest trump
        winner = 0
        highest_card = trick[0]
        trump_played = False
        trump_card = None
        for i in range(1, 4):
            if trick[i] == -1:
                continue
            if color_of_card[trick[i]] == trump:
                if trump_played:
                    # second trump, check if it is higher
                    if lower_trump[trick[i], trump_card]:
                        winner = i
                        trump_card = trick[i]
                else:
                    # first trump played
                    trump_played = True
                    trump_card = trick[i]
                    winner = i
            elif trump_played:
                # color played is not trump, but trump has been played, so ignore this card
                pass
            elif color_of_card[trick[i]] == color_of_first_card:
                # trump has not been played and this is the same color as the first card played
                # so check if it is higher
                if trick[i] < highest_card:
                    highest_card = trick[i]
                    winner = i
    # adjust actual winner by first player
    return (first_player - winner) % 4
