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
