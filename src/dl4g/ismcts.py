import numpy as np
import random
from jass.game.game_state import GameState
from jass.game.game_rule import GameRule
from copy import deepcopy
from jass.game.const import next_player
from jass.game.game_util import get_cards_encoded
from jass.game.game_sim import GameSim
import time


class Node:
    def __init__(
        self,
        state: GameState,
        rule: GameRule = None,
        parent=None,
        action=None,
    ):
        self.state = state
        self.rule = rule
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.value = 0
        self.unexplored_actions = self.get_legal_actions(self.state)
        self.simulation_results = []

    def get_legal_actions(self, state: GameState):
        """
        Get all legal actions for the current player.
        """
        return (
            list(np.flatnonzero(self.rule.get_valid_cards_from_state(state)))
            if state.player != -1
            else []
        )

    def is_fully_expanded(self):
        """
        Check if all actions have been expanded.
        """
        return not self.unexplored_actions

    def best_child(self, exploration_constant=1.0, alpha=1.0):
        """
        Select the best child according to the UCB formula.
        """
        best_value = -float("inf")
        best_child = None
        for child in self.children:
            # Average value (exploitation)
            avg_value = child.value / (child.visits + 1e-6)

            # Exploration term (standard UCB)
            exploration_term = exploration_constant * np.sqrt(
                np.log(self.visits + 1) / (child.visits + 1e-6)
            )

            # Bonus term (uncertainty penalty, stabalizes exploration)
            bonus_term = (
                alpha * np.std(child.simulation_results) / np.sqrt(child.visits + 1e-6)
            )

            # Compute final UCB value
            ucb_value = avg_value + exploration_term - bonus_term

            if ucb_value > best_value:
                best_value = ucb_value
                best_child = child
        return best_child


def sample_information_set(state: GameState):
    """
    Generate a complete game state consistent with the player's perspective.
    Replace -1 or masked information with plausible values.
    """
    # Example: fill unknown cards with random ones
    sampled_state = GameState()
    sampled_state.__dict__.update(state.__dict__)

    # Cards that have been played are known and should not be included in hands
    played_cards = set(sampled_state.tricks.flatten()) | set(
        sampled_state.current_trick
    )
    played_cards.discard(-1)

    hand = set(np.flatnonzero(sampled_state.hands[state.player]))

    all_cards = set(range(36))
    unseen_cards = list(all_cards - played_cards - hand)

    np.random.shuffle(unseen_cards)
    starting_player = sampled_state.trick_first_player[sampled_state.nr_tricks]

    if starting_player == -1:
        starting_player = sampled_state.player

    already_played = np.zeros(4, dtype=int)
    curr_player = starting_player

    for _ in range(state.nr_cards_in_trick):
        already_played[curr_player] = 1
        curr_player = next_player[curr_player]

    for i in range(4):
        if i != state.player:
            num_cards = 9 - sampled_state.nr_tricks - already_played[i]
            sampled_state.hands[i] = get_cards_encoded(unseen_cards[:num_cards])
            unseen_cards = unseen_cards[num_cards:]

    # flat non zero with -1
    all_cards_end = set(played_cards)
    for i in range(4):
        all_cards_end.update(np.flatnonzero(sampled_state.hands[i]))

    return sampled_state


def simulate_game(state: GameState, rule: GameRule):
    """
    Play out a random simulation to a terminal state.
    """
    if state.player != -1:
        game_sim = GameSim(rule=rule)
        game_sim.init_from_state(state)
        while not game_sim.is_done():
            obs = game_sim.get_observation()
            valid_actions = np.flatnonzero(game_sim.rule.get_valid_cards_from_obs(obs))
            action = valid_actions[random.randint(0, len(valid_actions) - 1)]
            game_sim.action_play_card(action)
        points_diff = game_sim.state.points[0] - game_sim.state.points[1]
    else:
        points_diff = state.points[0] - state.points[1]
    if state.player_view % 2 == 1:
        points_diff = -points_diff

    return points_diff / 157


def backpropagate(node, result):
    while node is not None:
        node.visits += 1
        node.value += result  # Aggregate total reward
        node.simulation_results.append(result)  # Store results for variance-based UCT
        node = node.parent


def play_action(state: GameState, rule: GameRule, action: int):
    game_sim = GameSim(rule=rule)
    game_sim.init_from_state(state)
    game_sim.action_play_card(action)
    # play random cards for the other players until its the current player's turn
    while game_sim.state.player != state.player and not game_sim.is_done():
        obs = game_sim.get_observation()
        valid_actions = np.flatnonzero(game_sim.rule.get_valid_cards_from_obs(obs))
        action = valid_actions[random.randint(0, len(valid_actions) - 1)]
        game_sim.action_play_card(action)

    return game_sim.state


def ismcts(root_state, rule, time_window=9, c=1.0):
    root_node = Node(root_state, rule=rule)

    # if root node only has one action, return it
    if len(root_node.unexplored_actions) == 1:
        return root_node.unexplored_actions[0], 0

    t1 = time.time()
    iterations = 0

    while time.time() - t1 < time_window:
        # Step 1: Selection
        current_node = root_node
        sampled_state = sample_information_set(current_node.state)

        while current_node.is_fully_expanded() and current_node.children:
            current_node = current_node.best_child(c)

        # Step 2: Expansion
        if not current_node.is_fully_expanded():
            action = current_node.unexplored_actions[-1]
            current_node.unexplored_actions = current_node.unexplored_actions[:-1]
            new_state = play_action(sampled_state, rule, action)
            child_node = Node(new_state, rule, parent=current_node, action=action)
            current_node.children.append(child_node)
            current_node = child_node

        # Step 3: Simulation
        resampled_state = sample_information_set(current_node.state)
        result = simulate_game(resampled_state, rule)

        # Step 4: Backpropagation
        backpropagate(current_node, result)
        iterations += 1

    # return the node with the most visits
    max_visits = -1
    for child in root_node.children:
        if child.visits > max_visits:
            max_visits = child.visits
            best_child = child

    return best_child.action, iterations
