import numpy as np
import random
from collections import defaultdict
from jass.game.game_observation import GameObservation
from jass.game.game_state import GameState
from jass.game.game_rule import GameRule
from jass.game.game_state_util import observation_from_state, state_from_observation
from copy import deepcopy


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

    def get_legal_actions(self, state: GameState):
        valid_cards = np.flatnonzero(self.rule.get_valid_cards_from_state(state))
        return valid_cards

    def is_fully_expanded(self):
        return len(self.unexplored_actions) == 0

    def best_child(self, exploration_constant=1.0):
        # UCB: Select the best child using the UCB formula
        best_value = -float("inf")
        best_child = None
        for child in self.children:
            ucb_value = (
                child.value / (child.visits + 1e-6)
            ) + exploration_constant * np.sqrt(
                np.log(self.visits + 1) / (child.visits + 1e-6)
            )
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
    sampled_state = deepcopy(state)

    # Cards that have been played are known and should not be included in hands
    played_cards = set(np.where(state.tricks >= 0)[0])
    played_cards.update(np.where(state.current_trick >= 0)[0])

    all_cards = set(range(36))  # Assuming a deck of 36 cards
    unseen_cards = (
        all_cards - played_cards - set(np.flatnonzero(state.hands[state.player]))
    )

    # Randomly distribute unseen cards among other players
    unseen_cards = list(unseen_cards)
    np.random.shuffle(unseen_cards)
    for i in range(4):
        if i != state.player:
            num_cards = np.sum(state.hands[i] == -1)  # Count unknowns in this hand
            sampled_state.hands[i] = np.random.choice(
                unseen_cards, num_cards, replace=False
            )
            unseen_cards = unseen_cards[num_cards:]  # Remove assigned cards
    breakpoint()
    return sampled_state


def simulate_game(state):
    """
    Play out a random simulation to a terminal state.
    Return the result of the simulation (e.g., points or win/loss).
    """
    # Example random simulation
    return random.choice([-1, 1])  # Win or loss


def backpropagate(node, result):
    """
    Update node values up the tree.
    """
    while node is not None:
        node.visits += 1
        node.value += result
        node = node.parent


def ismcts(root_state, rule, iter_count=1000, c=1.0):
    root_node = Node(root_state, rule=rule)

    for _ in range(iter_count):
        # Step 1: Selection
        current_node = root_node
        while current_node.is_fully_expanded() and current_node.children:
            current_node = current_node.best_child(c)

        # Step 2: Expansion
        if not current_node.is_fully_expanded():
            action = current_node.unexplored_actions[-1]
            current_node.unexplored_actions = current_node.unexplored_actions[:-1]

            new_state = sample_information_set(current_node.state)
            # Update the state by applying the action
            # Example: new_state = apply_action(new_state, action)
            child_node = Node(new_state, rule, parent=current_node, action=action)
            current_node.children.append(child_node)
            current_node = child_node

        # Step 3: Simulation
        sampled_state = sample_information_set(current_node.state)
        result = simulate_game(sampled_state)

        # Step 4: Backpropagation
        backpropagate(current_node, result)

    # Select the best action
    best_child = root_node.best_child(exploration_constant=0)
    return best_child.action
