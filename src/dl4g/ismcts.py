import numpy as np
from collections import defaultdict
from typing import List, Optional, Any, Dict


class JassState:
    def __init__(self, state):
        self.state = state
        self.current_player = 0
        self.game_over = False

    def get_legal_actions(self):
        return self.state.get_legal_actions()

    def move(self, action):
        new_state = self.state.move(action)
        new_jass_state = JassState(new_state)
        new_jass_state.current_player = 1 - self.current_player
        return new_jass_state

    def is_game_over(self):
        return self.state.is_game_over()

    def game_result(self):
        return self.state.game_result()

    def determinize(self):
        return self.state.determinize()


class InformationSetMonteCarloTreeSearchNode:
    def __init__(
        self,
        state: JassState,
        parent: Optional["InformationSetMonteCarloTreeSearchNode"] = None,
        parent_action: Optional[Any] = None,
    ):
        self.state = state
        self.parent = parent
        self.parent_action = parent_action
        self.children = []
        self._number_of_visits = 0
        self._results = defaultdict(int)
        self._results[1] = 0
        self._results[-1] = 0
        self._untried_actions = None
        self._untried_actions = self.untried_actions()

    def untried_actions(self) -> List:
        self._untried_actions = self.state.get_legal_actions()
        return self._untried_actions

    def q(self) -> int:
        wins = self._results[1]
        loses = self._results[-1]
        return wins - loses

    def n(self) -> int:
        return self._number_of_visits

    def expand(self) -> "InformationSetMonteCarloTreeSearchNode":
        action = self._untried_actions.pop()
        next_state = self.state.move(action)
        child_node = InformationSetMonteCarloTreeSearchNode(
            next_state, parent=self, parent_action=action
        )

        self.children.append(child_node)
        return child_node

    def is_terminal_node(self) -> bool:
        return self.state.is_game_over()

    def determinize(self) -> JassState:
        # Create a fully observable state from the current state
        return self.state.determinize()

    def rollout(self) -> int:
        current_rollout_state = self.determinize()

        while not current_rollout_state.is_game_over():

            possible_moves = current_rollout_state.get_legal_actions()

            action = self.rollout_policy(possible_moves)
            current_rollout_state = current_rollout_state.move(action)
        return current_rollout_state.game_result()

    def backpropagate(self, result: int) -> None:
        self._number_of_visits += 1.0
        self._results[result] += 1.0
        if self.parent:
            self.parent.backpropagate(result)

    def is_fully_expanded(self) -> bool:
        return len(self._untried_actions) == 0

    def best_child(
        self, c_param: float = 0.1
    ) -> "InformationSetMonteCarloTreeSearchNode":

        choices_weights = [
            (c.q() / c.n()) + c_param * np.sqrt((2 * np.log(self.n()) / c.n()))
            for c in self.children
        ]
        return self.children[np.argmax(choices_weights)]

    def rollout_policy(self, possible_moves: List) -> Any:
        return possible_moves[np.random.randint(len(possible_moves))]

    def _tree_policy(self) -> "InformationSetMonteCarloTreeSearchNode":
        current_node = self
        while not current_node.is_terminal_node():

            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.best_child()
        return current_node

    def best_action(self) -> "InformationSetMonteCarloTreeSearchNode":
        simulation_no = 100

        for i in range(simulation_no):

            v = self._tree_policy()
            reward = v.rollout()
            v.backpropagate(reward)

        return self.best_child(c_param=0.1)
