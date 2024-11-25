from jass.game.rule_schieber import RuleSchieber
from jass.game.game_util import convert_one_hot_encoded_cards_to_int_encoded_list
from jass.game.const import PUSH
from dl4g.utils import calculate_trump_selection_score
from jass.game.game_sim import GameSim
from jass.game.game_util import deal_random_hand
from jass.game.game_observation import GameObservation
import random
import torch
from dl4g.rl_ppo import PPOPolicy, encode_state, TrajectoryBuffer, ppo_update, reward_from_trick
from dl4g.agents.baseline_agent import BaselineAgent

import random
import numpy as np


def action_trump(obs: GameObservation) -> int:
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


# Initialize game, policy, optimizer, and buffer
state_dim = 84  # From your encode_state dimensions
action_dim = 36  # Assuming 36 possible actions
policy = PPOPolicy(state_dim, action_dim, hidden_dim=128)
optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
buffer = TrajectoryBuffer(state_dim, action_dim, max_steps=1000)
baseline_agent = BaselineAgent()

# Training loop
num_episodes = 10000
for episode in range(num_episodes):
    # Reset game and agent
    game = GameSim(rule=RuleSchieber())
    game.init_from_cards(hands=deal_random_hand(), dealer=random.randint(0, 3))

    trump = baseline_agent.action_trump(game.get_observation())
    game.action_trump(trump)
    done = False
    turn = 0
    while not game.is_done():

        obs = game.get_observation()

        if obs.player != 0:
            action = baseline_agent.action_play_card(obs)
            game.action_play_card(action)
            continue

        # Encode state and get action
        state = torch.tensor(encode_state(obs), dtype=torch.float32).unsqueeze(0)
        action_mask = game.rule.get_valid_cards_from_obs(obs)
        probs, value = policy(state, torch.tensor(action_mask))
        action = torch.multinomial(probs, 1).item()

        # Step game and get reward
        game.action_play_card(action)
        cards_played = game.get_observation().nr_cards_in_trick

        if cards_played != 0:
            for i in range(4 - cards_played):
                action = baseline_agent.action_play_card(obs)
                game.action_play_card(action)

        reward = reward_from_trick(game.get_observation(), turn)
        turn += 1

        # Add to buffer
        buffer.add(
            encode_state(obs),
            action,
            reward,
            torch.log(torch.tensor(probs[0, action]).float()),
            value.item(),
            1.0 if not game.is_done() else 0.0,
        )
    # Update policy after each episode
    ppo_update(policy, optimizer, buffer)
    buffer.reset()

    # Logging
    if episode % 100 == 0:
        print(f"Episode {episode}: Reward = {sum(buffer.rewards)}")

torch.save(policy.state_dict(), "ppo_model.pth")
