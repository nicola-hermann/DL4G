import torch
import torch.nn as nn
from jass.game.const import MAX_PLAYER, MAX_TRUMP
from jass.game.game_observation import GameObservation

import numpy as np

MAX_POINTS = 157
NUM_TRICKS = 9
NUM_CARDS = 36


def encode_state(state: GameObservation):
    # Normalize scalars
    dealer = state.dealer / MAX_PLAYER
    player = state.player / MAX_PLAYER
    player_view = state.player_view / MAX_PLAYER
    trump = state.trump / MAX_TRUMP  # Normalize or set to 0 for no trump
    declared_trump = state.declared_trump / MAX_PLAYER
    nr_tricks = state.nr_tricks / NUM_TRICKS

    # Flatten tricks and current_trick
    tricks = state.tricks.flatten() / NUM_CARDS  # Normalize by card range
    current_trick = state.current_trick / NUM_CARDS  # Normalize by card range

    # Points
    points = state.points / MAX_POINTS  # Normalize by max points per team (example value)

    # Concatenate everything into a single state vector
    state_vector = np.concatenate(
        [
            [dealer, player, player_view, trump, declared_trump, nr_tricks],
            state.hand,  # Binary hand vector
            tricks,
            current_trick,
            points,
        ]
    )
    return state_vector


def masked_action_distribution(logits, action_mask):
    if action_mask is not None:
        # Set logits for invalid actions to a very negative value
        masked_logits = logits + (action_mask - 1) * 1e9
    else:
        masked_logits = logits
    return torch.nn.functional.softmax(masked_logits, dim=-1)


class PPOPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x, action_mask):
        shared_out = self.shared(x)
        logits = self.actor(shared_out)
        value = self.critic(shared_out)
        probs = masked_action_distribution(logits, action_mask)
        return probs, value


class TrajectoryBuffer:
    def __init__(self, state_dim, action_dim, max_steps):
        self.states = np.zeros((max_steps, state_dim), dtype=np.float32)
        self.actions = np.zeros(max_steps, dtype=np.int32)
        self.rewards = np.zeros(max_steps, dtype=np.float32)
        self.log_probs = np.zeros(max_steps, dtype=np.float32)
        self.values = np.zeros(max_steps, dtype=np.float32)
        self.masks = np.zeros(max_steps, dtype=np.float32)
        self.ptr = 0
        self.max_steps = max_steps

    def add(self, state, action, reward, log_prob, value, mask):
        idx = self.ptr
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.log_probs[idx] = log_prob
        self.values[idx] = value
        self.masks[idx] = mask
        self.ptr += 1

    def get(self):
        return (
            self.states[: self.ptr],
            self.actions[: self.ptr],
            self.rewards[: self.ptr],
            self.log_probs[: self.ptr],
            self.values[: self.ptr],
            self.masks[: self.ptr],
        )

    def reset(self):
        self.ptr = 0


def ppo_update(policy, optimizer, buffer, clip_epsilon=0.2, gamma=0.99, lambda_=0.95, epochs=4, batch_size=64):
    states, actions, rewards, old_log_probs, values, masks = buffer.get()
    advantages, returns = compute_gae(rewards, values, masks, gamma, lambda_)

    for _ in range(epochs):
        for start in range(0, len(states), batch_size):
            end = start + batch_size
            batch_states = torch.tensor(states[start:end], dtype=torch.float32)
            batch_actions = torch.tensor(actions[start:end], dtype=torch.int64)
            batch_log_probs = torch.tensor(old_log_probs[start:end], dtype=torch.float32)
            batch_returns = torch.tensor(returns[start:end], dtype=torch.float32)
            batch_advantages = torch.tensor(advantages[start:end], dtype=torch.float32)

            # Forward pass
            action_dist, value = policy(batch_states, None)  # No mask in update
            new_log_probs = torch.log(action_dist.gather(1, batch_actions.unsqueeze(-1)).squeeze(-1))

            # Policy Loss (Clipped Surrogate Objective)
            ratio = torch.exp(new_log_probs - batch_log_probs)
            clipped_ratio = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
            policy_loss = -torch.min(ratio * batch_advantages, clipped_ratio * batch_advantages).mean()

            # Value Loss
            value_loss = torch.nn.functional.mse_loss(value.squeeze(-1), batch_returns)

            # Entropy Bonus (encourage exploration)
            entropy_bonus = -torch.mean(action_dist * torch.log(action_dist + 1e-10))

            # Total Loss
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_bonus

            # Update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def compute_gae(rewards, values, masks, gamma, lambda_):
    advantages = np.zeros_like(rewards)
    returns = np.zeros_like(rewards)
    last_advantage = 0

    # Add a terminal value (0) for the last step (value at the end of the episode)
    values = np.append(values, 0)

    for t in reversed(range(len(rewards))):
        td_error = rewards[t] + gamma * values[t + 1] * masks[t] - values[t]
        advantages[t] = last_advantage = td_error + gamma * lambda_ * masks[t] * last_advantage
        returns[t] = advantages[t] + values[t]

    return advantages, returns


def reward_from_trick(obs: GameObservation, turn: int) -> int:
    """
    Calculate the reward from a trick
    Args:
        trick_winner: the winner of the trick
        trick_points: the points in the trick
    Returns:
        the reward
    """
    trick_winner = obs.trick_winner[turn]
    trick_points = obs.trick_points[turn]
    if trick_winner == 0 or trick_winner == 2:
        return trick_points
    else:
        return -trick_points
