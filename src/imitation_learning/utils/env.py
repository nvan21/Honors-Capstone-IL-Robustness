import gymnasium as gym
from gymnasium.wrappers import RescaleAction
import torch
from copy import copy


def make_env(env_id, **kwargs):
    env = gym.make(env_id, **kwargs)

    scale = env.action_space.high
    min_action = env.action_space.low / scale
    max_action = env.action_space.high / scale

    return RescaleAction(env, min_action=min_action, max_action=max_action)


def make_custom_reward_env(env, reward_model, device):

    return AIRLRewardWrapper(env, reward_model=reward_model, device=device)


class AIRLRewardWrapper(gym.Wrapper):
    def __init__(self, env, reward_model, device):
        super().__init__(env)

        self.reward_model = reward_model
        self.reward_model.eval()

        self.state = None
        self.device = device

    def reset(self, **kwargs):
        state, info = self.env.reset(**kwargs)
        self.state = state

        return state, info

    def step(self, action):
        # Take a step using the original environment
        next_state, _, done, truncated, info = self.env.step(action)

        # Get custom reward
        state_tensor = torch.tensor(self.state, dtype=torch.float, device=self.device)
        next_state_tensor = torch.tensor(
            copy(next_state), dtype=torch.float, device=self.device
        )
        reward = self.reward_model.f(state_tensor, done, next_state_tensor).item()

        # Update last state
        self.state = next_state

        return next_state, reward, done, truncated, info
