import gymnasium as gym
from gymnasium.wrappers import RescaleAction, NormalizeObservation
import torch
from copy import copy

from imitation_learning.utils.utils import disable_gradient


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
        disable_gradient(self.reward_model)

        self.device = device

    def calc_rewards(self, states, dones, next_states):
        with torch.no_grad():
            rewards = self.reward_model.f(states, dones, next_states)

        return rewards
