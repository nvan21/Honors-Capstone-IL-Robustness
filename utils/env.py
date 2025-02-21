import gymnasium as gym
from gymnasium.wrappers import RescaleAction


def make_env(env_id, **kwargs):
    env = gym.make(env_id, **kwargs)

    scale = env.action_space.high
    min_action = env.action_space.low / scale
    max_action = env.action_space.high / scale

    return RescaleAction(env, min_action=min_action, max_action=max_action)
