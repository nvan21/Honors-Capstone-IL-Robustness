import torch
import torch.nn as nn
import math
import numpy as np
from tqdm import tqdm

from utils.buffer import Buffer


def build_mlp(
    input_dim,
    output_dim,
    hidden_units=[64, 64],
    hidden_activation=nn.Tanh(),
    output_activation=None,
):
    layers = []
    units = input_dim
    for next_units in hidden_units:
        layers.append(nn.Linear(units, next_units))
        layers.append(hidden_activation)
        units = next_units
    layers.append(nn.Linear(units, output_dim))
    if output_activation is not None:
        layers.append(output_activation)
    return nn.Sequential(*layers)


def reparameterize(means, log_stds):
    noises = torch.randn_like(means)
    us = means + noises * log_stds.exp()
    actions = torch.tanh(us)
    return actions, calculate_log_pi(log_stds, noises, actions)


def calculate_log_pi(log_stds, noises, actions):
    gaussian_log_probs = (-0.5 * noises.pow(2) - log_stds).sum(
        dim=-1, keepdim=True
    ) - 0.5 * math.log(2 * math.pi) * log_stds.size(-1)

    return gaussian_log_probs - torch.log(1 - actions.pow(2) + 1e-6).sum(
        dim=-1, keepdim=True
    )


def atanh(x):
    return 0.5 * (torch.log(1 + x + 1e-6) - torch.log(1 - x + 1e-6))


def evaluate_lop_pi(means, log_stds, actions):
    noises = (atanh(actions) - means) / (log_stds.exp() + 1e-8)
    return calculate_log_pi(log_stds, noises, actions)


def soft_update(target, source, tau):
    for t, s in zip(target.parameters(), source.parameters()):
        t.data.mul_(1.0 - tau)
        t.data.add_(tau * s.data)


def disable_gradient(network):
    for param in network.parameters():
        param.requires_grad = False


def add_random_noise(action, std):
    action += np.random.randn(*action.shape) * std
    return action.clip(-1.0, 1.0)


def collect_demo(env, algo, buffer_size, device, std, p_rand, seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    buffer = Buffer(
        buffer_size=buffer_size,
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        device=device,
    )

    total_return = 0.0
    num_episodes = 0

    state, _ = env.reset(seed=seed)
    t = 0
    episode_return = 0.0

    for _ in tqdm(range(1, buffer_size + 1)):
        t += 1

        if np.random.rand() < p_rand:
            action = env.action_space.sample()
        else:
            action = algo.exploit(state)
            action = add_random_noise(action, std)

        next_state, reward, terminated, truncated, _ = env.step(action)
        mask = False if truncated else terminated
        buffer.append(state, action, reward, mask, next_state)
        episode_return += reward

        if terminated or truncated:
            num_episodes += 1
            total_return += episode_return
            state, _ = env.reset()
            t = 0
            episode_return = 0.0

        state = next_state

    print(f"Mean return of the expert is {total_return / num_episodes}")
    return buffer


def visualize_expert(env, algo, seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    state, _ = env.reset(seed=seed)

    terminated, truncated = False, False

    while not terminated or truncated:
        action = algo.exploit(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        env.render()
        state = next_state
