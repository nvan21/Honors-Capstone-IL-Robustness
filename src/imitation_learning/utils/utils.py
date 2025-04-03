import torch
import torch.nn as nn
import math
import random
import numpy as np
import os
from copy import deepcopy
import yaml
from tqdm import tqdm

from imitation_learning.utils.buffer import Buffer


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


import torch
import re


def get_hidden_units_from_state_dict(state_dict_path):
    """
    Extract hidden unit sizes from a PyTorch state dictionary.

    Args:
        state_dict_path (str): Path to the saved state dictionary

    Returns:
        tuple: Hidden unit sizes (e.g., (256, 256))
    """
    try:
        # Load the state dictionary
        state_dict = torch.load(state_dict_path, map_location=torch.device("cpu"))

        # Find the first and second layer weight keys
        first_layer_key = None
        second_layer_key = None

        for key in state_dict:
            if re.search(r"net\.0\.weight$", key):
                first_layer_key = key
            elif re.search(r"net\.2\.weight$", key):
                second_layer_key = key

        if first_layer_key and second_layer_key:
            # Extract hidden sizes
            hidden_size1 = state_dict[first_layer_key].shape[0]
            hidden_size2 = state_dict[second_layer_key].shape[0]

            print(f"Found hidden sizes: {hidden_size1}, {hidden_size2}")
            return (hidden_size1, hidden_size2)
        else:
            print("Could not find expected layer weights in the state dictionary.")
            return None

    except Exception as e:
        print(f"Error loading or processing state dictionary: {e}")
        return None


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

    mean_return = total_return / num_episodes
    print(f"Mean return of the expert is {total_return / num_episodes}")

    return buffer, mean_return


def visualize_expert(env, algo, seeds=None, num_eval_episodes=5):
    if seeds is None:
        seeds = random.sample(range(2**31), num_eval_episodes)

    mean_return = 0
    with tqdm(total=num_eval_episodes) as pbar:
        for seed in seeds:
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

            state, _ = env.reset(seed=seed)
            episode_return = 0.0
            terminated = False
            truncated = False

            while not terminated and not truncated:
                action = algo.exploit(state)
                state, reward, terminated, truncated, _ = env.step(action)
                episode_return += reward

            mean_return += episode_return / num_eval_episodes

            # Update progress bar with latest episode return
            pbar.set_postfix(
                {
                    "seed": seed,
                    "return": f"{episode_return:.2f}",
                }
            )
            pbar.update(1)

    env.close()
    print(f"Mean Return: {mean_return:<5.1f}")


def load_yaml(file_path):
    """Load a YAML file."""
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


def get_config(algo_name, env_name, experiment_name=None):
    """
    Load configuration with the following priority (highest to lowest):
    1. Environment-specific experiment overrides (if in env_experiments)
    2. Named experiment overrides (if experiment_name provided)
    3. Environment-specific defaults (if in environments)
    4. Algorithm defaults

    Args:
        algo_name (str): Name of the algorithm (e.g., 'ppo', 'sac')
        env_name (str): Name of the environment (e.g., 'Hopper-v5')
        experiment_name (str, optional): Name of the experiment variant

    Returns:
        dict: Configuration dictionary
    """
    # Base path
    config_dir = "./hyperparameters"
    config_path = os.path.join(config_dir, f"{algo_name}.yaml")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    # Load the configuration
    config = load_yaml(config_path)

    # Start with algorithm defaults
    result = deepcopy(config["defaults"])

    # Add environment ID
    result["env_id"] = env_name

    # Apply environment-specific defaults if available
    if "environments" in config and env_name in config["environments"]:
        result.update(config["environments"][env_name])

    # Apply named experiment settings if provided
    if experiment_name is not None:
        # Check if it's a global named experiment
        if "experiments" in config and experiment_name in config["experiments"]:
            result.update(config["experiments"][experiment_name])

        # Check if it's an environment-specific experiment
        elif (
            "env_experiments" in config
            and env_name in config["env_experiments"]
            and experiment_name in config["env_experiments"][env_name]
        ):
            result.update(config["env_experiments"][env_name][experiment_name])
        else:
            raise ValueError(
                f"Experiment '{experiment_name}' not found for {algo_name} and {env_name}"
            )

    return result
