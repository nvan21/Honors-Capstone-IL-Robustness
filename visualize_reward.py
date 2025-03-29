import os
import argparse
import torch

from utils.env import make_env
from utils.utils import visualize_expert
from network import AIRLDiscrim


def run(args):
    env = make_env(args.env_id, render_mode="human")

    disc_file = "./logs/InvertedPendulum-v5/airl_ppo/seed0-20250309-0841/model/step100000/disc.pth"
    disc = AIRLDiscrim(
        state_shape=env.observation_space.shape,
        hidden_units_r=(100, 100),
        hidden_units_v=(100, 100),
        gamma=0.995,
    )
    disc.load_state_dict(torch.load(disc_file))
    states = torch.randn((1, 4)).clamp(-0.2, 0.2)
    next_states = torch.randn((1, 4))
    dones = torch.zeros((1, 1))
    log_pis = torch.randn((1, 4))
    next_states = torch.tensor((0, 0, 0, 0), dtype=torch.float)
    print(f"Starting state: {states}")
    print(f"Next state: {next_states}")
    print(f"Reward: {disc.f(states, dones, next_states)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--env_id", type=str, default="InvertedPendulum-v5")
    p.add_argument("--cuda", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    run(args)
