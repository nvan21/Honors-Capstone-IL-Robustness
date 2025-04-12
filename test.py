import os
import argparse
import torch

from imitation_learning.utils.env import make_env
from imitation_learning.utils.utils import visualize_expert
from imitation_learning.network import AIRLDiscrim


def run(args):
    env = make_env(args.env_id, render_mode="human")

    disc_file = "/work/flemingc/nvan21/projects/Honors-Capstone/logs/InvertedPendulum-v5/airl/normal_env-seed0-20250405-1912/model/step250000/disc.pth"
    disc = AIRLDiscrim(
        state_shape=env.observation_space.shape,
        hidden_units_r=(100, 100),
        hidden_units_v=(100, 100),
        gamma=0.99,
    )
    disc.load_state_dict(torch.load(disc_file))
    states = torch.randn((1, 4)).clamp(-0.2, 0.2)
    next_states = torch.randn((1, 4))
    dones = torch.zeros((1, 1))
    states = torch.tensor((-0.2000, 0.0313, -0.2000, -0.2000))
    next_states = torch.tensor((0.0, 0.0, 0.0, 0.0), dtype=torch.float)
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
