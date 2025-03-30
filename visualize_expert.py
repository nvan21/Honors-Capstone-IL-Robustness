import os
import argparse
import torch

from utils.env import make_env
from algos import SACExpert, PPOExpert
from utils.utils import visualize_expert


def run(args):
    env = make_env(
        args.env_id,
        render_mode="human",
    )

    # algo = SACExpert(
    #     state_shape=env.observation_space.shape,
    #     action_shape=env.action_space.shape,
    #     device=torch.device("cuda" if args.cuda else "cpu"),
    #     path=args.weights,
    # )

    algo = PPOExpert(
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        device=torch.device("cuda" if args.cuda else "cpu"),
        path=args.weights,
    )

    visualize_expert(env, algo, args.seed)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=str, required=True)
    p.add_argument("--env_id", type=str, default="InvertedPendulum-v5")
    p.add_argument("--cuda", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    run(args)
