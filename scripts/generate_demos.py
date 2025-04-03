import os
import argparse
import torch

from imitation_learning.utils.env import make_env
from imitation_learning.algos import SACExpert
from imitation_learning.utils.utils import collect_demo


def run(args):
    env = make_env(args.env_id)

    algo = SACExpert(
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        device=torch.device("cuda" if args.cuda else "cpu"),
        path=args.weights,
    )

    buffer, mean_return = collect_demo(
        env=env,
        algo=algo,
        buffer_size=args.buffer_size,
        device=torch.device("cuda" if args.cuda else "cpu"),
        std=args.std,
        p_rand=args.p_rand,
        seed=args.seed,
    )
    buffer.save(
        os.path.join(
            "buffers",
            args.env_id,
            f"size{args.buffer_size}_std{args.std}_prand{args.p_rand}_return{mean_return}.pth",
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=str, required=True)
    p.add_argument("--env_id", type=str, default="InvertedPendulum-v5")
    p.add_argument("--buffer_size", type=int, default=10**6)
    p.add_argument("--std", type=float, default=0.0)
    p.add_argument("--p_rand", type=float, default=0.0)
    p.add_argument("--cuda", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    run(args)
