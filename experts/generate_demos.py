import stable_baselines3 as sb
import torch
import os
import argparse


from imitation_learning.utils.utils import collect_demo
from imitation_learning.utils.env import make_env


class SBPolicyWrapper:
    def __init__(self, model: sb.SAC, device):
        self.model = model
        self.device = device

    def exploit(self, state):
        with torch.no_grad():
            action = self.model.predict(state, deterministic=True)
        return action[0]


def run(args):
    env = make_env(args.env)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = sb.SAC.load(args.weights, device=device)
    model = SBPolicyWrapper(model, device=device)

    buffer, mean_return = collect_demo(
        env=env,
        algo=model,
        buffer_size=args.buffer_size,
        device=device,
        std=args.std,
        p_rand=args.p_rand,
        seed=args.seed,
    )
    buffer.save(
        os.path.join(
            "buffers",
            args.env,
            f"size{args.buffer_size}_std{args.std}_prand{args.p_rand}_return{round(mean_return, 1)}.pth",
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=str, required=True)
    p.add_argument("--env", type=str, default="InvertedPendulum-v5")
    p.add_argument("--buffer_size", type=int, default=10**6)
    p.add_argument("--std", type=float, default=0.0)
    p.add_argument("--p_rand", type=float, default=0.0)
    p.add_argument("--cuda", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    run(args)
