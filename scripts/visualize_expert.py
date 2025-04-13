from pathlib import Path
import argparse
import torch
import json
import os
from datetime import datetime

from imitation_learning.utils.env import make_env
from imitation_learning.algos import SACExpert, PPOExpert, BCExpert
from imitation_learning.utils.utils import visualize_expert


def run(args):
    if args.display:
        env = make_env(
            args.env,
            render_mode="human",
        )
    else:
        env = make_env(args.env)

    weights_path = Path(args.weights)
    weights_split = [part.lower() for part in weights_path.parts]
    algo_name = None

    if "sac" in weights_split:
        algo_name = "sac"
        algo = SACExpert(
            state_shape=env.observation_space.shape,
            action_shape=env.action_space.shape,
            device=torch.device("cuda" if args.cuda else "cpu"),
            path=args.weights,
        )
    elif "airl" in weights_split or "airl_ppo" in weights_split:
        algo_name = "airl"
        algo = PPOExpert(
            state_shape=env.observation_space.shape,
            action_shape=env.action_space.shape,
            device=torch.device("cuda" if args.cuda else "cpu"),
            path=args.weights,
        )
    elif "bc" in weights_split:
        algo_name = "bc"
        algo = BCExpert(
            state_shape=env.observation_space.shape,
            action_shape=env.action_space.shape,
            device=torch.device("cuda" if args.cuda else "cpu"),
            path=args.weights,
        )
    elif "dagger" in weights_split:
        algo_name = "dagger"
        algo = BCExpert(
            state_shape=env.observation_space.shape,
            action_shape=env.action_space.shape,
            device=torch.device("cuda" if args.cuda else "cpu"),
            path=args.weights,
        )

    returns = visualize_expert(env, algo, args.seed, args.num_eval_episodes)
    time_str = datetime.now().strftime("%Y%m%d-%H%M")
    returns.update(
        {
            "algo": algo_name,
            "weights_path": args.weights,
            "time": time_str,
        }
    )

    if args.log:
        run_path = f"./runs/{args.env}/{algo_name}"
        os.makedirs(run_path, exist_ok=True)
        with open(os.path.join(run_path, f"results_{time_str}.json"), "w") as f:
            json.dump(returns, f, indent=4, sort_keys=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=str, required=True)
    p.add_argument("--env", type=str, default="InvertedPendulum-v5")
    p.add_argument("--cuda", action="store_true")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--num_eval_episodes", type=int, default=5)
    p.add_argument("--display", action="store_true")
    p.add_argument("--log", action="store_true")
    args = p.parse_args()
    run(args)
