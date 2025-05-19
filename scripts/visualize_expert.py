import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import stable_baselines3 as sb
import torch

from imitation_learning.algos import SBSAC, BCExpert, PPOExpert, SACExpert
from imitation_learning.utils.env import make_env, make_flattened_env
from imitation_learning.utils.utils import visualize_expert


def run(args):
    xml_file = os.path.join("./xml", args.env, args.xml_file)

    render_mode = "human" if args.display else None

    env = make_env(args.env, xml_file=xml_file, render_mode=render_mode)

    weights_path = Path(args.weights)
    weights_split = [part.lower() for part in weights_path.parts]
    is_sb_model = weights_path.suffix == ".zip"
    algo_name = None

    if "sac" in weights_split:
        algo_name = "sac"
        algo = SACExpert(
            state_shape=env.observation_space.shape,
            action_shape=env.action_space.shape,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            path=args.weights,
        )
    elif "airl" in weights_split or "airl_ppo" in weights_split:
        algo_name = "airl"
        algo = PPOExpert(
            state_shape=env.observation_space.shape,
            action_shape=env.action_space.shape,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            path=args.weights,
        )
    elif "gail" in weights_split:
        algo_name = "gail"
        algo = PPOExpert(
            state_shape=env.observation_space.shape,
            action_shape=env.action_space.shape,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            path=args.weights,
        )
    elif "bc" in weights_split:
        algo_name = "bc"
        algo = BCExpert(
            state_shape=env.observation_space.shape,
            action_shape=env.action_space.shape,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            path=args.weights,
        )
    elif "dagger" in weights_split:
        algo_name = "dagger"
        algo = BCExpert(
            state_shape=env.observation_space.shape,
            action_shape=env.action_space.shape,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            path=args.weights,
        )

    # Overwrite model if it's an SB3 model. This keeps the algo_name update from previously though
    if is_sb_model:
        algo_name = "sac"
        algo = SBSAC(weights=args.weights, env=env)

    if args.modified and algo_name == "sac":
        algo_name = "modified_sac"

    returns = visualize_expert(env, algo, args.seed, args.num_eval_episodes)
    time_str = datetime.now().strftime("%Y%m%d-%H%M")
    returns.update(
        {
            "algo": algo_name,
            "weights_path": args.weights,
            "time": time_str,
            "xml_file": xml_file,
            "env": args.env,
        }
    )

    if args.log:
        run_path = f"./runs/{args.env}_{Path(args.xml_file).stem}/{algo_name}"
        os.makedirs(run_path, exist_ok=True)
        with open(os.path.join(run_path, f"results.json"), "w") as f:
            json.dump(returns, f, indent=4, sort_keys=True)

        print(f"Logged to {run_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=str, required=True)
    p.add_argument("--env", type=str, default="InvertedPendulum-v5")
    p.add_argument("--xml_file", type=str, default="invpend.xml")
    p.add_argument("--cuda", action="store_true")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--num_eval_episodes", type=int, default=5)
    p.add_argument("--display", action="store_true")
    p.add_argument("--log", action="store_true")
    p.add_argument("--modified", action="store_true")
    args = p.parse_args()
    run(args)
