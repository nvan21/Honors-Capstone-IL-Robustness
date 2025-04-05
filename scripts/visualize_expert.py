from pathlib import Path
import argparse
import torch

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

    if "sac" in weights_split:
        algo = SACExpert(
            state_shape=env.observation_space.shape,
            action_shape=env.action_space.shape,
            device=torch.device("cuda" if args.cuda else "cpu"),
            path=args.weights,
        )
    elif "airl" in weights_split or "airl_ppo" in weights_split:
        algo = PPOExpert(
            state_shape=env.observation_space.shape,
            action_shape=env.action_space.shape,
            device=torch.device("cuda" if args.cuda else "cpu"),
            path=args.weights,
        )
    elif "bc" in weights_split or "dagger" in weights_split:
        algo = BCExpert(
            state_shape=env.observation_space.shape,
            action_shape=env.action_space.shape,
            device=torch.device("cuda" if args.cuda else "cpu"),
            path=args.weights,
        )

    visualize_expert(env, algo, args.seed, args.num_eval_episodes)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=str, required=True)
    p.add_argument("--env", type=str, default="InvertedPendulum-v5")
    p.add_argument("--cuda", action="store_true")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--num_eval_episodes", type=int, default=5)
    p.add_argument("--display", action="store_true")
    args = p.parse_args()
    run(args)
