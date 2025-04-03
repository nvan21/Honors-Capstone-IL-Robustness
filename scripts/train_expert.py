import os
import argparse
import torch
from datetime import datetime

from imitation_learning.utils.env import make_env
from imitation_learning.algos import SAC
from imitation_learning.utils.trainer import Trainer


def run(args):
    print(
        f"Attempting to train SAC expert on '{args.env_id}' with the following parameters: {args}"
    )
    env = make_env(args.env_id)
    env_test = make_env(args.env_id)

    algo = SAC(
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        device=torch.device("cuda" if args.cuda else "cpu"),
        seed=args.seed,
    )

    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(
        "logs", args.env_id, "expert", "sac", f"seed{args.seed}-{time}"
    )

    trainer = Trainer(
        env=env,
        env_test=env_test,
        algo=algo,
        log_dir=log_dir,
        eval_interval=args.eval_interval,
        seed=args.seed,
    )
    trainer.online_train(num_steps=args.num_steps)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--num_steps", type=int, default=10**6)
    p.add_argument("--eval_interval", type=int, default=10**4)
    p.add_argument("--env_id", type=str, default="Hopper-v5")
    p.add_argument("--cuda", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    run(args)
