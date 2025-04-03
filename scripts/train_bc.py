import os
import argparse
from datetime import datetime
import torch

from imitation_learning.utils.env import make_env
from imitation_learning.utils.buffer import SerializedBuffer
from imitation_learning.utils.trainer import Trainer
from imitation_learning.algos import BC


def run(args):
    env = make_env(args.env_id)
    env_test = make_env(args.env_id)
    buffer_exp = SerializedBuffer(
        path=args.buffer, device=torch.device("cuda" if args.cuda else "cpu")
    )

    algo = BC(
        buffer_exp=buffer_exp,
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        device=torch.device("cuda" if args.cuda else "cpu"),
        seed=args.seed,
    )

    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join("logs", args.env_id, "BC", f"seed{args.seed}-{time}")

    trainer = Trainer(
        env=env,
        env_test=env_test,
        algo=algo,
        log_dir=log_dir,
        eval_interval=args.eval_interval,
        seed=args.seed,
    )
    trainer.offline_train(num_epochs=args.num_epochs)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--buffer", type=str, required=True)
    p.add_argument("--num_epochs", type=int, default=10)
    p.add_argument("--eval_interval", type=int, default=1)
    p.add_argument("--env_id", type=str, default="Hopper-v5")
    p.add_argument("--cuda", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    run(args)
