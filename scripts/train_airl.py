import os
import argparse
from datetime import datetime
import torch

from imitation_learning.utils.env import make_env
from imitation_learning.utils.buffer import SerializedBuffer
from imitation_learning.algos import AIRLPPO
from imitation_learning.utils.trainer import Trainer


def run(args):
    env = make_env(args.env_id)
    env_test = make_env(args.env_id)
    buffer_exp = SerializedBuffer(
        path=args.buffer, device=torch.device("cuda" if args.cuda else "cpu")
    )

    algo = AIRLPPO(
        buffer_exp=buffer_exp,
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        device=torch.device("cuda" if args.cuda else "cpu"),
        seed=args.seed,
        rollout_length=args.rollout_length,
        units_actor=(64, 64),
        units_critic=(64, 64),
        units_disc_r=(100, 100),
        units_disc_v=(100, 100),
    )

    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join("logs", args.env_id, "airl", f"seed{args.seed}-{time}")

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
    p.add_argument("--buffer", type=str, required=True)
    p.add_argument("--rollout_length", type=int, default=50000)
    p.add_argument("--num_steps", type=int, default=10**7)
    p.add_argument("--eval_interval", type=int, default=10**5)
    p.add_argument("--env_id", type=str, default="Hopper-v5")
    p.add_argument("--cuda", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    run(args)
