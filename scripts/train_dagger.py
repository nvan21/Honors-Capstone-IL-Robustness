import os
import argparse
from datetime import datetime
import torch

from imitation_learning.utils.env import make_env
from imitation_learning.utils.buffer import SerializedBuffer
from imitation_learning.utils.trainer import Trainer
from imitation_learning.algos import DAgger, SACExpert


def run(args):
    env = make_env(args.env_id)
    env_test = make_env(args.env_id)

    # Load expert buffer
    buffer_exp = SerializedBuffer(
        path=args.buffer, device=torch.device("cuda" if args.cuda else "cpu")
    )

    # Create SAC expert that generated the data
    expert = SACExpert(
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        device=torch.device("cuda" if args.cuda else "cpu"),
        path=args.expert_weights,
    )

    # Create DAgger algorithm
    algo = DAgger(
        expert=expert,
        buffer_exp=buffer_exp,
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        device=torch.device("cuda" if args.cuda else "cpu"),
        seed=args.seed,
        rollout_length=args.rollout_length,
        batch_size=args.batch_size,
        lr_actor=args.lr_actor,
        units_actor=eval(args.units_actor),  # Convert string to tuple
        beta=args.beta,
    )

    # Setup logging
    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(
        "logs", args.env_id, "DAgger", f"beta{args.beta}_seed{args.seed}-{time}"
    )

    # Create trainer and run training
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
    # Expert and buffer configuration
    p.add_argument(
        "--expert_weights",
        type=str,
        required=True,
        help="Path to SAC expert policy weights",
    )
    p.add_argument("--buffer", type=str, required=True, help="Path to expert buffer")

    # Training parameters
    p.add_argument(
        "--num_steps",
        type=int,
        default=10**5,
        help="Number of environment steps to train for",
    )
    p.add_argument(
        "--rollout_length",
        type=int,
        default=1000,
        help="Number of steps to collect before updating policy",
    )
    p.add_argument(
        "--eval_interval", type=int, default=5000, help="Steps between evaluations"
    )
    p.add_argument(
        "--beta",
        type=float,
        default=0.5,
        help="Probability of using expert actions during data collection (0 = pure DAgger, 1 = pure expert)",
    )

    # Environment
    p.add_argument("--env_id", type=str, default="Hopper-v5", help="Gym environment ID")

    # Model parameters
    p.add_argument("--batch_size", type=int, default=256, help="Batch size for updates")
    p.add_argument(
        "--lr_actor", type=float, default=3e-4, help="Learning rate for actor network"
    )
    p.add_argument(
        "--units_actor",
        type=str,
        default="(256, 256)",
        help="Hidden units in actor network (as string, e.g. '(256, 256)')",
    )

    # General
    p.add_argument("--cuda", action="store_true", help="Use CUDA if available")
    p.add_argument("--seed", type=int, default=0, help="Random seed")

    args = p.parse_args()
    run(args)
