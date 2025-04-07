import os
import argparse
from datetime import datetime
import torch

from utils.env import make_env, make_custom_reward_env
from utils.trainer import Trainer
from algos import SAC
from network import AIRLDiscrim


def run(args):
    modified_xml = args.modified_xml
    reward_model_file = args.reward_model

    env = make_env(args.env_id, xml_file=modified_xml)
    env_test = make_env(args.env_id, xml_file=modified_xml)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reward_model = AIRLDiscrim(
        state_shape=env.observation_space.shape,
        hidden_units_r=(100, 100),
        hidden_units_v=(100, 100),
        gamma=0.995,
    ).to(device)
    reward_model.load_state_dict(torch.load(reward_model_file))

    env = make_custom_reward_env(env, reward_model=reward_model, device=device)
    env_test = make_custom_reward_env(
        env_test, reward_model=reward_model, device=device
    )

    algo = SAC(
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        device=torch.device("cuda" if args.cuda else "cpu"),
        seed=args.seed,
    )

    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(
        "logs", args.env_id, "sac", modified_xml, f"seed{args.seed}-{time}"
    )

    trainer = Trainer(
        env=env,
        env_test=env_test,
        algo=algo,
        log_dir=log_dir,
        num_steps=args.num_steps,
        eval_interval=args.eval_interval,
        seed=args.seed,
    )
    trainer.train()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--reward_model",
        type=str,
        default="./logs/InvertedPendulum-v5/airl_ppo/seed0-20250309-0841/model/step100000/disc.pth",
    )
    p.add_argument("--modified_xml", type=str, required=True)
    p.add_argument("--num_steps", type=int, default=10**5)
    p.add_argument("--eval_interval", type=int, default=10**3)
    p.add_argument("--env_id", type=str, default="InvertedPendulum-v5")
    p.add_argument("--cuda", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    run(args)
