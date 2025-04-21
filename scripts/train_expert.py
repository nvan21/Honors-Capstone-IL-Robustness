import os
import argparse
import torch
from datetime import datetime
import wandb

from imitation_learning.utils.env import (
    make_env,
    make_custom_reward_env,
    make_flattened_env,
)
from imitation_learning.algos import SAC
from imitation_learning.utils.trainer import Trainer
from imitation_learning.utils.utils import get_config, get_hidden_units_from_state_dict

from imitation_learning.network import AIRLDiscrim


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Expert Training Script")
    parser.add_argument(
        "--env", type=str, required=True, help="Environment name (e.g., Hopper-v5)"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Experiment name (e.g., small_network, high_lr)",
    )

    # Optional overrides
    parser.add_argument("--seed", type=int, help="Override config seed")

    return parser.parse_args()


def run_training():
    """Runs the training process"""
    args = parse_args()

    # Load configuration
    config = get_config("sac", args.env, args.experiment)

    # Add CUDA setting
    device = torch.device(
        "cuda" if config["cuda"] and torch.cuda.is_available() else "cpu"
    )

    # Initialize Weights & Biases
    run_name = f"sac-{args.env}"
    if args.experiment:
        run_name += f"-{args.experiment}"
    run_name += f"-seed{config['seed']}"

    writer = wandb.init(
        project="Honors Capstone",
        name=run_name,
        config=config,  # Log all hyperparameters
        group="sac",
        job_type=args.env,
        tags=["sac", args.env] + ([args.experiment] if args.experiment else []),
    )

    # Start by making it without the xml file to get around gymnasium_robotics env instantiation
    env = make_env(config["env_id"])
    env_test = make_env(config["env_id"])

    # Add observation flattening wrapper if it's a robotics env
    if isinstance(env.unwrapped, GoalEnv):
        env = make_flattened_env(env)
        env_test = make_flattened_env(env_test)
    else:
        env = make_env(config["env_id"], xml_file=config["xml_file"])
        env_test = make_env(config["env_id"], xml_file=config["xml_file"])

    # Get env shapes
    state_shape = env.observation_space.shape
    action_shape = env.action_space.shape

    # Alter environment to use AIRL reward if the experimental calls for it
    if config["use_reward_model"]:
        hidden_layers = get_hidden_units_from_state_dict(config["reward_model_path"])
        reward_model = AIRLDiscrim(
            state_shape=state_shape,
            gamma=config.get("reward_model_gamma", 0.99),
            hidden_units_r=hidden_layers["g"],
            hidden_units_v=hidden_layers["h"],
        ).to(device)
        reward_model.load_state_dict(torch.load(config["reward_model_path"]))
        env = make_custom_reward_env(
            env=env, reward_model=reward_model, device=device, normalize_reward=True
        )
        writer.config.update({"disc_hidden_layers": hidden_layers})

    # Log environment information
    writer.config.update(
        {
            "state_shape": state_shape,
            "action_shape": action_shape,
            "max_episode_steps": (
                env.spec.max_episode_steps
                if hasattr(env, "_max_episode_steps")
                else "unknown"
            ),
        }
    )
    # Instantiate algorithm
    algo = SAC(
        state_shape=state_shape,
        action_shape=action_shape,
        device=device,
        seed=config["seed"],
        gamma=config.get("discount_factor", 0.99),
        batch_size=config.get("batch_size", 256),
        buffer_size=config.get("buffer_size", 10**6),
        lr_actor=config.get("learning_rate", 3e-4),
        lr_critic=config.get("learning_rate", 3e-4),
        lr_alpha=config.get("alpha_lr", 3e-4),
        units_actor=config.get("hidden_sizes", (256, 256)),
        units_critic=config.get("hidden_sizes", (256, 256)),
        start_steps=config.get("start_steps", 10000),
        tau=config.get("tau", 5e-3),
        update_steps=config["update_steps"],
        use_reward_model=config.get("use_reward_model", False),
    )

    # Set log directory for model logging
    time_str = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(
        "logs",
        config["env_id"],
        "sac",
        f"{args.experiment or 'default'}-seed{config['seed']}-{time_str}",
    )

    trainer = Trainer(
        env=env,
        env_test=env_test,
        algo=algo,
        log_dir=log_dir,
        writer=writer,
        eval_interval=config["eval_interval"],
        seed=config["seed"],
    )
    trainer.online_train(num_steps=config["num_steps"])


if __name__ == "__main__":
    run_training()
