import os
import torch
from datetime import datetime
import argparse
import wandb
from pathlib import Path

from imitation_learning.utils.env import make_env
from imitation_learning.utils.utils import get_config
from imitation_learning.utils.buffer import SerializedBuffer
from imitation_learning.utils.trainer import Trainer
from imitation_learning.algos import SAC, PPO, BC, AIRLPPO, GAIL, DAgger, SACExpert


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="RL/IL Training Script")
    parser.add_argument(
        "--algo", type=str, required=True, help="Algorithm name (e.g., ppo, sac, bc)"
    )
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
    parser.add_argument("--seed", type=int, help="Override random seed")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA")

    return parser.parse_args()


def get_algorithm(algo_name, config, env, buffer_exp=None, expert=None):
    """Initialize the appropriate algorithm based on name and config."""
    # Common parameters
    state_shape = env.observation_space.shape
    action_shape = env.action_space.shape
    device = torch.device("cuda" if config["cuda"] else "cpu")
    seed = config["seed"]

    # Algorithm-specific initialization
    if algo_name == "ppo":
        return PPO(
            state_shape=state_shape,
            action_shape=action_shape,
            device=device,
            seed=seed,
            gamma=config.get("discount_factor", 0.99),
            rollout_length=config.get("rollout_length", 2048),
            mix_buffer=config.get("mix_buffer", 20),
            lr_actor=config.get("learning_rate", 3e-4),
            lr_critic=config.get("learning_rate", 3e-4),
            units_actor=config.get("hidden_sizes", (256, 256)),
            units_critic=config.get("hidden_sizes", (256, 256)),
            epoch_ppo=config.get("epoch_ppo", 10),
            clip_eps=config.get("clip_ratio", 0.2),
            lambd=config.get("gae_lambda", 0.97),
            coef_ent=config.get("coef_ent", 0.0),
            max_grad_norm=config.get("max_grad_norm", 10.0),
        )
    elif algo_name == "sac":
        return SAC(
            state_shape=state_shape,
            action_shape=action_shape,
            device=device,
            seed=seed,
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
        )
    elif algo_name == "bc":
        if buffer_exp is None:
            raise ValueError("BC requires expert buffer")
        return BC(
            buffer_exp=buffer_exp,
            state_shape=state_shape,
            action_shape=action_shape,
            device=device,
            seed=seed,
            units_actor=config.get("hidden_sizes", (256, 256)),
            lr_actor=config.get("learning_rate", 3e-4),
            batch_size=config.get("batch_size", 128),
        )
    elif algo_name == "dagger":
        if buffer_exp is None:
            raise ValueError("DAgger requires expert buffer")
        if expert is None:
            raise ValueError("DAgger requires expert weights")
        return DAgger(
            expert=expert,
            buffer_exp=buffer_exp,
            state_shape=state_shape,
            action_shape=action_shape,
            device=device,
            seed=seed,
            gamma=config.get("gamma", 0.99),
            units_actor=config.get("hidden_sizes", (256, 256)),
            lr_actor=config.get("learning_rate", 3e-4),
            batch_size=config.get("batch_size", 128),
            beta=config.get("beta", 0),
            rollout_length=config.get("rollout_length", 1000),
        )
    elif algo_name == "airl":
        if buffer_exp is None:
            raise ValueError("AIRL requires expert buffer")
        return AIRLPPO(
            buffer_exp=buffer_exp,
            state_shape=state_shape,
            action_shape=action_shape,
            device=device,
            seed=seed,
            gamma=config.get("discount_factor", 0.99),
            rollout_length=config.get("rollout_length", 10000),
            mix_buffer=config.get("mix_buffer", 20),
            batch_size=config.get("batch_size", 64),
            lr_actor=config.get("learning_rate", 3e-4),
            lr_critic=config.get("learning_rate", 3e-4),
            lr_disc=config.get("disc_lr", 3e-4),
            units_actor=config.get("hidden_sizes", (256, 256)),
            units_critic=config.get("hidden_sizes", (256, 256)),
            units_disc_r=config.get("disc_hidden_sizes", (100, 100)),
            units_disc_v=config.get("disc_hidden_sizes", (100, 100)),
            epoch_ppo=config.get("epoch_ppo", 50),
            epoch_disc=config.get("epoch_disc", 10),
            clip_eps=config.get("clip_ratio", 0.2),
            lambd=config.get("gae_lambda", 0.97),
            coef_ent=config.get("coef_ent", 0.0),
            max_grad_norm=config.get("max_grad_norm", 10.0),
        )
    elif algo_name == "gail":
        if buffer_exp is None:
            raise ValueError("GAIL requires expert buffer")
        return GAIL(
            buffer_exp=buffer_exp,
            state_shape=state_shape,
            action_shape=action_shape,
            device=device,
            seed=seed,
            gamma=config.get("discount_factor", 0.99),
            rollout_length=config.get("rollout_length", 50000),
            mix_buffer=config.get("mix_buffer", 1),
            batch_size=config.get("batch_size", 64),
            lr_actor=config.get("learning_rate", 3e-4),
            lr_critic=config.get("learning_rate", 3e-4),
            lr_disc=config.get("disc_lr", 3e-4),
            units_actor=config.get("hidden_sizes", (64, 64)),
            units_critic=config.get("hidden_sizes", (64, 64)),
            units_disc=config.get("disc_hidden_sizes", (100, 100)),
            epoch_ppo=config.get("epoch_ppo", 50),
            epoch_disc=config.get("epoch_disc", 10),
            clip_eps=config.get("clip_ratio", 0.2),
            lambd=config.get("gae_lambda", 0.97),
            coef_ent=config.get("coef_ent", 0.0),
            max_grad_norm=config.get("max_grad_norm", 10.0),
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")


def run_training():
    """Run the training process."""
    args = parse_args()

    # Load algorithm name
    algo_name = args.algo.lower()

    # Load configuration
    config = get_config(algo_name, args.env, args.experiment)

    # Override with command line arguments
    if args.seed is not None:
        config["seed"] = args.seed

    # Add CUDA setting
    config["cuda"] = args.cuda
    device = torch.device(
        "cuda" if config["cuda"] and torch.cuda.is_available() else "cpu"
    )

    # Initialize Weights & Biases
    run_name = f"{algo_name}-{args.env}"
    if args.experiment:
        run_name += f"-{args.experiment}"
    run_name += f"-seed{config['seed']}"

    writer = wandb.init(
        project="Honors Capstone",
        name=run_name,
        config=config,  # Log all hyperparameters
        group=algo_name,
        job_type=args.env,
        tags=[algo_name, args.env] + ([args.experiment] if args.experiment else []),
    )

    # Create environments
    env = make_env(env_id=config["env_id"], xml_file=config["xml_file"])
    env_test = make_env(env_id=config["env_id"], xml_file=config["xml_file"])

    # Get env shapes
    state_shape = env.observation_space.shape
    action_shape = env.action_space.shape

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

    # Load expert buffer if needed for imitation learning
    buffer_exp = None
    if algo_name in ["bc", "airl", "gail", "dagger"]:
        buffer_exp = SerializedBuffer(
            path=config["buffer"],
            device=device,
        )

        # Log buffer information
        components = os.path.basename(Path(config["buffer"]).with_suffix("")).split("_")
        writer.config.update(
            {
                "expert_buffer_path": config["buffer"],
                "expert_buffer_size": int(components[0][4:]),
                "expert_buffer_std": float(components[1][3:]),
                "expert_buffer_prand": float(components[2][5:]),
                "expert_buffer_return": float(components[3][6:]),
            }
        )

    # Load expert if needed for imitation learning
    expert = None
    if algo_name in ["dagger"]:
        expert = SACExpert(
            state_shape=state_shape,
            action_shape=action_shape,
            device=device,
            path=config.get("expert"),
        )

    # Initialize algorithm
    algo = get_algorithm(algo_name, config, env, buffer_exp, expert)

    # Setup logging
    time_str = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(
        "logs",
        config["env_id"],
        algo_name,
        f"{args.experiment or 'default'}-seed{config['seed']}-{time_str}",
    )

    # Initialize trainer
    trainer = Trainer(
        env=env,
        env_test=env_test,
        algo=algo,
        log_dir=log_dir,
        writer=writer,
        seed=config["seed"],
        eval_interval=config.get("eval_interval", 10000),
        num_eval_episodes=config.get("num_eval_episodes", 5),
    )

    # Run training
    if algo.needs_env:
        # Online training (RL algorithms or interactive IL like DAgger)
        trainer.online_train(num_steps=config.get("num_steps", 1000000))
    else:
        # Offline training (BC or other pure IL methods)
        trainer.offline_train(num_epochs=config.get("epochs", 10))

    print(f"Training completed. Models saved to: {log_dir}")


if __name__ == "__main__":
    run_training()
