import os
import argparse
import torch
from datetime import datetime
import wandb
from stable_baselines3 import SAC  # Import SB3 SAC
from stable_baselines3.common.env_util import make_vec_env  # Useful for SB3
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
)  # For evaluation and saving
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from wandb.integration.sb3 import WandbCallback  # W&B specific callback

# Your existing utility functions (assuming they return Gymnasium-compatible envs)
from imitation_learning.utils.env import (
    make_env,
    make_flattened_env,
)
from imitation_learning.utils.utils import get_config, get_hidden_units_from_state_dict
from imitation_learning.network import AIRLDiscrim
from imitation_learning.utils.buffer import AIRLReplayBuffer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Expert Training Script using Stable Baselines3"
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
    parser.add_argument(
        "--weights",
        type=str,
        default=None,  # Defaults to None if not provided
        help="Path to SB3 model (.zip file) to load and continue training.",
    )

    return parser.parse_args()


def run_training():
    """Runs the training process using Stable Baselines3"""
    args = parse_args()

    # Load configuration
    config = get_config("sac", args.env, args.experiment)

    # --- Set Device ---
    if config["cuda"] and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    # -------------------

    # --- Initialize Weights & Biases ---
    run_name = f"sb3-sac-{args.env}"
    if args.experiment:
        run_name += f"-{args.experiment}"
    run_name += f"-seed{config['seed']}"

    # Use run_id for local logging paths to avoid collisions if W&B run fails
    run = wandb.init(
        project="Honors Capstone",  # Or your desired project name
        name=run_name,
        config=config,  # Log all hyperparameters
        sync_tensorboard=True,  # Sync SB3 TensorBoard logs to W&B
        monitor_gym=True,  # Automatically log videos of agent interaction (if env has render 'rgb_array')
        save_code=True,  # Save the main script to W&B
        group="sb3-sac",
        job_type=args.env,
        tags=["sb3", "sac", args.env] + ([args.experiment] if args.experiment else []),
    )
    # ---------------------------------

    # --- Create Local Log Directory ---
    # Based on W&B run ID for consistency and avoiding conflicts
    log_dir = os.path.join("logs", "sb3", run.id)
    os.makedirs(log_dir, exist_ok=True)
    print(f"Logging locally to: {log_dir}")
    # ---------------------------------

    reward_model_instance = None
    if config.get("use_reward_model", False):
        print("Loading AIRL reward model...")
        # Need state shape *before* creating the model
        # Create a temporary env to get shapes
        temp_env = make_env(config["env_id"])

        state_shape = temp_env.observation_space.shape
        temp_env.close()
        del temp_env

        hidden_layers = get_hidden_units_from_state_dict(config["reward_model_path"])
        reward_model_instance = AIRLDiscrim(
            state_shape=state_shape,
            gamma=config.get(
                "reward_model_gamma", 0.99
            ),  # Pass gamma if AIRLDiscrim needs it
            hidden_units_r=hidden_layers["g"],
            hidden_units_v=hidden_layers["h"],
        ).to(
            device
        )  # Send reward model to the primary device initially

        reward_model_instance.load_state_dict(
            torch.load(config["reward_model_path"], map_location=device)
        )
        reward_model_instance.eval()  # Set to evaluation mode

        # Log reward model info to W&B config AFTER init
        run.config.update({"disc_hidden_layers": hidden_layers}, allow_val_change=True)
        print(f"AIRL reward model loaded from: {config['reward_model_path']}")

    # --- Environment Setup ---
    # Create the base training environment function
    def create_train_env():
        # Instantiate base env - handle XML file logic here
        xml_file = config.get("xml_file")
        env = make_env(config["env_id"], xml_file=xml_file)

        return env

    # Create the vectorized training environment
    # Note: SB3 generally performs better with vectorized environments.
    # If your custom envs/wrappers aren't compatible with SB3's default SubprocVecEnv,
    # you might need to use DummyVecEnv: vec_env_cls=DummyVecEnv
    # num_envs = config.get("num_envs", 1) # Default to 1 if not specified
    # vec_env = make_vec_env(create_train_env, n_envs=num_envs, seed=config["seed"])
    # For simplicity matching the original single-env Trainer, let's start with one env:
    env = create_train_env()

    # --- Create Test Environment (for EvalCallback) ---
    # Test env should generally have the same wrapping as the train env
    def create_test_env():
        xml_file = config.get("xml_file")

        _env_test = make_env(config["env_id"], xml_file=xml_file)

        return _env_test

    # vec_eval_env = make_vec_env(create_test_env, n_envs=1, seed=config["seed"] + 1000) # Use different seed
    env_test = create_test_env()
    # -------------------------------------

    # --- Log Final Environment Info to W&B ---
    # Update config AFTER env creation to get final shapes
    run.config.update(
        {
            "state_shape": env.observation_space.shape,
            "action_shape": env.action_space.shape,
            "max_episode_steps": getattr(
                env.spec, "max_episode_steps", None
            ),  # Get from spec if possible
        },
        allow_val_change=True,
    )
    # ------------------------------------------

    # --- Define Callbacks ---
    # W&B Callback for logging and model saving as artifacts
    wandb_callback = WandbCallback(
        gradient_save_freq=config.get(
            "gradient_save_freq", 0
        ),  # Set > 0 to log gradients
        model_save_path=os.path.join(
            log_dir, f"models/{run.id}"
        ),  # Save model checkpoints in W&B Artifacts folder
        model_save_freq=config.get(
            "save_interval", 10000
        ),  # How often to save models to W&B
        log="all",  # Log gradients, parameters histograms ('all' or None)
        verbose=2,
    )

    # Evaluation Callback
    eval_callback = EvalCallback(
        env_test,  # Use the non-vectorized test env
        best_model_save_path=os.path.join(
            log_dir, "best_model"
        ),  # Save best model locally
        log_path=log_dir,  # Log eval results locally (sync'd by W&B)
        eval_freq=max(
            config["eval_interval"], 1
        ),  # Frequency in steps (adjust if using >1 envs)
        n_eval_episodes=config.get(
            "n_eval_episodes", 5
        ),  # Number of episodes for evaluation
        deterministic=True,  # Use deterministic actions for evaluation
        render=False,  # Keep render false unless you need evaluation videos locally
    )

    # Checkpoint Callback
    checkpoint_callback = CheckpointCallback(
        save_freq=max(config.get("save_interval", 10000), 1),  # Freq in steps
        save_path=log_dir,
        name_prefix="sac_model",
        save_replay_buffer=False,  # Usually false for SAC to save space
        save_vecnormalize=False,  # Set True if using VecNormalize wrapper
    )

    # Combine callbacks
    callback_list = [wandb_callback, eval_callback, checkpoint_callback]
    # ------------------------

    # --- Instantiate SB3 SAC Algorithm ---
    # Map config parameters to SB3 SAC arguments
    policy_kwargs = dict(
        net_arch=config.get(
            "hidden_sizes", [256, 256]
        )  # Actor and Critic hidden layers
    )
    # Handle learning rate: SB3 often uses a single LR or schedules.
    # If your config has separate LRs, you might need custom schedules or policy.
    # Using a single 'learning_rate' from config as a common case.
    learning_rate = config.get("learning_rate", 3e-4)

    sac_kwargs = {
        "policy": "MlpPolicy",
        "learning_rate": learning_rate,
        "buffer_size": config.get("buffer_size", 1_000_000),
        "learning_starts": config.get("start_steps", 10000),
        "batch_size": config.get("batch_size", 256),
        "tau": config.get("tau", 0.005),
        "gamma": config.get("discount_factor", 0.99),
        "train_freq": (config["train_freq"], "step"),
        "gradient_steps": config.get("update_steps", 1),
        "policy_kwargs": policy_kwargs,
        "verbose": 1,
        "seed": config["seed"],
        "device": device,
        "use_sde": config["use_sde"],
        "tensorboard_log": os.path.join(log_dir, "tb_logs"),
    }

    if config.get("use_reward_model", False) and reward_model_instance is not None:
        print(">>> Using AIRLReplayBuffer <<<")
        sac_kwargs["replay_buffer_class"] = AIRLReplayBuffer
        sac_kwargs["replay_buffer_kwargs"] = dict(
            reward_model=reward_model_instance,
            # Pass other args AIRLReplayBuffer might need from config
            gamma=config.get(
                "discount_factor", 0.99
            ),  # Make sure buffer gamma matches agent gamma for consistency
            normalize_reward=True,  # Match the flag from your original make_custom_reward_env call
        )
    else:
        print(">>> Using standard SB3 ReplayBuffer <<<")

    # --- Instantiate or Load SB3 SAC Algorithm ---
    if args.weights:
        print(f"--- Loading model from: {args.weights} ---")
        # Load the model
        model = SAC.load(path=args.weights, env=env, kwargs=sac_kwargs)
        print("Model loaded successfully.")
    else:
        print("--- Creating new SAC model ---")
        model = SAC(env=env, kwargs=sac_kwargs)
        print("New model created.")

    # --- Start Training ---
    print("Starting training...")
    reset_timesteps = args.weights is None  # Reset if NOT loading from weights
    model.learn(
        total_timesteps=config["num_steps"],
        callback=callback_list,  # Pass the list of callbacks
        log_interval=config.get(
            "log_interval", 4
        ),  # How often to log training stats (in episodes)
        reset_num_timesteps=reset_timesteps,
    )
    print("Training finished.")
    # ----------------------

    # --- Save Final Model Locally (Optional) ---
    final_model_path = os.path.join(log_dir, "final_model.zip")
    model.save(final_model_path)
    print(f"Final model saved locally to: {final_model_path}")
    # -------------------------------------------

    # --- Finish W&B Run ---
    run.finish()
    # ----------------------

    # --- Close Environments ---
    env.close()
    env_test.close()
    # if 'vec_env' in locals(): vec_env.close() # Close if vectorized env was used
    # if 'vec_eval_env' in locals(): vec_eval_env.close()
    # -----------------------


if __name__ == "__main__":
    run_training()
