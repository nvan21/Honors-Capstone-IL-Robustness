import os
import argparse
import torch
from datetime import datetime
import wandb
from gymnasium_robotics.core import GoalEnv  # Use gymnasium directly
from stable_baselines3 import SAC  # Import SB3 SAC
from stable_baselines3.common.env_util import make_vec_env  # Useful for SB3
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
)  # For evaluation and saving
from wandb.integration.sb3 import WandbCallback  # W&B specific callback

# Your existing utility functions (assuming they return Gymnasium-compatible envs)
from imitation_learning.utils.env import (
    make_env,
    make_custom_reward_env,
    make_flattened_env,
)
from imitation_learning.utils.utils import get_config, get_hidden_units_from_state_dict
from imitation_learning.network import AIRLDiscrim


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

    # --- Environment Setup ---
    # Create the base training environment function
    def create_train_env():
        # Instantiate base env - handle XML file logic here
        xml_file = (
            config.get("xml_file")
            if not isinstance(make_env(config["env_id"]).unwrapped, GoalEnv)
            else None
        )
        env = make_env(config["env_id"], xml_file=xml_file)

        # Add observation flattening wrapper if it's a robotics env
        if isinstance(env.unwrapped, GoalEnv):
            # Important: Flatten AFTER potential custom reward wrapping if reward uses dict obs
            env = make_flattened_env(env)

        # Alter environment to use AIRL reward if specified
        if config.get("use_reward_model", False):
            state_shape = (
                env.observation_space.shape
            )  # Get shape AFTER flattening if applicable
            hidden_layers = get_hidden_units_from_state_dict(
                config["reward_model_path"]
            )
            reward_model = AIRLDiscrim(
                state_shape=state_shape,
                gamma=config.get("reward_model_gamma", 0.99),
                hidden_units_r=hidden_layers["g"],
                hidden_units_v=hidden_layers["h"],
            ).to(
                device
            )  # Send reward model to correct device
            # Make sure device used here matches the device SB3 will use
            reward_model.load_state_dict(
                torch.load(config["reward_model_path"], map_location=device)
            )
            env = make_custom_reward_env(
                env=env, reward_model=reward_model, device=device, normalize_reward=True
            )
            # Log reward model info to W&B config AFTER init
            run.config.update(
                {"disc_hidden_layers": hidden_layers}, allow_val_change=True
            )

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
        xml_file = (
            config.get("xml_file")
            if not isinstance(make_env(config["env_id"]).unwrapped, GoalEnv)
            else None
        )
        _env_test = make_env(config["env_id"], xml_file=xml_file)
        if isinstance(_env_test.unwrapped, GoalEnv):
            _env_test = make_flattened_env(_env_test)
        # We typically DON'T use the custom reward model for *evaluation* unless specifically desired.
        # Evaluation usually measures performance on the TRUE environment task reward.
        # If you DO want to eval on the learned reward, uncomment the AIRL wrapping here too.
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

    model = SAC(
        policy="MlpPolicy",  # Standard policy for continuous spaces
        env=env,  # Pass the single, wrapped training environment
        learning_rate=learning_rate,
        buffer_size=config.get("buffer_size", 1_000_000),
        learning_starts=config.get("start_steps", 10000),
        batch_size=config.get("batch_size", 256),
        tau=config.get("tau", 0.005),
        gamma=config.get("discount_factor", 0.99),
        # train_freq / gradient_steps control how often/much updates happen
        # train_freq=1, gradient_steps=1 is default (update once per env step)
        # If your 'update_steps' meant 'N updates per env step', set gradient_steps:
        train_freq=(1, "step"),  # Check documentation if using "episode" frequency
        gradient_steps=config.get("update_steps", 1),
        # action_noise=None, # SAC usually doesn't use action noise during training
        # optimize_memory_usage=False, # Can set True for large replay buffers
        # ent_coef='auto', # Default: learn entropy coefficient
        # target_update_interval=1, # Default
        # target_entropy='auto', # Default
        policy_kwargs=policy_kwargs,
        verbose=1,  # Set to 1 for training updates, 0 for quiet
        seed=config["seed"],
        device=device,
        tensorboard_log=os.path.join(
            log_dir, "tb_logs"
        ),  # Local TB log dir (sync'd by W&B)
    )
    # ------------------------------------

    # --- Start Training ---
    print("Starting training...")
    model.learn(
        total_timesteps=config["num_steps"],
        callback=callback_list,  # Pass the list of callbacks
        log_interval=config.get(
            "log_interval", 4
        ),  # How often to log training stats (in episodes)
        # reset_num_timesteps=False # Set True if you want to continue training from a loaded model
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
