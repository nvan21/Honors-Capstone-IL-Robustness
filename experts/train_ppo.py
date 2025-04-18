import os
import argparse
import torch
import numpy as np  # Make sure numpy is imported
from datetime import datetime
import wandb
import gymnasium as gym  # Import gymnasium
from gymnasium_robotics.core import GoalEnv

# <<< CHANGE: Import PPO >>>
from stable_baselines3 import PPO

# <<< END CHANGE >>>
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
)  # Import VecEnv types
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
)

# <<< CHANGE: Remove ReplayBuffer imports >>>
# from stable_baselines3.common.buffers import ReplayBuffer
# from stable_baselines3.common.type_aliases import ReplayBufferSamples
# <<< END CHANGE >>>
# <<< CHANGE: Import RunningMeanStd if using shared RMS >>>
from stable_baselines3.common.running_mean_std import RunningMeanStd

# <<< END CHANGE >>>
from wandb.integration.sb3 import WandbCallback

from imitation_learning.utils.env import (
    make_env,
    make_flattened_env,
    # <<< CHANGE: Import the new wrapper >>>
    AIRLRewardWrapper,  # Make sure this is accessible (defined above or imported)
    # <<< END CHANGE >>>
)
from imitation_learning.utils.utils import get_config, get_hidden_units_from_state_dict
from imitation_learning.network import AIRLDiscrim

# <<< CHANGE: Remove AIRLReplayBuffer import >>>
# from imitation_learning.utils.buffer import AIRLReplayBuffer
# <<< END CHANGE >>>
import functools  # Import functools for passing args during env creation

# ... (parse_args function remains the same) ...


def run_training():
    args = parse_args()
    # <<< CHANGE: Load PPO config >>>
    config = get_config("ppo", args.env, args.experiment)  # Use "ppo" config section
    # <<< END CHANGE >>>

    if config["cuda"] and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # --- W&B and Logging Setup (remains similar) ---
    run_name = f"sb3-ppo-{args.env}"  # Changed name to ppo
    if args.experiment:
        run_name += f"-{args.experiment}"
    run_name += f"-seed{config['seed']}"

    run = wandb.init(...)  # W&B init as before, adjust tags if needed
    log_dir = os.path.join("logs", "sb3", run.id)
    os.makedirs(log_dir, exist_ok=True)
    print(f"Logging locally to: {log_dir}")
    # ----------------------------------------------

    reward_model_instance = None
    # <<< CHANGE: Initialize Shared RMS object here if normalizing >>>
    shared_reward_rms = None
    use_reward_model_flag = config.get("use_reward_model", False)
    normalize_reward_flag = config.get(
        "normalize_airl_reward", True
    )  # Config flag for normalization

    if use_reward_model_flag:
        print("Loading AIRL reward model...")
        # ... (loading logic remains the same) ...
        state_shape = ...  # Get state shape
        reward_model_instance = AIRLDiscrim(...)
        # ... (loading state dict) ...
        print(f"AIRL reward model loaded from: {config['reward_model_path']}")

        # Initialize shared RMS if needed
        if normalize_reward_flag:
            print("Initializing SHARED RunningMeanStd for reward normalization.")
            shared_reward_rms = RunningMeanStd(
                shape=(), epsilon=config.get("reward_norm_epsilon", 1e-8)
            )
    # <<< END CHANGE >>>

    # --- Environment Setup ---
    # <<< CHANGE: Modify create_train_env to wrap with AIRLRewardWrapper >>>
    def create_train_env_lambda(
        reward_model, gamma, normalize, rms_obj, epsilon, device
    ):
        # This function is what make_vec_env will call for each environment
        xml_file = ...  # Your XML logic
        env = make_env(config["env_id"], xml_file=xml_file)
        if isinstance(env.unwrapped, GoalEnv):
            env = make_flattened_env(env)

        # Wrap the environment ONLY if using the reward model
        if reward_model is not None:
            print(f"Wrapping env with AIRLRewardWrapper (Normalize: {normalize})")
            env = AIRLRewardWrapper(
                env=env,
                reward_model=reward_model,
                gamma=gamma,
                normalize_reward=normalize,
                reward_norm_epsilon=epsilon,
                shared_reward_rms=rms_obj,  # Pass the shared RMS object
                device=device,
            )
        else:
            print("Using standard environment reward.")

        return env

    num_envs = config.get("num_envs", 4)
    print(f"Using {num_envs} parallel environments for training.")

    # Use functools.partial to pass the necessary fixed arguments to the env creation function
    train_env_creator = functools.partial(
        create_train_env_lambda,
        reward_model=reward_model_instance if use_reward_model_flag else None,
        gamma=config.get("discount_factor", 0.99),  # Use agent's gamma for consistency
        normalize=normalize_reward_flag if use_reward_model_flag else False,
        rms_obj=shared_reward_rms,  # Pass the shared (or None) RMS object
        epsilon=config.get("reward_norm_epsilon", 1e-8),
        device=device,  # Pass device for wrapper
    )

    vec_env = make_vec_env(
        train_env_creator,  # Pass the partial function
        n_envs=num_envs,
        seed=config["seed"],
        vec_env_cls=DummyVecEnv,  # Or SubprocVecEnv
    )
    # <<< END CHANGE >>>

    # --- Create Test Environment (remains the same, uses ground truth reward) ---
    def create_test_env():
        # ... (same as before) ...
        return _env_test

    env_test = create_test_env()
    # -------------------------------------

    # --- Log Final Environment Info (remains similar) ---
    run.config.update(...)
    # ------------------------------------------

    # --- Define Callbacks (remains similar, adjust frequencies/names if needed) ---
    wandb_callback = WandbCallback(...)
    eval_callback = EvalCallback(env_test, ...)  # Frequencies adjusted for num_envs
    checkpoint_callback = CheckpointCallback(...)  # Frequencies adjusted for num_envs
    callback_list = [wandb_callback, eval_callback, checkpoint_callback]
    # ------------------------

    # --- Instantiate SB3 PPO Algorithm ---
    policy_kwargs = dict(net_arch=config.get("hidden_sizes", [256, 256]))
    learning_rate = config.get("learning_rate", 3e-4)

    # <<< CHANGE: Define PPO kwargs, remove SAC kwargs >>>
    ppo_kwargs = {
        "policy": "MlpPolicy",
        "env": vec_env,  # Use the wrapped vectorized environment
        "learning_rate": learning_rate,
        "n_steps": config.get("n_steps", 2048)
        // num_envs,  # Rollout buffer size per env
        "batch_size": config.get("batch_size", 64),  # PPO minibatch size
        "n_epochs": config.get("n_epochs", 10),  # PPO optimization epochs per rollout
        "gamma": config.get("discount_factor", 0.99),
        "gae_lambda": config.get("gae_lambda", 0.95),
        "clip_range": config.get("clip_range", 0.2),
        "clip_range_vf": None,  # Often None or same as clip_range
        "ent_coef": config.get("ent_coef", 0.0),  # PPO entropy coefficient
        "vf_coef": config.get("vf_coef", 0.5),  # Value function coefficient
        "max_grad_norm": config.get("max_grad_norm", 0.5),
        "use_sde": config.get(
            "use_sde", False
        ),  # Add if you want State-Dependent Exploration
        "sde_sample_freq": config.get("sde_sample_freq", -1),
        "policy_kwargs": policy_kwargs,
        "verbose": 1,
        "seed": config["seed"],
        "device": device,
        "tensorboard_log": None,  # Keep disabled if using W&B sync
    }

    model = PPO(**ppo_kwargs)  # Instantiate PPO
    # <<< END CHANGE >>>
    # ------------------------------------

    # --- Start Training (remains similar) ---
    print("Starting training...")
    model.learn(
        total_timesteps=config["num_steps"],
        callback=callback_list,
        log_interval=config.get("log_interval", 1),  # PPO logs typically every rollout
    )
    print("Training finished.")
    # ----------------------

    # --- Save, Finish W&B, Close Envs (remains similar) ---
    final_model_path = os.path.join(log_dir, "final_model.zip")
    model.save(final_model_path)
    print(f"Final model saved locally to: {final_model_path}")
    run.finish()
    vec_env.close()
    env_test.close()
    # ----------------------------------------------------


if __name__ == "__main__":
    run_training()
