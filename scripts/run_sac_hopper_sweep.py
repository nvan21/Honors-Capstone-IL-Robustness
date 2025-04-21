import os
import argparse
import torch
from datetime import datetime
import wandb
import optuna  # Import Optuna
from functools import partial  # Useful for passing args to objective

from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    EvalCallback,  # We inherit from this
    CheckpointCallback,
    BaseCallback,  # Needed for inheritance typing if strict
)
from stable_baselines3.common.monitor import Monitor  # Crucial for EvalCallback
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
)  # Often needed for single env eval with EvalCallback
from wandb.integration.sb3 import WandbCallback

# Your existing utility functions
from imitation_learning.utils.env import make_env
from imitation_learning.utils.utils import get_config, get_hidden_units_from_state_dict
from imitation_learning.network import AIRLDiscrim
from imitation_learning.utils.buffer import AIRLReplayBuffer


# --- Custom Callback for Optuna Integration ---
class OptunaPruningEvalCallback(EvalCallback):
    """
    Custom EvalCallback that integrates with Optuna for reporting results
    and handling pruning.
    """

    def __init__(self, trial: optuna.trial.Trial, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trial = trial
        # Store the best reward explicitly for retrieval later
        self._best_mean_reward = -float("inf")  # Internal variable to store the best

    def _on_step(self) -> bool:
        continue_training = super()._on_step()
        if continue_training is False:
            return False

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            current_mean_reward = self.last_mean_reward
            self.trial.report(current_mean_reward, self.num_timesteps)

            # Update our internal best reward tracker
            if current_mean_reward > self._best_mean_reward:
                self._best_mean_reward = (
                    current_mean_reward  # Assign to internal variable
                )

            if self.trial.should_prune():
                print(
                    f"Trial {self.trial.number} pruned at step {self.num_timesteps} with reward {current_mean_reward:.4f}."
                )
                raise optuna.exceptions.TrialPruned()

        return continue_training


# --- Define Hyperparameter Search Space ---
# (Same as before - edit ranges as needed)
def suggest_hyperparameters(trial: optuna.trial.Trial, base_config: dict):
    """Suggests hyperparameters for a given Optuna trial."""
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
        "buffer_size": trial.suggest_categorical(
            "buffer_size", [int(3e5), int(5e5), int(1e6)]
        ),
        "gamma": trial.suggest_float("gamma", 0.98, 0.999, log=True),
        "tau": trial.suggest_float("tau", 0.001, 0.02, log=True),
        "gradient_steps": trial.suggest_int("gradient_steps", 1, 5),
        "use_sde": trial.suggest_categorical("use_sde", [True, False]),
    }
    combined_config = base_config.copy()
    combined_config.update(params)
    combined_config["discount_factor"] = combined_config["gamma"]
    combined_config["update_steps"] = combined_config["gradient_steps"]

    return combined_config


# --- Define the Objective Function for Optuna ---
def objective(
    trial: optuna.trial.Trial,
    args: argparse.Namespace,
    base_config: dict,
    study_name: str,
) -> float:
    """Runs a single training trial with hyperparameters suggested by Optuna."""

    config = suggest_hyperparameters(trial, base_config)
    print(f"\n--- Starting Trial {trial.number} ---")
    print(
        "Using hyperparameters:",
        {k: config[k] for k in suggest_hyperparameters(trial, {}).keys()},
    )

    if config["cuda"] and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    run_name = f"sb3-sac-{args.env}-trial{trial.number}"
    if args.experiment:
        run_name = f"{args.experiment}-trial{trial.number}"

    log_dir = os.path.join("logs", "sb3", study_name, f"trial_{trial.number}")
    os.makedirs(log_dir, exist_ok=True)
    print(f"Logging locally for Trial {trial.number} to: {log_dir}")

    run = None
    env = None
    env_test_vec = None  # Use VecEnv for EvalCallback

    try:
        run = wandb.init(
            project="Honors Capstone HPO",
            name=run_name,
            config=config,
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True,
            group=study_name,
            job_type=f"{args.env}-optuna",
            tags=["sb3", "sac", args.env, "optuna", study_name]
            + ([args.experiment] if args.experiment else []),
            reinit=True,
            dir=log_dir,
        )

        # --- Load Reward Model (if applicable) ---
        reward_model_instance = None
        if config.get("use_reward_model", False):
            # ... (Reward model loading code remains the same) ...
            print("Loading AIRL reward model...")
            temp_env = make_env(config["env_id"])
            state_shape = temp_env.observation_space.shape
            temp_env.close()
            del temp_env

            hidden_layers = get_hidden_units_from_state_dict(
                config["reward_model_path"]
            )
            print(hidden_layers, config["reward_model_path"])
            reward_model_instance = AIRLDiscrim(
                state_shape=state_shape,
                gamma=config.get("reward_model_gamma", 0.99),
                hidden_units_r=hidden_layers["g"],
                hidden_units_v=hidden_layers["h"],
            ).to(device)
            reward_model_instance.load_state_dict(
                torch.load(config["reward_model_path"], map_location=device)
            )
            reward_model_instance.eval()
            run.config.update(
                {"disc_hidden_layers": hidden_layers}, allow_val_change=True
            )
            print(f"AIRL reward model loaded from: {config['reward_model_path']}")

        # --- Environment Setup ---
        # Training Environment (needs Monitor)
        def create_train_env():
            xml_file = config.get("xml_file")
            _env = Monitor(
                make_env(config["env_id"], xml_file=xml_file),
                filename=os.path.join(log_dir, "train_monitor.csv"),
            )  # Log train stats
            return _env

        env = create_train_env()  # Single env for training

        # Test Environment (needs Monitor and often DummyVecEnv for EvalCallback)
        def create_test_env():
            xml_file = config.get("xml_file")
            # Log eval stats separately if desired
            _env_test = Monitor(
                make_env(config["env_id"], xml_file=xml_file),
                filename=os.path.join(log_dir, "eval_monitor.csv"),
            )
            return _env_test

        # Wrap the single test env in DummyVecEnv for EvalCallback
        env_test_vec = DummyVecEnv([create_test_env])

        run.config.update(
            {
                "state_shape": env.observation_space.shape,
                "action_shape": env.action_space.shape,
                "max_episode_steps": getattr(env.spec, "max_episode_steps", None),
            },
            allow_val_change=True,
        )

        # --- Define Callbacks for this trial ---
        wandb_callback = WandbCallback(
            gradient_save_freq=config.get("gradient_save_freq", 0),
            model_save_path=os.path.join(log_dir, f"models"),
            model_save_freq=config.get("save_interval", 20000),
            log="gradients" if config.get("gradient_save_freq", 0) > 0 else None,
            verbose=0,
        )

        # Use the CUSTOM Eval Callback for Optuna Pruning
        eval_freq = max(config["eval_interval"], 1)
        optuna_eval_callback = OptunaPruningEvalCallback(
            trial,  # Pass the Optuna trial object HERE
            eval_env=env_test_vec,  # MUST use the VecEnv version for EvalCallback
            n_eval_episodes=config.get("n_eval_episodes", 5),
            eval_freq=eval_freq,
            log_path=log_dir,  # Log eval results locally
            best_model_save_path=os.path.join(
                log_dir, "best_model"
            ),  # Still save best model locally
            deterministic=True,
            render=False,
            verbose=0,  # Keep eval quiet
        )

        checkpoint_callback = CheckpointCallback(
            save_freq=max(config.get("save_interval", 50000), eval_freq),
            save_path=os.path.join(log_dir, "checkpoints"),
            name_prefix="sac_model",
            save_replay_buffer=False,
            save_vecnormalize=False,
            verbose=0,
        )

        callback_list = [wandb_callback, optuna_eval_callback, checkpoint_callback]

        # --- Instantiate SB3 SAC Algorithm ---
        policy_kwargs = dict(net_arch=config["hidden_sizes"])
        sac_kwargs = {
            "policy": "MlpPolicy",
            "env": env,  # Use the single, monitored training env
            "learning_rate": config["learning_rate"],
            "buffer_size": config["buffer_size"],
            "learning_starts": config.get("start_steps", 10000),
            "batch_size": config["batch_size"],
            "tau": config["tau"],
            "gamma": config["gamma"],
            "train_freq": (config.get("train_freq", 1), "step"),
            "gradient_steps": config["gradient_steps"],
            "policy_kwargs": policy_kwargs,
            "verbose": 0,
            "seed": config["seed"] + trial.number,  # Vary seed per trial
            "device": device,
            "use_sde": config["use_sde"],
            "tensorboard_log": os.path.join(log_dir, "tb_logs"),
        }

        if config.get("use_reward_model", False) and reward_model_instance is not None:
            # ... (Replay buffer setup remains the same) ...
            print(">>> Using AIRLReplayBuffer <<<")
            sac_kwargs["replay_buffer_class"] = AIRLReplayBuffer
            sac_kwargs["replay_buffer_kwargs"] = dict(
                reward_model=reward_model_instance,
                gamma=config["gamma"],
                normalize_reward=config.get("normalize_reward", True),
            )
        else:
            print(">>> Using standard SB3 ReplayBuffer <<<")

        model = SAC(**sac_kwargs)

        # --- Start Training for this trial ---
        print(f"Starting training for Trial {trial.number}...")
        # Train the model
        model.learn(
            total_timesteps=config["num_steps"],
            callback=callback_list,
            log_interval=100,
        )
        # If training completes without pruning, get the best reward found by the callback
        final_best_reward = optuna_eval_callback.best_mean_reward
        print(
            f"Training finished for Trial {trial.number}. Best reward: {final_best_reward:.4f}"
        )
        return final_best_reward  # Return the best reward achieved during the trial

    except optuna.exceptions.TrialPruned:
        print(f"Trial {trial.number} successfully pruned.")
        # Let Optuna know it was pruned
        raise optuna.exceptions.TrialPruned()
    # except Exception as e:
    #     print(f"Trial {trial.number} failed with error: {e}")
    #     # Log the error to W&B if the run exists
    #     if run:
    #         run.log({"trial_error": str(e)})
    #     # Indicate failure to Optuna
    #     return -float("inf")

    finally:
        # --- Cleanup for this trial ---
        print(f"Cleaning up resources for Trial {trial.number}...")
        if env is not None and hasattr(env, "close"):
            env.close()
        if env_test_vec is not None and hasattr(env_test_vec, "close"):
            env_test_vec.close()
        if run is not None:
            run.finish()
        print(f"--- Finished Trial {trial.number} ---")


# --- Parse Args ---
# (Same as before)
def parse_args():
    parser = argparse.ArgumentParser(
        description="Expert Training Script using Stable Baselines3 with Optuna HPO"
    )
    parser.add_argument("--env", type=str, required=True, help="Environment name")
    parser.add_argument(
        "--experiment", type=str, default=None, help="Base experiment name"
    )
    parser.add_argument(
        "--n-trials", type=int, default=50, help="Number of Optuna trials"
    )
    parser.add_argument(
        "--study-name", type=str, default=None, help="Name for the Optuna study"
    )
    parser.add_argument("--seed", type=int, default=None, help="Base seed for study")
    return parser.parse_args()


# --- Run Optuna Study ---
# (Same as before)
def run_optuna_study():
    args = parse_args()
    base_config = get_config("sac", args.env, args.experiment)

    if args.seed is not None:
        base_config["seed"] = args.seed
    elif "seed" not in base_config:
        base_config["seed"] = 42
    print(f"Using base seed: {base_config['seed']}")

    study_name = (
        args.study_name
        or f"sb3-sac-{args.env}-{args.experiment or 'base'}-{datetime.now().strftime('%Y%m%d-%H%M')}"
    )
    print(f"Running Optuna study: {study_name}")

    # Configure pruner based on eval_interval
    eval_interval = base_config.get(
        "eval_interval", 1000
    )  # Get eval interval from config
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=3,  # Prune after 3 evaluations
        interval_steps=1,  # Check pruning every evaluation step
    )
    print(f"Using Pruner: {pruner}")

    study = optuna.create_study(
        study_name=study_name, direction="maximize", pruner=pruner
    )

    objective_partial = partial(
        objective, args=args, base_config=base_config, study_name=study_name
    )

    try:
        study.optimize(
            objective_partial, n_trials=args.n_trials, timeout=None, n_jobs=2
        )
    except KeyboardInterrupt:
        print("Optimization interrupted by user.")

    # --- Analyze Results ---
    print("\n--- Optimization Finished ---")
    # ... (Result analysis code remains the same) ...
    print(f"Study Name: {study_name}")
    print(f"Number of finished trials: {len(study.trials)}")
    try:
        best_trial = study.best_trial
        print("\nBest trial:")
        print(f"  Value (Max Reward): {best_trial.value:.4f}")
        print("  Params: ")
        for key, value in best_trial.params.items():
            display_key = key
            if key == "gamma":
                display_key = "discount_factor"
            if key == "gradient_steps":
                display_key = "update_steps"
            if key == "hidden_dim":
                display_key = "hidden_layer_size"
            print(f"    {display_key}: {value}")
        print(f"  Trial Number: {best_trial.number}")
    except ValueError:
        print("No trials completed successfully.")


if __name__ == "__main__":
    run_optuna_study()
