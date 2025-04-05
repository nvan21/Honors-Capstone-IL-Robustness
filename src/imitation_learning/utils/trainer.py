import os
from datetime import timedelta
from time import time, sleep
from wandb.wandb_run import Run

from gymnasium import Env
from imitation_learning.algos import Algorithm


class Trainer:

    def __init__(
        self,
        env: Env,
        env_test: Env,
        algo: Algorithm,
        log_dir: str,
        writer: Run,
        seed: int = 0,
        eval_interval: int = 10**3,
        num_eval_episodes: int = 5,
    ):

        # Env to collect samples
        self.env = env
        self.env_seed = seed

        # Env for evaluation
        self.env_test = env_test
        self.env_test_seed = 2**31 - seed

        self.algo = algo
        self.log_dir = log_dir

        # Log setting.
        self.summary_dir = os.path.join(log_dir, "summary")
        self.writer = writer
        self.writer.define_metric("Steps")
        self.writer.define_metric("return/test", step_metric="Steps")
        self.writer.define_metric("loss/*", step_metric="Steps")
        self.writer.define_metric("stats/*", step_metric="Steps")
        self.model_dir = os.path.join(log_dir, "model")
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        # Other parameters
        self.eval_interval = eval_interval
        self.num_eval_episodes = num_eval_episodes

    def online_train(self, num_steps):
        # Training start time
        self.start_time = time()

        # Episode's timestep
        t = 0

        # Initialize the environment
        state, _ = self.env.reset(seed=self.env_seed)

        for step in range(1, num_steps + 1):
            state, t = self.algo.step(self.env, state, t, step)

            # Update the algorithm whenever ready
            if self.algo.is_update(step):
                self.algo.update(self.writer)

            # Evaluate regularly
            if step % self.eval_interval == 0:
                self.evaluate(step)
                self.algo.save_models(os.path.join(self.model_dir, f"step{step}"))

        # Wait for the logging to be finished.
        sleep(10)

    def offline_train(self, num_epochs):
        # Training start time
        self.start_time = time()

        for step in range(num_epochs):
            # Update the algorithm whenever ready
            if self.algo.is_update(step):
                self.algo.update(self.writer)

            # Evaluate regularly
            if step % self.eval_interval == 0:
                self.evaluate(step)
                self.algo.save_models(os.path.join(self.model_dir, f"step{step}"))

    def evaluate(self, step):
        mean_return = 0.0

        for _ in range(self.num_eval_episodes):
            state, _ = self.env_test.reset(seed=self.env_test_seed)
            episode_return = 0.0
            terminated = False
            truncated = False

            while not terminated and not truncated:
                action = self.algo.exploit(state)
                state, reward, terminated, truncated, _ = self.env_test.step(action)
                episode_return += reward

            mean_return += episode_return / self.num_eval_episodes

        self.writer.log({"return/test": mean_return, "Steps": step})
        print(
            f"Num steps: {step:<6}   "
            f"Return: {mean_return:<5.1f}   "
            f"Time: {self.time}"
        )

    @property
    def time(self):
        return str(timedelta(seconds=int(time() - self.start_time)))
