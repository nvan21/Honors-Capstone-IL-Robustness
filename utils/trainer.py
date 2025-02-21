import os
from datetime import timedelta
from time import time, sleep
from torch.utils.tensorboard.writer import SummaryWriter

from gymnasium import Env
from algos import Algorithm


class Trainer:

    def __init__(
        self,
        env: Env,
        env_test: Env,
        algo: Algorithm,
        log_dir: str,
        seed: int = 0,
        num_steps: int = 10**5,
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
        self.writer = SummaryWriter(log_dir=self.summary_dir)
        self.model_dir = os.path.join(log_dir, "model")
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        # Other parameters
        self.num_steps = num_steps
        self.eval_interval = eval_interval
        self.num_eval_episodes = num_eval_episodes

    def train(self):
        # Training start time
        self.start_time = time()

        # Episode's timestep
        t = 0

        # Initialize the environment
        state, _ = self.env.reset(seed=self.env_seed)

        for step in range(1, self.num_steps + 1):
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

        self.writer.add_scalar("return/test", mean_return, step)
        print(
            f"Num steps: {step:<6}   "
            f"Return: {mean_return:<5.1f}   "
            f"Time: {self.time}"
        )

    @property
    def time(self):
        return str(timedelta(seconds=int(time() - self.start_time)))
