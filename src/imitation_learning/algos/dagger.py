import torch
import numpy as np
from tqdm import tqdm
import os

from .bc import BC
from imitation_learning.utils.buffer import Buffer


class DAgger(BC):
    """
    DAgger (Dataset Aggregation) algorithm for imitation learning.

    DAgger improves upon Behavior Cloning by iteratively:
    1. Training a policy on the current dataset
    2. Running that policy to collect new states
    3. Querying an expert for actions on those states
    4. Adding these state-action pairs to the dataset
    5. Retraining the policy on the augmented dataset

    This algorithm addresses covariate shift in Behavior Cloning by collecting
    states that the learner actually visits during execution.
    """

    def __init__(
        self,
        expert,  # Expert policy to query for actions
        buffer_exp,  # Initial expert buffer
        state_shape,
        action_shape,
        device,
        seed,
        gamma=0.99,
        rollout_length=1000,  # How many steps to collect before updating
        batch_size=64,
        lr_actor=3e-4,
        units_actor=(256, 256),
        beta=0.0,  # Mixing parameter: 0 = pure learner, 1 = pure expert during data collection
    ):
        super().__init__(
            buffer_exp=buffer_exp,
            state_shape=state_shape,
            action_shape=action_shape,
            device=device,
            seed=seed,
            gamma=gamma,
            needs_env=True,  # DAgger needs environment
            units_actor=units_actor,
            lr_actor=lr_actor,
            batch_size=batch_size,
        )

        self.expert = expert
        self.beta = beta
        self.rollout_length = rollout_length
        self.dagger_iteration = 0

        # Create a new buffer for DAgger data collection
        self.dagger_buffer = Buffer(
            buffer_size=rollout_length * 10,  # Store multiple iterations
            state_shape=state_shape,
            action_shape=action_shape,
            device=device,
        )
        self.transitions_collected = 0

    def is_update(self, step):
        """Determine if we should update the policy."""
        return self.transitions_collected >= self.rollout_length

    def step(self, env, state, t, step):
        """
        Execute one step in the environment, collect data for DAgger.

        In DAgger, we use our current policy to generate trajectories,
        but query the expert for the correct actions at those states.
        """
        t += 1

        # Decide which policy to use for state collection
        if np.random.rand() < self.beta:
            # With probability beta, use expert's policy
            action = self.expert.exploit(state)
        else:
            # With probability 1-beta, use current learned policy
            action = self.exploit(state)

        # Execute action in environment
        next_state, reward, terminated, truncated, _ = env.step(action)

        # Query expert for the correct action at this state (DAgger)
        expert_action = self.expert.exploit(state)

        # Add to dagger buffer with expert's action
        self.dagger_buffer.append(
            state, expert_action, reward, terminated or truncated, next_state
        )
        self.transitions_collected += 1

        if terminated or truncated:
            t = 0
            next_state, _ = env.reset()

        return next_state, t

    def update(self, writer):
        """
        Update policy with BC on the aggregated dataset.
        """
        self.learning_steps += 1
        self.dagger_iteration += 1
        self.transitions_collected = 0

        # Calculate number of BC iterations based on data size
        expert_size = len(self.buffer_exp)
        dagger_size = len(self.dagger_buffer)
        total_size = expert_size + dagger_size
        iterations = total_size // self.batch_size

        # Use a fixed sampling ratio instead of proportional
        fixed_dagger_ratio = 0.5  # 50% expert, 50% DAgger

        for _ in range(iterations):
            # Fixed ratio sampling
            if dagger_size > 0 and np.random.rand() < fixed_dagger_ratio:
                states, actions, _, _, _ = self.dagger_buffer.sample(self.batch_size)
            else:
                states, actions, _, _, _ = self.buffer_exp.sample(self.batch_size)

            # Train on the batch
            pred_actions = self.actor(states)
            loss = self.criterion(pred_actions, actions)

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

        # Log DAgger-specific metrics
        writer.log("stats/dagger_iteration", self.dagger_iteration, self.learning_steps)
        writer.log("stats/dagger_buffer_size", dagger_size, self.learning_steps)
        writer.log("stats/total_buffer_size", total_size, self.learning_steps)

    def save_models(self, save_dir):
        """
        Save all models and the DAgger buffer.
        """
        # Call parent to save the actor
        super().save_models(save_dir)

        # Also save the DAgger buffer
        self.dagger_buffer.save(os.path.join(save_dir, "dagger_buffer.pth"))
