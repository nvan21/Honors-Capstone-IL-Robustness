import os
import numpy as np
import torch
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.running_mean_std import RunningMeanStd


class SerializedBuffer:

    def __init__(self, path, device):
        tmp = torch.load(path)
        self.buffer_size = self._n = tmp["state"].size(0)
        self.device = device

        self.states = tmp["state"].clone().to(self.device)
        self.actions = tmp["action"].clone().to(self.device)
        self.rewards = tmp["reward"].clone().to(self.device)
        self.dones = tmp["done"].clone().to(self.device)
        self.next_states = tmp["next_state"].clone().to(self.device)

    def __len__(self):
        return len(self.states)

    def sample(self, batch_size):
        idxes = np.random.randint(low=0, high=self._n, size=batch_size)
        return (
            self.states[idxes],
            self.actions[idxes],
            self.rewards[idxes],
            self.dones[idxes],
            self.next_states[idxes],
        )


class Buffer(SerializedBuffer):

    def __init__(self, buffer_size, state_shape, action_shape, device):
        self._n = 0
        self._p = 0
        self.buffer_size = buffer_size
        self.device = device

        self.states = torch.empty(
            (buffer_size, *state_shape), dtype=torch.float, device=device
        )
        self.actions = torch.empty(
            (buffer_size, *action_shape), dtype=torch.float, device=device
        )
        self.rewards = torch.empty((buffer_size, 1), dtype=torch.float, device=device)
        self.dones = torch.empty((buffer_size, 1), dtype=torch.float, device=device)
        self.next_states = torch.empty(
            (buffer_size, *state_shape), dtype=torch.float, device=device
        )

    def append(self, state, action, reward, done, next_state):
        self.states[self._p].copy_(torch.tensor(state))
        self.actions[self._p].copy_(torch.tensor(action))
        self.rewards[self._p] = float(reward)
        self.dones[self._p] = float(done)
        self.next_states[self._p].copy_(torch.tensor(next_state))

        self._p = (self._p + 1) % self.buffer_size
        self._n = min(self._n + 1, self.buffer_size)

    def append_batch(self, states_t, actions_t, rewards_t, dones_t, next_states_t):
        """
        Appends a batch of transitions.
        Assumes input arguments are tensors already on self.device.
        """
        batch_size = states_t.shape[0]
        if batch_size == 0:
            return  # Nothing to append

        # Ensure rewards and dones have the correct shape [batch_size, 1]
        if rewards_t.ndim == 1:
            rewards_t = rewards_t.unsqueeze(1)
        if dones_t.ndim == 1:
            dones_t = dones_t.unsqueeze(1)

        p = self._p  # Current write position

        # Calculate indices for insertion, handling wrap-around
        if p + batch_size <= self.buffer_size:
            # No wrap-around: Copy directly
            indices = slice(p, p + batch_size)
            self.states[indices] = states_t
            self.actions[indices] = actions_t
            self.rewards[indices] = rewards_t
            self.dones[indices] = dones_t
            self.next_states[indices] = next_states_t
        else:
            # Wrap-around: Copy in two parts
            num_part1 = self.buffer_size - p
            indices1 = slice(p, self.buffer_size)
            self.states[indices1] = states_t[:num_part1]
            self.actions[indices1] = actions_t[:num_part1]
            self.rewards[indices1] = rewards_t[:num_part1]
            self.dones[indices1] = dones_t[:num_part1]
            self.next_states[indices1] = next_states_t[:num_part1]

            num_part2 = batch_size - num_part1
            indices2 = slice(0, num_part2)
            self.states[indices2] = states_t[num_part1:]
            self.actions[indices2] = actions_t[num_part1:]
            self.rewards[indices2] = rewards_t[num_part1:]
            self.dones[indices2] = dones_t[num_part1:]
            self.next_states[indices2] = next_states_t[num_part1:]

        # Update pointers
        self._p = (p + batch_size) % self.buffer_size
        self._n = min(self._n + batch_size, self.buffer_size)

    def save(self, path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        torch.save(
            {
                "state": self.states.clone().cpu(),
                "action": self.actions.clone().cpu(),
                "reward": self.rewards.clone().cpu(),
                "done": self.dones.clone().cpu(),
                "next_state": self.next_states.clone().cpu(),
            },
            path,
        )


class RolloutBuffer:

    def __init__(self, buffer_size, state_shape, action_shape, device, mix=1):
        self._n = 0
        self._p = 0
        self.mix = mix
        self.buffer_size = buffer_size
        self.total_size = mix * buffer_size

        self.states = torch.empty(
            (self.total_size, *state_shape), dtype=torch.float, device=device
        )
        self.actions = torch.empty(
            (self.total_size, *action_shape), dtype=torch.float, device=device
        )
        self.rewards = torch.empty(
            (self.total_size, 1), dtype=torch.float, device=device
        )
        self.dones = torch.empty((self.total_size, 1), dtype=torch.float, device=device)
        self.log_pis = torch.empty(
            (self.total_size, 1), dtype=torch.float, device=device
        )
        self.next_states = torch.empty(
            (self.total_size, *state_shape), dtype=torch.float, device=device
        )

    def append(self, state, action, reward, done, log_pi, next_state):
        self.states[self._p].copy_(torch.from_numpy(state))
        self.actions[self._p].copy_(torch.from_numpy(action))
        self.rewards[self._p] = float(reward)
        self.dones[self._p] = float(done)
        self.log_pis[self._p] = float(log_pi)
        self.next_states[self._p].copy_(torch.from_numpy(next_state))

        self._p = (self._p + 1) % self.total_size
        self._n = min(self._n + 1, self.total_size)

    def get(self):
        assert self._p % self.buffer_size == 0
        start = (self._p - self.buffer_size) % self.total_size
        idxes = slice(start, start + self.buffer_size)
        return (
            self.states[idxes],
            self.actions[idxes],
            self.rewards[idxes],
            self.dones[idxes],
            self.log_pis[idxes],
            self.next_states[idxes],
        )

    def sample(self, batch_size):
        assert self._p % self.buffer_size == 0
        idxes = np.random.randint(low=0, high=self._n, size=batch_size)
        return (
            self.states[idxes],
            self.actions[idxes],
            self.rewards[idxes],
            self.dones[idxes],
            self.log_pis[idxes],
            self.next_states[idxes],
        )


class AIRLReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_space,
        action_space,
        reward_model,
        use_actions_disc: bool,
        device: str = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
        gamma: float = 0.99,
        normalize_reward: bool = True,
        reward_norm_epsilon: float = 1e-8,
    ):
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device,
            n_envs,
            optimize_memory_usage,
            handle_timeout_termination,
        )
        self.reward_model = reward_model
        self.reward_model.to(self.device)

        self.gamma = gamma
        self.normalize_reward = normalize_reward
        self.reward_epsilon = reward_norm_epsilon
        self.use_actions_disc = use_actions_disc

        if self.normalize_reward:
            print("Initializing RunningMeanStd for reward normalization.")
            # Initialize RunningMeanStd for scalar rewards (shape=())
            self.reward_rms = RunningMeanStd(shape=(), epsilon=reward_norm_epsilon)
        else:
            self.reward_rms = None  # Explicitly set to None if not used

    @torch.no_grad()  # Ensure no gradients are computed for reward calculation
    def predict_batch_rewards(self, obs, dones, next_obs):
        """
        Helper method to predict rewards using the AIRLDiscrim model.
        Handles device placement and reward calculation logic.
        Optionally normalizes rewards using RunningMeanStd.
        """
        # Ensure data is on the same device as the model
        obs = obs.to(torch.float32)
        next_obs = next_obs.to(torch.float32)
        dones = dones.to(torch.float32)

        # Calculate raw AIRL rewards
        if self.use_actions_disc:
            rewards_tensor = self.reward_model.f(obs, dones, next_obs)
        else:
            rewards_tensor = self.reward_model.g(obs)

        # Ensure rewards are shaped correctly (batch_size,) before normalization
        rewards_tensor = rewards_tensor.squeeze()
        if rewards_tensor.ndim == 0:  # Handle case where batch_size might be 1
            rewards_tensor = rewards_tensor.unsqueeze(0)

        if self.normalize_reward and self.reward_rms is not None:
            # 1. Convert rewards tensor to numpy array for update
            # Use detach() in case tensor requires grad, though no_grad should prevent it
            batch_rewards_np = rewards_tensor.detach().cpu().numpy()

            # 2. Update RunningMeanStd statistics
            self.reward_rms.update(batch_rewards_np)

            # 3. Normalize the original tensor using the updated stats
            # Convert mean and std back to tensors on the correct device
            mean = torch.tensor(
                self.reward_rms.mean,
                dtype=rewards_tensor.dtype,
                device=rewards_tensor.device,
            )
            # Ensure std is calculated safely (using epsilon)
            std_val = np.sqrt(
                self.reward_rms.var + self.reward_epsilon
            )  # Use stored epsilon
            std = torch.tensor(
                std_val, dtype=rewards_tensor.dtype, device=rewards_tensor.device
            )

            # Apply normalization: (x - mean) / std
            rewards_tensor = (rewards_tensor - mean) / std

        # Reshape to (batch_size, 1) as expected by SB3 samples
        return rewards_tensor.reshape(-1, 1)

    def sample(self, batch_size: int, env=None) -> ReplayBufferSamples:
        # Get the original samples from the env interactions
        original_samples = super().sample(batch_size=batch_size, env=env)

        # Extract information from original samples
        obs = original_samples.observations
        actions = original_samples.actions
        next_obs = original_samples.next_observations
        dones = original_samples.dones

        # Query reward model in batch
        airl_rewards = self.predict_batch_rewards(obs, dones, next_obs)

        # Create new ReplayBufferSamples with AIRL reward
        modified_samples = ReplayBufferSamples(
            obs, actions, next_obs, dones, airl_rewards
        )

        return modified_samples
