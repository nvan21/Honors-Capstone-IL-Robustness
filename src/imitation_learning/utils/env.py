import gymnasium as gym
from gymnasium.wrappers import RescaleAction, NormalizeObservation
import torch
import numpy as np
from copy import copy

from imitation_learning.utils.utils import disable_gradient


def make_env(env_id, **kwargs):
    env = gym.make(env_id, **kwargs)

    scale = env.action_space.high
    min_action = env.action_space.low / scale
    max_action = env.action_space.high / scale

    return RescaleAction(env, min_action=min_action, max_action=max_action)


def make_custom_reward_env(env, reward_model, device, normalize_reward):

    return AIRLRewardWrapper(
        env, reward_model=reward_model, device=device, normalize_reward=normalize_reward
    )


class AIRLRewardWrapper(gym.Wrapper):
    def __init__(self, env, reward_model, device, normalize_reward):
        super().__init__(env)

        self.reward_model = reward_model
        self.reward_model.eval()
        disable_gradient(self.reward_model)

        self.device = device
        self.normalize_reward = normalize_reward
        self.epsilon = 1e-8
        self.clip_reward = 10.0
        if self.normalize_reward:
            self.reward_rms = RunningMeanStd(shape=())

    def calc_rewards(self, states, dones, next_states):
        with torch.no_grad():
            rewards = self.reward_model.f(states, dones, next_states)

            if self.normalize_reward:
                # Ensure reward tensor is correctly shaped for normalization stats (e.g., (batch_size,))
                if rewards.ndim > 1:
                    rewards = rewards.squeeze()  # Make it (batch_size,) if needed

                # 2. Update Running Statistics (Tensor-based)
                if rewards.numel() > 0:  # Check if tensor is not empty
                    self.reward_rms.update(rewards)

                # 3. Normalize the Rewards Tensor
                # Use the normalize method which handles device/dtype and clipping
                normalized_rewards_tensor = self.reward_rms.normalize(
                    rewards, clip_range=self.clip_reward
                )
                # Ensure it has the correct shape for buffer/loss (e.g., add feature dim back)
                if normalized_rewards_tensor.ndim == 1:
                    normalized_rewards_tensor = normalized_rewards_tensor.unsqueeze(
                        1
                    )  # Reshape to (batch_size, 1)

                rewards = normalized_rewards_tensor

        return rewards


class RunningMeanStd:
    def __init__(self, epsilon=1e-8, shape=(), device=None):
        """
        Tracks running mean and variance for tensors.

        Args:
            epsilon (float): Prevents division by zero.
            shape (tuple): Shape of the statistics (e.g., () for scalar).
            device (torch.device): Device where tensors should be stored.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.mean = torch.zeros(shape, dtype=torch.float64, device=self.device)
        self.var = torch.ones(shape, dtype=torch.float64, device=self.device)
        self.count = torch.tensor(epsilon, dtype=torch.float64, device=self.device)
        self.epsilon = torch.tensor(epsilon, dtype=torch.float64, device=self.device)

    @torch.no_grad()  # Ensure no gradients are computed within update
    def update(self, x_tensor):
        """
        Update running stats with a batch of tensor data.

        Args:
            x_tensor (torch.Tensor): Input tensor batch (e.g., rewards).
                                     Should have shape (batch_size, *shape).
        """
        # Ensure input is on the correct device and dtype
        x_tensor = x_tensor.to(device=self.device, dtype=torch.float64)

        # Calculate batch statistics
        batch_mean = torch.mean(x_tensor, dim=0)  # Mean across batch dim
        batch_var = torch.var(
            x_tensor, dim=0, unbiased=False
        )  # Use population variance (N)
        batch_count = torch.tensor(
            x_tensor.shape[0], dtype=torch.float64, device=self.device
        )

        if batch_count == 0:
            return

        # Welford's algorithm update (tensorized)
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + torch.square(delta) * self.count * batch_count / tot_count
        new_var = m_2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    @property
    def std(self):
        # Ensure variance doesn't become negative due to floating point issues
        return torch.sqrt(
            torch.max(self.var, self.epsilon)
        )  # Use epsilon floor for stability

    def normalize(self, x_tensor, clip_range=None):
        """
        Normalizes a tensor using the running statistics.

        Args:
            x_tensor (torch.Tensor): Input tensor to normalize.
            clip_range (Optional[tuple or float]): If provided, clips normalized
                                                 values to [-clip_range, clip_range]
                                                 or [clip_range[0], clip_range[1]].

        Returns:
            torch.Tensor: Normalized tensor with the same dtype as input.
        """
        input_dtype = x_tensor.dtype
        # Ensure input is on the correct device for calculation
        x_tensor = x_tensor.to(device=self.device)

        # Normalize: Use float32 for stability in division if needed, then cast back
        normalized_tensor = (x_tensor - self.mean.to(input_dtype)) / (
            self.std.to(input_dtype) + self.epsilon.to(input_dtype)
        )

        # Clip if requested
        if clip_range is not None:
            if isinstance(clip_range, (int, float)):
                clip_min = -abs(clip_range)
                clip_max = abs(clip_range)
            else:  # Assume tuple/list [min, max]
                clip_min, clip_max = clip_range
            normalized_tensor = torch.clamp(
                normalized_tensor, min=clip_min, max=clip_max
            )

        return normalized_tensor.to(input_dtype)  # Return with original dtype
