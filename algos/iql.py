import torch
import torch.nn.functional as F
from torch.optim import Adam
import os

from algos import Algorithm
from network.value import TwinnedStateActionFunction, StateFunction
from network.policy import StateDependentPolicy
from utils.utils import soft_update, disable_gradient


def expectile_loss(diff, expectile=0.8):
    """
    Asymmetric L2 loss for expectile regression.
    """
    weight = torch.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff**2)


class IQL(Algorithm):
    """
    Implicit Q-Learning implementation.
    """

    def __init__(
        self,
        state_shape,
        action_shape,
        device,
        seed,
        gamma=0.99,
        tau=0.005,
        expectile=0.7,
        temperature=3.0,
        batch_size=256,
        lr_actor=3e-4,
        lr_critic=3e-4,
        lr_value=3e-4,
        hidden_units_actor=(256, 256),
        hidden_units_critic=(256, 256),
        hidden_units_value=(256, 256),
        update_freq=1,
        max_grad_norm=10.0,
        needs_env=False,
    ):
        super().__init__(state_shape, action_shape, device, seed, gamma, needs_env)

        # Networks
        self.critic = TwinnedStateActionFunction(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_units=hidden_units_critic,
        ).to(device)

        self.critic_target = TwinnedStateActionFunction(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_units=hidden_units_critic,
        ).to(device)

        self.value = StateFunction(
            state_shape=state_shape,
            hidden_units=hidden_units_value,
        ).to(device)

        self.actor = StateDependentPolicy(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_units=hidden_units_actor,
        ).to(device)

        # Copy parameters of the critic to the target
        self.critic_target.load_state_dict(self.critic.state_dict())
        disable_gradient(self.critic_target)

        # Optimizers
        self.optim_critic = Adam(self.critic.parameters(), lr=lr_critic)
        self.optim_value = Adam(self.value.parameters(), lr=lr_value)
        self.optim_actor = Adam(self.actor.parameters(), lr=lr_actor)

        # Hyperparameters
        self.tau = tau
        self.expectile = expectile
        self.temperature = temperature
        self.batch_size = batch_size
        self.update_freq = update_freq
        self.max_grad_norm = max_grad_norm

        # Counters
        self.learning_steps_value = 0
        self.learning_steps_critic = 0
        self.learning_steps_actor = 0

    def is_update(self, step):
        return step % self.update_freq == 0

    def update(self, buffer, writer):
        """
        Update the networks using samples from buffer.
        """
        self.learning_steps += 1

        # Sample from the buffer
        states, actions, rewards, dones, next_states = buffer.sample(self.batch_size)

        # Update networks
        self.update_value(states, actions, writer)
        self.update_critic(states, actions, rewards, dones, next_states, writer)
        self.update_actor(states, actions, writer)

        # Update target networks
        soft_update(self.critic_target, self.critic, self.tau)

    def update_value(self, states, actions, writer):
        """
        Update value function through expectile regression.
        """
        self.learning_steps_value += 1

        # Get current value estimates
        values = self.value(states)

        # Get Q-values
        with torch.no_grad():
            q1, q2 = self.critic(states, actions)
            q = torch.min(q1, q2)

        # Calculate expectile loss
        value_loss = expectile_loss(q - values, self.expectile).mean()

        # Update value network
        self.optim_value.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value.parameters(), self.max_grad_norm)
        self.optim_value.step()

        # Log statistics
        if self.learning_steps_value % 100 == 0:
            writer.add_scalar("loss/value", value_loss.item(), self.learning_steps)

    def update_critic(self, states, actions, rewards, dones, next_states, writer):
        """
        Update Q-functions with TD learning.
        """
        self.learning_steps_critic += 1

        # Get current Q-values
        curr_q1, curr_q2 = self.critic(states, actions)

        # Calculate target using value function (no max operation needed)
        with torch.no_grad():
            next_v = self.value(next_states)
            target_q = rewards + (1.0 - dones) * self.gamma * next_v

        # TD errors
        q1_loss = F.mse_loss(curr_q1, target_q)
        q2_loss = F.mse_loss(curr_q2, target_q)
        critic_loss = q1_loss + q2_loss

        # Update critic
        self.optim_critic.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.optim_critic.step()

        # Log statistics
        if self.learning_steps_critic % 100 == 0:
            writer.add_scalar("loss/critic", critic_loss.item(), self.learning_steps)

    def update_actor(self, states, actions, writer):
        """
        Update actor using advantage-weighted regression.
        """
        self.learning_steps_actor += 1

        # Calculate advantages
        with torch.no_grad():
            q1, q2 = self.critic(states, actions)
            q = torch.min(q1, q2)
            v = self.value(states)
            advantages = q - v
            # Calculate weights for AWR
            weights = torch.exp(advantages / self.temperature)

        # Get action distribution from the policy
        means, log_stds = self.actor.net(states).chunk(2, dim=-1)
        log_stds = log_stds.clamp(-20, 2)
        stds = torch.exp(log_stds)

        # Calculate log probabilities
        noise = (actions - torch.tanh(means)) / (torch.tanh(stds) + 1e-8)
        log_probs = (-0.5 * noise.pow(2) - log_stds).sum(
            dim=-1, keepdim=True
        ) - torch.log(1 - actions.pow(2) + 1e-6).sum(dim=-1, keepdim=True)

        # AWR loss: weighted negative log probabilities
        actor_loss = -(weights * log_probs).mean()

        # Update actor
        self.optim_actor.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.optim_actor.step()

        # Log statistics
        if self.learning_steps_actor % 100 == 0:
            writer.add_scalar("loss/actor", actor_loss.item(), self.learning_steps)
            writer.add_scalar(
                "stats/adv_mean", advantages.mean().item(), self.learning_steps
            )
            writer.add_scalar(
                "stats/adv_max", advantages.max().item(), self.learning_steps
            )
            writer.add_scalar(
                "stats/weight_mean", weights.mean().item(), self.learning_steps
            )

    def explore(self, state):
        """
        Get exploratory action and its log probability.
        """
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        with torch.no_grad():
            action, log_pi = self.actor.sample(state.unsqueeze(0))
        return action.cpu().numpy()[0], log_pi.item()

    def exploit(self, state):
        """
        Get deterministic action.
        """
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        with torch.no_grad():
            action = self.actor(state.unsqueeze(0))
        return action.cpu().numpy()[0]

    def save_models(self, save_dir):
        # Make sure directory exists
        super().save_models(save_dir)

        # Save the networks
        torch.save(self.critic.state_dict(), os.path.join(save_dir, "critic.pth"))
        torch.save(
            self.critic_target.state_dict(), os.path.join(save_dir, "critic_target.pth")
        )
        torch.save(self.value.state_dict(), os.path.join(save_dir, "value.pth"))
        torch.save(self.actor.state_dict(), os.path.join(save_dir, "actor.pth"))
