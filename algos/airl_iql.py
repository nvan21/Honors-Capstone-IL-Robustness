import torch

from .airl import AIRL
from .iql import IQL
from utils.utils import soft_update


class AIRLIQL(AIRL, IQL):
    """
    AIRL algorithm that uses IQL as the policy optimizer.
    Inherits from both AIRL (for IRL functionality) and IQL (for policy optimization).
    """

    def __init__(
        self,
        buffer_exp,
        state_shape,
        action_shape,
        device,
        seed,
        rollout_length,
        gamma=0.99,
        tau=0.005,
        expectile=0.7,
        temperature=3.0,
        batch_size=256,
        lr_actor=3e-4,
        lr_critic=3e-4,
        lr_value=3e-4,
        lr_disc=3e-4,
        hidden_units_actor=(256, 256),
        hidden_units_critic=(256, 256),
        hidden_units_value=(256, 256),
        units_disc_r=(100, 100),
        units_disc_v=(100, 100),
        update_freq=1,
        epoch_disc=10,
        max_grad_norm=10.0,
    ):
        # Initialize AIRL components (discriminator, expert buffer, etc.)
        AIRL.__init__(
            self,
            buffer_exp=buffer_exp,
            state_shape=state_shape,
            action_shape=action_shape,
            device=device,
            batch_size=batch_size,
            lr_disc=lr_disc,
            units_disc_r=units_disc_r,
            units_disc_v=units_disc_v,
            epoch_disc=epoch_disc,
            gamma=gamma,
        )

        # Initialize IQL components (actor, critic, value, etc.)
        IQL.__init__(
            self,
            state_shape=state_shape,
            action_shape=action_shape,
            device=device,
            seed=seed,
            gamma=gamma,
            tau=tau,
            expectile=expectile,
            temperature=temperature,
            batch_size=batch_size,
            lr_actor=lr_actor,
            lr_critic=lr_critic,
            lr_value=lr_value,
            hidden_units_actor=hidden_units_actor,
            hidden_units_critic=hidden_units_critic,
            hidden_units_value=hidden_units_value,
            update_freq=update_freq,
            max_grad_norm=max_grad_norm,
            needs_env=False,  # IQL is an offline algorithm
        )

    def update(self, buffer, writer):
        """
        Update both the discriminator and the policy.
        This combines AIRL and IQL update logic.
        """
        self.learning_steps += 1

        # Sample from the replay buffer and the expert buffer
        states, actions, _, dones, next_states = buffer.sample(self.batch_size)
        states_exp, actions_exp, _, dones_exp, next_states_exp = self.buffer_exp.sample(
            self.batch_size
        )

        # Calculate log probabilities for both current policy data and expert data
        with torch.no_grad():
            log_pis = self._calculate_log_probs(states, actions)
            log_pis_exp = self._calculate_log_probs(states_exp, actions_exp)

        # Update discriminator
        for _ in range(self.epoch_disc):
            self.learning_steps_disc += 1

            # Update discriminator (method from AIRL parent class)
            self.update_disc(
                states,
                dones,
                log_pis,
                next_states,
                states_exp,
                dones_exp,
                log_pis_exp,
                next_states_exp,
                writer,
            )

        # Calculate rewards using the discriminator (method from AIRL parent class)
        rewards = self.calculate_rewards(states, dones, log_pis, next_states)

        # Update IQL using estimated rewards
        self.update_value(states, actions, writer)
        self.update_critic(states, actions, rewards, dones, next_states, writer)
        self.update_actor(states, actions, writer)

        # Update target networks
        soft_update(self.critic_target, self.critic, self.tau)

    def _calculate_log_probs(self, states, actions):
        """
        Calculate log probabilities of actions under the current policy.
        This is a helper method for the discriminator update.
        """
        # Get distribution parameters
        means, log_stds = self.actor.net(states).chunk(2, dim=-1)
        log_stds = log_stds.clamp(-20, 2)
        stds = torch.exp(log_stds)

        # Calculate normalized actions
        noise = (actions - torch.tanh(means)) / (stds + 1e-8)

        # Calculate log probabilities
        log_probs = (-0.5 * noise.pow(2) - log_stds).sum(
            dim=-1, keepdim=True
        ) - torch.log(1 - actions.pow(2) + 1e-6).sum(dim=-1, keepdim=True)

        return log_probs

    def save_models(self, save_dir):
        """
        Save all models (IQL models and discriminator).
        """
        # Save IQL models (from IQL parent class)
        IQL.save_models(self, save_dir=save_dir)

        # Save discriminator (from AIRL parent class)
        self.save_discriminator(save_dir)
