import torch
import os

from .airl import AIRL
from .ppo import PPO


class AIRLPPO(AIRL, PPO):
    """
    AIRL algorithm that uses PPO as the policy optimizer.
    Inherits from both AIRL (for IRL functionality) and PPO (for policy optimization).
    """

    def __init__(
        self,
        buffer_exp,
        state_shape,
        action_shape,
        device,
        seed,
        gamma=0.995,
        rollout_length=10000,
        mix_buffer=1,
        batch_size=64,
        lr_actor=3e-4,
        lr_critic=3e-4,
        lr_disc=3e-4,
        units_actor=(256, 256),
        units_critic=(256, 256),
        units_disc_r=(100, 100),
        units_disc_v=(100, 100),
        epoch_ppo=50,
        epoch_disc=20,
        clip_eps=0.2,
        lambd=0.97,
        coef_ent=0.0,
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

        # Initialize PPO components (actor, critic, etc.)
        PPO.__init__(
            self,
            state_shape=state_shape,
            action_shape=action_shape,
            device=device,
            seed=seed,
            gamma=gamma,
            rollout_length=rollout_length,
            mix_buffer=mix_buffer,
            lr_actor=lr_actor,
            lr_critic=lr_critic,
            units_actor=units_actor,
            units_critic=units_critic,
            epoch_ppo=epoch_ppo,
            clip_eps=clip_eps,
            lambd=lambd,
            coef_ent=coef_ent,
            max_grad_norm=max_grad_norm,
            needs_env=True,
        )

    def update(self, writer):
        """
        Update both the discriminator and the policy.
        This combines AIRL and PPO update logic.
        """
        self.learning_steps += 1

        # Update discriminator
        for _ in range(self.epoch_disc):
            self.learning_steps_disc += 1

            # Samples from current policy's trajectories
            states, _, _, dones, log_pis, next_states = self.buffer.sample(
                self.batch_size
            )
            # Samples from expert's demonstrations
            states_exp, actions_exp, _, dones_exp, next_states_exp = (
                self.buffer_exp.sample(self.batch_size)
            )
            # Calculate log probabilities of expert actions
            with torch.no_grad():
                log_pis_exp = self.actor.evaluate_log_pi(states_exp, actions_exp)

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

        # Get trajectory data
        states, actions, _, dones, log_pis, next_states = self.buffer.get()

        # Calculate rewards using the discriminator (method from AIRL parent class)
        rewards = self.calculate_rewards(states, dones, log_pis, next_states)

        # Update PPO using estimated rewards (method from PPO parent class)
        self.update_ppo(states, actions, rewards, dones, log_pis, next_states, writer)

    def save_models(self, save_dir):
        """
        Save all models (PPO models and discriminator).
        """
        # Save PPO models (from PPO parent class)
        PPO.save_models(self, save_dir=save_dir)

        # Save discriminator (from AIRL parent class)
        self.save_discriminator(save_dir)
