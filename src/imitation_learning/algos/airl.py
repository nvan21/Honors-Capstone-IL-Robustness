import os

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam

from imitation_learning.network import AIRLDiscrim

from .ppo import PPO


class AIRL:
    """
    Base class for AIRL (Adversarial Inverse Reinforcement Learning) algorithm.
    This class contains common IRL functionality to be used with different policy optimizers.

    To use this class, create a new class that inherits from both AIRL and a policy optimizer:

    class AIRLPPO(AIRL, PPO):
        def __init__(self, buffer_exp, state_shape, action_shape, **kwargs):
            # Initialize both parent classes
            AIRL.__init__(self, buffer_exp, state_shape, action_shape, **kwargs)
            PPO.__init__(self, state_shape, action_shape, **kwargs)

        # Override methods as needed
    """

    def __init__(
        self,
        buffer_exp,
        state_shape,
        action_shape,
        device,
        batch_size=64,
        lr_disc=3e-4,
        units_disc_r=(100, 100),
        units_disc_v=(100, 100),
        epoch_disc=10,
        gamma=0.995,
        **kwargs,  # Accept additional kwargs for other base classes
    ):
        # Expert's buffer
        self.buffer_exp = buffer_exp

        # Store common parameters
        self.device = device
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.gamma = gamma

        # Discriminator
        self.disc = AIRLDiscrim(
            state_shape=state_shape,
            gamma=gamma,
            hidden_units_r=units_disc_r,
            hidden_units_v=units_disc_v,
            hidden_activation_r=nn.ReLU(inplace=True),
            hidden_activation_v=nn.ReLU(inplace=True),
        ).to(device)

        # AIRL-specific parameters
        self.learning_steps_disc = 0
        self.optim_disc = Adam(self.disc.parameters(), lr=lr_disc)
        self.batch_size = batch_size
        self.epoch_disc = epoch_disc
        self.needs_env = True

    def update_disc(
        self,
        states,
        dones,
        log_pis,
        next_states,
        states_exp,
        dones_exp,
        log_pis_exp,
        next_states_exp,
        writer,
    ):
        """
        Update the discriminator using samples from current policy and expert demonstrations.
        """
        # Output of discriminator is (-inf, inf), not [0, 1]
        logits_pi = self.disc(states, dones, log_pis, next_states)
        logits_exp = self.disc(states_exp, dones_exp, log_pis_exp, next_states_exp)

        # Discriminator is to maximize E_{\pi} [log(1 - D)] + E_{exp} [log(D)]
        loss_pi = -F.logsigmoid(-logits_pi).mean()
        loss_exp = -F.logsigmoid(logits_exp).mean()
        loss_disc = loss_pi + loss_exp

        self.optim_disc.zero_grad()
        loss_disc.backward()
        self.optim_disc.step()

        if self.learning_steps_disc % self.epoch_disc == 0:
            writer.log({"loss/disc": loss_disc.item()})

            # Discriminator's accuracies
            with torch.no_grad():
                acc_pi = (logits_pi < 0).float().mean().item()
                acc_exp = (logits_exp > 0).float().mean().item()
            writer.log({"stats/acc_pi": acc_pi})
            writer.log({"stats/acc_exp": acc_exp})

    def calculate_rewards(self, states, dones, log_pis, next_states):
        """
        Calculate rewards using the discriminator.
        """

        return self.disc.calculate_reward(states, dones, log_pis, next_states)

    def save_discriminator(self, save_dir):
        """
        Save the discriminator model.
        """
        torch.save(self.disc.state_dict(), os.path.join(save_dir, "disc.pth"))
