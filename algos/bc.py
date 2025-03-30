import torch
from torch import nn
from torch.optim import Adam
import os

from algos.base import Algorithm
from network import StateIndependentPolicy


class BC(Algorithm):
    def __init__(
        self,
        buffer_exp,
        state_shape,
        action_shape,
        device,
        seed,
        gamma=None,
        needs_env=False,
        units_actor=(64, 64),
        lr_actor=3e-4,
        batch_size=128,
        epochs=10,
        rollout_length=None,
    ):
        super().__init__(state_shape, action_shape, device, seed, gamma, needs_env)

        self.actor = StateIndependentPolicy(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_units=units_actor,
        ).to(device)

        self.optim = Adam(self.actor.parameters(), lr=lr_actor)
        self.criterion = nn.MSELoss()

        self.batch_size = batch_size
        self.epochs = epochs
        self.buffer_exp = buffer_exp

    def is_update(self, step):
        """
        Returns whether or not it's time to update the actor
        """
        return True

    def update(self, writer):
        """This will update the actor"""
        for _ in range(self.epochs):
            states, actions, _, dones, next_states = self.buffer_exp.sample(
                self.batch_size
            )

            pred_actions = self.actor(states)
            loss = self.criterion(pred_actions, actions)

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

    def save_models(self, save_dir):
        """
        Save all models
        """
        # Make sure that directory is created
        super().save_models(save_dir)

        # Save policy
        torch.save(self.actor.state_dict(), os.path.join(save_dir, "policy.pth"))
