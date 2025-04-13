import torch
from torch import nn
from torch.optim import Adam
import os
from tqdm import tqdm

from imitation_learning.algos.base import Algorithm
from imitation_learning.network import StateIndependentPolicy
from imitation_learning.utils.utils import (
    disable_gradient,
    get_hidden_units_from_state_dict,
)


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
        units_actor=(256, 256),
        lr_actor=3e-4,
        batch_size=128,
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
        self.steps_per_epoch = len(buffer_exp) // batch_size
        self.buffer_exp = buffer_exp

    def is_update(self, step):
        """
        Returns whether or not it's time to update the actor
        """
        return True

    def update(self, writer):
        """This will update the actor"""
        for _ in tqdm(range(self.steps_per_epoch)):
            # BC only needs states and actions
            states, actions, _, _, _ = self.buffer_exp.sample(self.batch_size)

            pred_actions = self.actor(states)
            loss = self.criterion(pred_actions, actions)

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            writer.log({"loss/actor": loss.item()})

    def save_models(self, save_dir):
        """
        Save all models
        """
        # Make sure that directory is created
        super().save_models(save_dir)

        # Save policy
        torch.save(self.actor.state_dict(), os.path.join(save_dir, "actor.pth"))


class BCExpert(BC):

    def __init__(
        self,
        state_shape,
        action_shape,
        device,
        path,
        units_actor=(256, 256),
    ):
        actor_path = os.path.join(path, "actor.pth")
        units_actor = get_hidden_units_from_state_dict(actor_path)

        self.actor = StateIndependentPolicy(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_units=units_actor,
            hidden_activation=nn.Tanh(),
        ).to(device)

        self.actor.load_state_dict(torch.load(actor_path))

        disable_gradient(self.actor)
        self.device = device
