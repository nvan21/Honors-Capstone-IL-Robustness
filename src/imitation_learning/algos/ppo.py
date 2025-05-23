import torch
from torch import nn
from torch.optim import Adam
import os

from imitation_learning.algos.base import Algorithm
from imitation_learning.utils.buffer import RolloutBuffer
from imitation_learning.utils.utils import (
    disable_gradient,
    get_hidden_units_from_state_dict,
)
from imitation_learning.network import StateIndependentPolicy, StateFunction


def calculate_gae(values, rewards, dones, next_values, gamma, lambd):
    # Calculate TD errors.
    deltas = rewards + gamma * next_values * (1 - dones) - values
    # Initialize gae.
    gaes = torch.empty_like(rewards)

    # Calculate gae recursively from behind.
    gaes[-1] = deltas[-1]
    for t in reversed(range(rewards.size(0) - 1)):
        gaes[t] = deltas[t] + gamma * lambd * (1 - dones[t]) * gaes[t + 1]

    return gaes + values, (gaes - gaes.mean()) / (gaes.std() + 1e-8)


class PPO(Algorithm):

    def __init__(
        self,
        state_shape,
        action_shape,
        device,
        seed,
        gamma=0.995,
        rollout_length=2048,
        mix_buffer=20,
        lr_actor=3e-4,
        lr_critic=3e-4,
        units_actor=(256, 256),
        units_critic=(256, 256),
        epoch_ppo=10,
        clip_eps=0.2,
        lambd=0.97,
        coef_ent=0.0,
        max_grad_norm=10.0,
        needs_env=True,
        use_reward_model=False,
    ):
        super().__init__(state_shape, action_shape, device, seed, gamma, needs_env)

        # Rollout buffer. Mix 20 policies as in the AIRL paper
        self.buffer = RolloutBuffer(
            buffer_size=rollout_length,
            state_shape=state_shape,
            action_shape=action_shape,
            device=device,
            mix=mix_buffer,
        )

        # Actor.
        self.actor = StateIndependentPolicy(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_units=units_actor,
            hidden_activation=nn.Tanh(),
        ).to(device)

        # Critic.
        self.critic = StateFunction(
            state_shape=state_shape,
            hidden_units=units_critic,
            hidden_activation=nn.Tanh(),
        ).to(device)

        self.optim_actor = Adam(self.actor.parameters(), lr=lr_actor)
        self.optim_critic = Adam(self.critic.parameters(), lr=lr_critic)

        self.learning_steps_ppo = 0
        self.rollout_length = rollout_length
        self.epoch_ppo = epoch_ppo
        self.clip_eps = clip_eps
        self.lambd = lambd
        self.coef_ent = coef_ent
        self.max_grad_norm = max_grad_norm
        self.needs_env = True
        self.use_reward_model = use_reward_model

    def is_update(self, step):
        return step % self.rollout_length == 0

    def step(self, env, state, t, step):
        t += 1

        action, log_pi = self.explore(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        mask = False if truncated else terminated

        self.buffer.append(state, action, reward, mask, log_pi, next_state)

        if terminated or truncated:
            t = 0
            next_state, _ = env.reset()

        return next_state, t

    def update(self, writer, env):
        self.learning_steps += 1
        states, actions, rewards, dones, log_pis, next_states = self.buffer.get()
        if self.use_reward_model:
            rewards = env.calc_rewards(states, dones, next_states)
        self.update_ppo(states, actions, rewards, dones, log_pis, next_states, writer)

    def update_ppo(self, states, actions, rewards, dones, log_pis, next_states, writer):
        with torch.no_grad():
            values = self.critic(states)
            next_values = self.critic(next_states)

        targets, gaes = calculate_gae(
            values, rewards, dones, next_values, self.gamma, self.lambd
        )

        for _ in range(self.epoch_ppo):
            self.learning_steps_ppo += 1
            self.update_critic(states, targets, writer)
            self.update_actor(states, actions, log_pis, gaes, writer)

    def update_critic(self, states, targets, writer):
        loss_critic = (self.critic(states) - targets).pow_(2).mean()

        self.optim_critic.zero_grad()
        loss_critic.backward(retain_graph=False)
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.optim_critic.step()

        if self.learning_steps_ppo % self.epoch_ppo == 0:
            writer.log({"loss/critic": loss_critic.item()})

    def update_actor(self, states, actions, log_pis_old, gaes, writer):
        log_pis = self.actor.evaluate_log_pi(states, actions)
        entropy = -log_pis.mean()

        ratios = (log_pis - log_pis_old).exp_()
        loss_actor1 = -ratios * gaes
        loss_actor2 = (
            -torch.clamp(ratios, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * gaes
        )
        loss_actor = torch.max(loss_actor1, loss_actor2).mean()

        self.optim_actor.zero_grad()
        (loss_actor - self.coef_ent * entropy).backward(retain_graph=False)
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.optim_actor.step()

        if self.learning_steps_ppo % self.epoch_ppo == 0:
            writer.log({"loss/actor": loss_actor.item()})
            writer.log({"stats/entropy": entropy.item()})

    def save_models(self, save_dir):
        # Make sure that directory is created
        super().save_models(save_dir)

        # Save actor and critic models
        torch.save(self.actor.state_dict(), os.path.join(save_dir, "actor.pth"))
        torch.save(self.critic.state_dict(), os.path.join(save_dir, "critic.pth"))


class PPOExpert(PPO):

    def __init__(
        self,
        state_shape,
        action_shape,
        device,
        path,
        units_actor=(64, 64),
        units_critic=(64, 64),
    ):
        actor_path = os.path.join(path, "actor.pth")
        critic_path = os.path.join(path, "critic.pth")
        units_actor = get_hidden_units_from_state_dict(actor_path)["net"]
        units_critic = get_hidden_units_from_state_dict(critic_path)["net"]

        self.actor = StateIndependentPolicy(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_units=units_actor,
            hidden_activation=nn.Tanh(),
        ).to(device)

        self.critic = StateFunction(
            state_shape=state_shape,
            hidden_units=units_critic,
            hidden_activation=nn.Tanh(),
        ).to(device)

        self.actor.load_state_dict(torch.load(actor_path))
        self.critic.load_state_dict(torch.load(critic_path))

        disable_gradient(self.actor)
        self.device = device
