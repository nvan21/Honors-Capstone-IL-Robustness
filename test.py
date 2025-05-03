import gymnasium as gym
import stable_baselines3 as sb
import torch

env = gym.make("Hopper-v5")
obs, info = env.reset()

print(f"Original obs: {obs}")


def probe_dynamics(env: gym.Env, qpos, qvel, action):
    env.unwrapped.set_state(qpos, qvel)

    next_obs, reward, terminated, truncated, info = env.unwrapped.step(action)

    return next_obs, reward, terminated, truncated, info


oracle = sb.SAC.load("./experts/hopper-v5-SAC-expert.zip")
action = oracle.predict(obs)[0]
qpos = env.unwrapped.data.qpos
qvel = env.unwrapped.data.qvel
next_obs, _, ter, trunc, info = probe_dynamics(env, qpos, qvel, action)

obs_tensor = torch.Tensor(obs)
print(f"Next obs: {next_obs}")
print(f"Env obs:")
print(torch.empty_like(obs_tensor))
print(torch.empty(obs_tensor.shape[0]))
