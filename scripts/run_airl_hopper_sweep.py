import wandb
import yaml
import subprocess
import time

num_agents = 10

config_path = "./hyperparameters/airl_hopper_sweep.yaml"
with open(config_path, "r") as f:
    sweep_config = yaml.safe_load(f)

sweep_id = wandb.sweep(sweep=sweep_config, project="Honors Capstone")
sweep_path = f"nvanutrecht/Honors Capstone/{sweep_id}"

for i in range(num_agents):
    command = ["wandb", "agent", sweep_path]

    process = subprocess.Popen(command)

    if i < num_agents - 1:
        time.sleep(2)
