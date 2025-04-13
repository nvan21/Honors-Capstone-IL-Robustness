import wandb
import yaml
import subprocess
import time

num_agents = 10

sweep_config = yaml.safe_load("./hyperparameters/airl_hopper_sweep.yaml")
sweep_id = wandb.sweep(sweep=sweep_config, project="Honors Capstone")
sweep_path = f"nvanutrecht/Honors-Capstone-scripts/{sweep_id}"

for i in range(num_agents):
    command = ["wandb", "agent", sweep_path]

    process = subprocess.Popen(command)

    if i < num_agents - 1:
        time.sleep(2)
