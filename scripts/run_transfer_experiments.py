# Example: parse_yaml_and_run.py
import yaml
import subprocess
import sys

YAML_FILE = "./hyperparameters/sac.yaml"
PYTHON_SCRIPT = "./scripts/train_expert.py"


# Environment name mapping based on suffix
def get_env_from_experiment(exp_name):
    if exp_name.endswith("_ant"):
        return "Ant-v5"
    elif exp_name.endswith("_hopper"):
        return "Hopper-v5"
    elif exp_name.endswith("_invpend"):
        return "InvertedPendulum-v5"
    elif exp_name.endswith("_pusher"):
        return "Pusher-v5"
    else:
        print(
            f"Warning: Could not determine environment for experiment '{exp_name}'. Skipping.",
            file=sys.stderr,
        )
        return None


try:
    with open(YAML_FILE, "r") as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    print(f"Error: YAML file not found at '{YAML_FILE}'", file=sys.stderr)
    sys.exit(1)
except yaml.YAMLError as e:
    print(f"Error parsing YAML file: {e}", file=sys.stderr)
    sys.exit(1)


if "experiments" not in config or not isinstance(config["experiments"], dict):
    print(
        f"Error: 'experiments' key not found or is not a dictionary in '{YAML_FILE}'",
        file=sys.stderr,
    )
    sys.exit(1)

print("Starting experiments...")

for experiment_name in config["experiments"]:
    print("-----------------------------------------")
    print(f"Processing experiment: {experiment_name}")

    env_name = get_env_from_experiment(experiment_name)
    if env_name is None:
        continue

    print(f"Determined Environment: {env_name}")

    command = [
        "python",
        PYTHON_SCRIPT,
        "--env",
        env_name,
        "--experiment",
        experiment_name,
    ]

    print(f"Executing: {' '.join(command)}")
    try:
        # Run the command, check=True raises an exception on non-zero exit code
        subprocess.run(command, check=True)
        print(f"Finished experiment: {experiment_name}")
    except subprocess.CalledProcessError as e:
        print(
            f"Error running experiment {experiment_name}: Command failed with exit code {e.returncode}",
            file=sys.stderr,
        )
    except FileNotFoundError:
        print(f"Error: Python script '{PYTHON_SCRIPT}' not found.", file=sys.stderr)
        # Optionally stop all experiments on first error
        # sys.exit(1)

    print("-----------------------------------------")
    print()  # Blank line

print("All experiments processed.")
