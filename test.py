import pandas as pd
import os
import yaml

best_runs = pd.read_csv("runs/best_runs_per_tag.csv")

run_config = {}
all_envs = {
    env: sorted(xml_list.tolist())
    for env, xml_list in best_runs.groupby("env_id")["xml_file"].unique().items()
}


def make_base_model_path(run_name: str, created_at: str, num_steps: str, env: str):
    run_name_split = run_name.split("-")
    algo_name = run_name_split[0]
    run_id = f"{run_name_split[-2]}-{run_name_split[-1]}-{created_at}"
    path = os.path.join(
        ".", "logs", env, algo_name, run_id, "model", f"step{num_steps}"
    )

    return path


def make_sb3_model_path(run_name: str):
    path = os.path.join(".", "logs", "sb3", run_name, "best_model", "best_model.zip")

    return path


for index, row in best_runs.iterrows():
    if "AND" in row["tag_specification"]:
        num_steps = row["num_steps"] if row["num_steps"] != 0 else row["num_epochs"]
        model_path = make_base_model_path(
            run_name=row["run_name"],
            created_at=row["created_at_str"],
            num_steps=num_steps,
            env=row["env_id"],
        )
        run_config[model_path] = all_envs
    else:
        model_path = make_sb3_model_path(row["run_id"])
        run_config[model_path] = {row["env_id"]: [row["xml_file"]]}

# Manual addition of random SAC experts because I'm lazy
run_config["./experts/ant-v5-sac-expert.zip"] = all_envs
run_config["./experts/hopper-v5-SAC-expert.zip"] = all_envs
run_config["./experts/pusher-v5-SAC-expert.zip"] = all_envs
run_config[
    "./logs/InvertedPendulum-v5/sac/normal_env-seed0-20250412-1007/model/step250000"
] = all_envs

with open("test.yaml", "w") as f:
    yaml.safe_dump(run_config, f)

with open("test.yaml", "r") as f:
    config = yaml.safe_load(f)

for path, envs in config.items():
    for env, xml_files in envs.items():
        for xml_file in xml_files:
            print(f"--weights {path} --env {env} --xml_file {xml_file}")
