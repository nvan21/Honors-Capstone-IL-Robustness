import pandas as pd
import os

best_runs = pd.read_csv("runs/best_runs_per_tag.csv")

run_config = {}
all_envs = {
    env: sorted(xml_list.tolist())
    for env, xml_list in best_runs.groupby("env_id")["xml_file"].unique().items()
}


def make_base_model_path(run_name: str, created_at: str, num_steps: str):
    run_name_split = run_name.split("-")
    algo_name = run_name_split[0]
    run_id = f"{run_name_split[-2]}-{run_name_split[-1]}-{created_at}"
    print(run_id)
    path = os.path.join(".", "logs", run_id, "model", f"step{num_steps}")


def make_sb3_model_path(run_name: str):
    path = os.path.join(".", "logs", "sb3", run_name, "best_model", "best_model.zip")

    return path


for index, row in best_runs.iterrows():
    if "AND" in row["tag_specification"]:
        model_path = make_base_model_path(
            run_name=row["run_name"],
            created_at=row["created_at_str"],
            num_steps=row["Steps"],
        )
        run_config[model_path] = all_envs
    else:
        model_path = make_sb3_model_path(row["run_name"])
        run_config[row["run_name"]]
