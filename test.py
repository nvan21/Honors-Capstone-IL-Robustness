import pandas as pd
import os

best_runs = pd.read_csv("runs/best_runs_per_tag.csv")

# Remove the rows that have AND
col = "tag_specification"
value = "AND"
del_condition = best_runs[col].str.contains("AND", case=True, na=False)
best_runs = best_runs[~del_condition]

rq = {"environments": {}}
envs = set(best_runs["env_id"])
configs = {}

for env in envs:
    env_df = best_runs[best_runs["env_id"] == env]
    xml_files = env_df["xml_file"].to_list()
    model_ids = env_df["run_id"].to_list()
    model_paths = [
        os.path.join(".", "logs", "sb3", model_id, "best_model", "best_model.zip")
        for model_id in model_ids
    ]
    print(model_paths)
    print(xml_files)
    rq["environments"][env] = {}
