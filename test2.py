import wandb
from datetime import datetime

api = wandb.Api(timeout=60)

runs = api.run(path="nvanutrecht/Honors Capstone/l51x11zj")
time = datetime.strptime(runs.created_at, "%Y-%m-%dT%H:%M:%SZ")
print(datetime.astimezone(time, "US/Central"))
