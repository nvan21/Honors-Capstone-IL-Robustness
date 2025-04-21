import yaml
import argparse
import subprocess


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rq", type=str, default="1234")
    p.add_argument("--num_eval_episodes", type=int, default=100)

    return p.parse_args()


def run():
    args = get_args()
    rqs = list(args.rq)

    for rq in rqs:
        rq_config_path = f"./experiments/RQ{rq}.yaml"

        with open(rq_config_path, "r") as f:
            experiments = yaml.safe_load(f)

        for env, config in experiments["environments"].items():
            xml_file = config["xml_file"]

            for expert in config["experts"].values():
                command = [
                    "python",
                    "scripts/visualize_expert.py",
                    "--weights",
                    expert,
                    "--env",
                    env,
                    "--xml_file",
                    xml_file,
                    "--num_eval_episodes",
                    str(args.num_eval_episodes),
                    "--log",
                ]
                subprocess.Popen(command)


if __name__ == "__main__":
    run()
