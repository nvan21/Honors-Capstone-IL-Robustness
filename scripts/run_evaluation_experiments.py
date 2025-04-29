import yaml
import argparse
import subprocess
import sys
import os
import concurrent.futures
import time
from typing import Dict, Any, List, Tuple
import pandas as pd

# --- Configuration ---
# Adjust based on your CPU cores and script's resource usage.
# os.cpu_count() is a good starting point, but might need tuning.
# Don't set it too high if scripts are memory-intensive or GPU-bound.
MAX_WORKERS = 4  # Use CPU count or default to 4
# Optional: Add a timeout for each individual script run (in seconds)
TASK_TIMEOUT = None  # 5 minutes, adjust as needed, or set to None for no timeout
SCRIPT_TO_RUN = "scripts/visualize_expert.py"
# --- ------------- ---


def get_args():
    """Parses command line arguments."""
    p = argparse.ArgumentParser(
        description="Run expert visualization scripts in parallel."
    )
    p.add_argument(
        "--rq", type=str, default="1234", help="String containing RQ IDs to process."
    )
    p.add_argument(
        "--num_eval_episodes",
        type=int,
        default=100,
        help="Number of evaluation episodes for each run.",
    )
    p.add_argument(
        "--max_workers",
        type=int,
        default=MAX_WORKERS,
        help="Maximum number of parallel processes.",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=TASK_TIMEOUT,
        help="Timeout in seconds for each visualization script run.",
    )
    return p.parse_args()


def run_single_visualization(
    task_id: int, expert: str, env: str, xml_file: str, num_episodes: int
) -> Dict[str, Any]:
    """
    Function executed by each worker process.
    It launches the visualize_expert.py script as a subprocess.
    """
    # Use sys.executable to ensure the same Python interpreter is used
    command = [
        sys.executable,
        SCRIPT_TO_RUN,
        "--weights",
        expert,
        "--env",
        env,
        "--xml_file",
        xml_file,
        "--num_eval_episodes",
        str(num_episodes),
        "--log",  # Assuming this is always needed based on original script
    ]
    # Create a descriptive identifier for logging
    run_description = f"Task {task_id} (Env: {env}, XML: {os.path.basename(xml_file)}, Expert: ...{expert[-20:]})"

    print(f"[Launcher PID {os.getpid()}] Starting: {run_description}")
    print(
        f"[Launcher PID {os.getpid()}] Command: {' '.join(command)}"
    )  # Log the command for debugging

    start_time = time.time()
    try:
        # Run the subprocess
        result = subprocess.run(
            command,
            capture_output=True,  # Capture stdout/stderr
            text=True,  # Decode output as text
            check=True,  # Raise CalledProcessError if exit code != 0
            timeout=TASK_TIMEOUT,  # Use the configured timeout
        )
        duration = time.time() - start_time
        print(
            f"[Launcher PID {os.getpid()}] SUCCESS: {run_description} (Duration: {duration:.2f}s)"
        )
        # Uncomment below if you want to see full output for successful runs
        # print(f"--- STDOUT ---\n{result.stdout}\n--- STDERR ---\n{result.stderr}\n----------------")
        return {
            "id": task_id,
            "description": run_description,
            "status": "success",
            "duration": duration,
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

    except subprocess.CalledProcessError as e:
        # Script returned a non-zero exit code
        duration = time.time() - start_time
        print(
            f"[Launcher PID {os.getpid()}] FAILED: {run_description} (Exit Code: {e.returncode}, Duration: {duration:.2f}s)"
        )
        print(
            f"--- STDOUT ---\n{e.stdout}\n--- STDERR ---\n{e.stderr}\n----------------"
        )
        return {
            "id": task_id,
            "description": run_description,
            "status": "failed",
            "duration": duration,
            "exit_code": e.returncode,
            "stdout": e.stdout,
            "stderr": e.stderr,
        }
    except subprocess.TimeoutExpired as e:
        # Script took too long
        duration = time.time() - start_time
        print(
            f"[Launcher PID {os.getpid()}] TIMEOUT: {run_description} (Duration: >{duration:.2f}s)"
        )
        # Output might be None or partial if timeout occurred
        stdout = e.stdout if e.stdout else ""
        stderr = e.stderr if e.stderr else ""
        print(
            f"--- STDOUT (Timeout) ---\n{stdout}\n--- STDERR (Timeout) ---\n{stderr}\n----------------"
        )
        return {
            "id": task_id,
            "description": run_description,
            "status": "timeout",
            "duration": duration,
            "exit_code": -1,
            "stdout": stdout,
            "stderr": stderr,
        }
    except FileNotFoundError:
        # Script_to_run not found
        duration = time.time() - start_time
        print(
            f"[Launcher PID {os.getpid()}] ERROR: Script '{SCRIPT_TO_RUN}' not found for {run_description}"
        )
        return {
            "id": task_id,
            "description": run_description,
            "status": "script_not_found",
            "duration": duration,
            "exit_code": -1,
            "error_message": f"Script not found: {SCRIPT_TO_RUN}",
        }
    except Exception as e:
        # Other unexpected errors during subprocess launch/management
        duration = time.time() - start_time
        print(f"[Launcher PID {os.getpid()}] ERROR launching {run_description}: {e}")
        return {
            "id": task_id,
            "description": run_description,
            "status": "launch_error",
            "duration": duration,
            "exit_code": -1,
            "error_message": str(e),
        }


def run(args):
    """Loads configs, prepares tasks, and runs them in parallel."""
    global MAX_WORKERS, TASK_TIMEOUT  # Allow updating globals from args
    MAX_WORKERS = args.max_workers
    TASK_TIMEOUT = args.timeout

    tasks_to_run: List[Tuple[str, str, str, int]] = []
    rqs = list(args.rq)
    print(f"Processing RQ identifiers: {rqs}")

    # 1. Prepare all tasks first
    for rq in rqs:
        rq_config_path = f"./experiments/RQ{rq}.yaml"
        print(f"Loading config: {rq_config_path}")
        try:
            with open(rq_config_path, "r") as f:
                experiments = yaml.safe_load(f)
                if not experiments or "environments" not in experiments:
                    print(
                        f"Warning: Config {rq_config_path} is empty or missing 'environments' key. Skipping."
                    )
                    continue
        except FileNotFoundError:
            print(f"Error: Config file not found: {rq_config_path}. Skipping RQ {rq}.")
            continue
        except yaml.YAMLError as e:
            print(f"Error parsing YAML file {rq_config_path}: {e}. Skipping RQ {rq}.")
            continue
        except Exception as e:
            print(f"Error reading file {rq_config_path}: {e}. Skipping RQ {rq}.")
            continue

        if rq == 3:
            best_runs = pd.read_csv("runs/best_runs_per_tag.csv")

            # Remove the rows that have AND
            col = "tag_specification"
            value = "AND"
            del_condition = best_runs[col].str.contains("AND", case=True, na=False)
            best_runs = best_runs[~del_condition]

        for env, config in experiments.get("environments", {}).items():
            if not config or "xml_files" not in config or "experts" not in config:
                print(
                    f"Warning: Invalid config structure for environment '{env}' in {rq_config_path}. Skipping."
                )
                continue

            xml_files = config.get("xml_files", [])
            experts = config.get("experts", {})

            if not xml_files:
                print(
                    f"Warning: No 'xml_files' listed for environment '{env}' in {rq_config_path}. Skipping env."
                )
                continue
            if not experts:
                print(
                    f"Warning: No 'experts' listed for environment '{env}' in {rq_config_path}. Skipping env."
                )
                continue

            for xml_file in xml_files:
                for expert_key, expert_path in experts.items():
                    if not expert_path:  # Skip if expert path is empty/null
                        print(
                            f"Warning: Empty expert path for key '{expert_key}' in env '{env}'. Skipping."
                        )
                        continue
                    # Add task details to the list
                    tasks_to_run.append(
                        (expert_path, env, xml_file, args.num_eval_episodes)
                    )

    if not tasks_to_run:
        print("No tasks generated based on the provided configurations. Exiting.")
        return

    total_tasks = len(tasks_to_run)
    print(f"\nPrepared {total_tasks} visualization tasks.")
    print(f"Starting parallel execution with up to {MAX_WORKERS} workers...")
    print(f"Individual task timeout set to {TASK_TIMEOUT} seconds.\n")

    # 2. Execute tasks in parallel
    start_time = time.time()
    results = []
    # Use ProcessPoolExecutor to manage worker processes
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit tasks to the pool. `submit` returns a Future object.
        # We assign a unique task_id (0 to total_tasks-1) to each task.
        future_to_task_id = {
            executor.submit(run_single_visualization, i, *task_args): i
            for i, task_args in enumerate(tasks_to_run)
        }

        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_task_id):
            task_id = future_to_task_id[future]
            try:
                result_data = (
                    future.result()
                )  # Get the result dict from run_single_visualization
                results.append(result_data)
            except Exception as exc:
                # Should not happen often if run_single_visualization catches exceptions,
                # but good practice to include.
                print(
                    f"[Launcher] CRITICAL ERROR processing result for Task {task_id}: {exc}"
                )
                # Try to find description if possible, otherwise use task_id
                task_desc = f"Task {task_id}"
                if tasks_to_run and task_id < len(tasks_to_run):
                    expert, env, xml_file, _ = tasks_to_run[task_id]
                    task_desc = f"Task {task_id} (Env: {env}, XML: {os.path.basename(xml_file)}, Expert: ...{expert[-20:]})"

                results.append(
                    {
                        "id": task_id,
                        "description": task_desc,
                        "status": "critical_error",
                        "exit_code": -1,
                        "error_message": str(exc),
                    }
                )

    end_time = time.time()

    # 3. Summarize results
    print(f"\n--- Execution Summary ---")
    print(f"Finished {total_tasks} tasks in {end_time - start_time:.2f} seconds.")

    status_counts = {}
    failed_details = []
    for r in results:
        status = r.get("status", "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
        if status != "success":
            failed_details.append(r)

    print("Outcomes:")
    for status, count in status_counts.items():
        print(f"  - {status.capitalize()}: {count}")

    if failed_details:
        print("\nDetails for Failed/Timeout/Error runs:")
        # Sort by task ID for consistency
        failed_details.sort(key=lambda x: x.get("id", -1))
        for r in failed_details:
            print(
                f"  - ID {r.get('id')}: Status={r.get('status')}, Desc={r.get('description', 'N/A')}, ExitCode={r.get('exit_code', 'N/A')}"
            )
            if "error_message" in r:
                print(f"      Error Msg: {r['error_message']}")
            # Optionally print captured stderr for failures here if needed
            # if r.get('stderr'):
            #     print(f"      Stderr: {r['stderr'][:200]}...") # Limit output length


# --- Main Execution ---
if __name__ == "__main__":
    cmd_args = get_args()
    run(cmd_args)
