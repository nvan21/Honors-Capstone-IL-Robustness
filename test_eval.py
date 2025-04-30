import multiprocessing
import subprocess
import time
import os
import sys
import yaml

# --- Configuration ---
TARGET_SCRIPT = "scripts/visualize_expert.py"
with open("test.yaml", "r") as f:
    experiments = yaml.safe_load(f)

# Each inner list represents the command line arguments for one run,
# excluding 'python' and the script name itself.
NUM_EVAL_EPISODES = "100"
ARGUMENT_SETS = []
for path, envs in experiments.items():
    for env, xml_files in envs.items():
        for xml_file in xml_files:
            ARGUMENT_SETS.append(
                [
                    "--weights",
                    path,
                    "--env",
                    env,
                    "--xml_file",
                    xml_file,
                    "--num_eval_episodes",
                    NUM_EVAL_EPISODES,
                    "--log",
                ]
            )

# Maximum number of concurrent processes
MAX_CONCURRENT_PROCESSES = 8


# --- Worker Function ---
# This function will be executed by each process in the pool
def run_script_worker(args_list):
    """
    Runs the TARGET_SCRIPT with the given arguments using subprocess.

    Args:
        args_list (list): A list of string arguments for the target script.

    Returns:
        tuple: (args_list, success (bool), return_code, stdout, stderr)
    """
    process_id = os.getpid()
    script_path = os.path.abspath(TARGET_SCRIPT)  # Get absolute path
    python_executable = sys.executable  # Use the same python interpreter

    # Construct the command line arguments
    # ['/path/to/python', '/path/to/your_target_script.py', 'arg1', 'arg2', ...]
    command = [python_executable, script_path] + args_list

    print(f"[Worker {process_id}] Running: {' '.join(command)}", flush=True)
    start_time = time.time()

    try:
        # Execute the command
        result = subprocess.run(
            command,
            capture_output=True,  # Capture stdout and stderr
            text=True,  # Decode stdout/stderr as text
            check=False,  # Don't raise exception on non-zero exit code
            # We will check result.returncode manually
        )
        end_time = time.time()
        duration = end_time - start_time

        success = True
        status_msg = "SUCCESS"

        print(
            f"[Worker {process_id}] Finished: {' '.join(args_list)} -> {status_msg} in {duration:.2f}s",
            flush=True,
        )

        # Return relevant information
        return (args_list, success, result.returncode, result.stdout, result.stderr)

    except FileNotFoundError:
        print(
            f"[Worker {process_id}] ERROR: Script not found at '{script_path}'",
            flush=True,
        )
        return (args_list, False, -1, "", f"Script not found: {script_path}")
    except Exception as e:
        print(
            f"[Worker {process_id}] ERROR: Unexpected exception running {' '.join(args_list)}: {e}",
            flush=True,
        )
        return (args_list, False, -1, "", f"Unexpected exception: {e}")


# --- Main Execution Block ---
if __name__ == "__main__":
    print(f"Main Process {os.getpid()}: Starting script runner.")
    print(f"Main Process: Target Script: {TARGET_SCRIPT}")
    print(f"Main Process: Max Concurrent Processes: {MAX_CONCURRENT_PROCESSES}")
    print(f"Main Process: Total tasks: {len(ARGUMENT_SETS)}")

    if not os.path.exists(TARGET_SCRIPT):
        print(f"\n--- ERROR ---")
        print(f"Target script '{TARGET_SCRIPT}' not found.")
        print(f"Please ensure the path is correct and the file exists.")
        print(f"---------------\n")
        sys.exit(1)  # Exit if the script doesn't exist

    start_total_time = time.time()

    results = []
    # Create a pool of worker processes
    # Using 'with' ensures the pool is properly closed and joined
    with multiprocessing.Pool(processes=MAX_CONCURRENT_PROCESSES) as pool:
        print(f"Main Process: Submitting {len(ARGUMENT_SETS)} tasks to the pool...")

        # Use map_async for non-blocking submission and collect results later
        # This applies 'run_script_worker' to each item in 'ARGUMENT_SETS'
        async_result = pool.map_async(run_script_worker, ARGUMENT_SETS)

        # Optional: You could add a progress indicator here by checking
        # async_result.ready() or using the callback/error_callback arguments
        # of map_async, but for simplicity, we'll just wait.

        print("Main Process: Waiting for all tasks to complete...")
        # Get the results. This call will block until all processes are done.
        # It also automatically calls pool.close() and pool.join() implicitly
        # when exiting the 'with' block.
        results = async_result.get()

    end_total_time = time.time()
    total_duration = end_total_time - start_total_time
    print(f"\nMain Process: All tasks finished in {total_duration:.2f} seconds.")

    # --- Process Results ---
    print("\n--- Execution Summary ---")
    success_count = 0
    fail_count = 0
    for result_data in results:
        args, success, return_code, stdout_str, stderr_str = result_data
        if success:
            success_count += 1
            print(f"[SUCCESS] Args: {args}")
            # Optionally print stdout for successful runs if needed
            # if stdout_str:
            #     print(f"  stdout: {stdout_str[:100]}...") # Print first 100 chars
        else:
            fail_count += 1
            print(f"[FAILED]  Code: {return_code}, Args: {args}")
            if stderr_str:
                print(f"  stderr: {stderr_str.strip()}")
            if stdout_str:  # Sometimes errors print to stdout too
                print(f"  stdout: {stdout_str.strip()}")

    print("-------------------------")
    print(f"Total Runs:   {len(ARGUMENT_SETS)}")
    print(f"Successful:   {success_count}")
    print(f"Failed:       {fail_count}")
    print("-------------------------")

    print(f"Main Process {os.getpid()}: Script finished.")
