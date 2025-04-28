import wandb
import pandas as pd
import sys
import yaml
from datetime import datetime  # Import datetime for timestamp conversion
import os

# --- Configuration ---
# User-provided values
entity = "nvanutrecht"
project = "Honors Capstone"
metric_names_priority = [
    "eval/mean_reward",  # Try this metric first
    "return/test",  # If the first isn't found, try this one
    # Add more fallback metric names here if needed
]
tags_to_process = []


# --- Load Tags/Specifications from YAML files ---
# Helper function to load and transform specs
def load_and_transform_specs(filepath, spec_type, base_tags=None):
    """Loads environment or experiment names and transforms them into tag specs."""
    specs = []
    try:
        with open(filepath, "r") as f:
            config = yaml.safe_load(f)

        if spec_type == "experiments" and "experiments" in config:
            # Get experiment names (single tags)
            exp_names = list(config["experiments"].keys())
            # Optionally remove the first item if needed (e.g., 'defaults' or 'normal_env')
            if exp_names and filepath.endswith("sac.yaml"):  # Example specific logic
                exp_names.pop(0)
            specs.extend(exp_names)  # Add single experiment names

        elif spec_type == "environments" and "environments" in config:
            # Get environment names and combine them with base tags
            env_names = list(config["environments"].keys())
            if base_tags and isinstance(base_tags, list):
                for env in env_names:
                    # Create a list: [env_name, base_tag1, base_tag2, ...]
                    specs.append([env] + base_tags)
            else:
                print(
                    f"Warning: 'base_tags' missing or not a list for {filepath}. Adding env names only."
                )
                specs.extend(env_names)  # Fallback: add env names as single tags

        else:
            print(f"Warning: Expected key ('{spec_type}') not found in {filepath}")

    except FileNotFoundError:
        print(f"Warning: Configuration file not found: {filepath}")
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file {filepath}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred loading {filepath}: {e}")
    return specs


# Load SAC experiments (single tags, remove first)
tags_to_process.extend(
    load_and_transform_specs("./hyperparameters/sac.yaml", "experiments")
)

# Load AIRL environments and combine with ["airl", "normal_env"]
tags_to_process.extend(
    load_and_transform_specs(
        "./hyperparameters/airl.yaml", "environments", base_tags=["airl", "normal_env"]
    )
)

# Load GAIL environments and combine with ["gail", "normal_env"]
tags_to_process.extend(
    load_and_transform_specs(
        "./hyperparameters/gail.yaml", "environments", base_tags=["gail", "normal_env"]
    )
)

# Load BC environments and combine with ["bc", "normal_env"]
tags_to_process.extend(
    load_and_transform_specs(
        "./hyperparameters/bc.yaml", "environments", base_tags=["bc", "normal_env"]
    )
)

print(f"\nGenerated {len(tags_to_process)} tag specifications to process:")
# for spec in tags_to_process: # Optional: print all specs
#     print(f"  - {spec}")


# --- Derived Configuration ---
runs_path = f"{entity}/{project}"

# --- Initialization ---
best_run_results = {}
api = None

# --- Authentication and API Initialization ---
try:
    api = wandb.Api(timeout=60)  # Increased timeout
    print(f"\nSuccessfully connected to Wandb API.")
    print(f"Processing project: {runs_path}")
    _ = api.project(name=project, entity=entity)
    print(f"Confirmed access to project '{project}' under entity '{entity}'.")
except wandb.errors.CommError as e:
    print(f"Communication Error: Could not connect. Details: {e}")
    sys.exit(1)
except wandb.errors.AuthenticationError:
    print(
        "Authentication error: Please log in using 'wandb login' or set WANDB_API_KEY."
    )
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during initialization: {e}")
    sys.exit(1)


# --- Main Processing Loop ---
for tag_spec in tags_to_process:

    filter_dict = None
    tag_key_string = None

    # Determine Filter Logic Based on Type (same as before)
    if isinstance(tag_spec, str):
        tag_key_string = tag_spec
        print(f"\n--- Processing Single Tag: '{tag_key_string}' ---")
        filter_dict = {"tags": tag_key_string}

    elif isinstance(tag_spec, (list, tuple)):
        valid_tags_in_spec = [tag for tag in tag_spec if isinstance(tag, str)]
        if not valid_tags_in_spec:
            print(f"\n--- Skipping Invalid/Empty Combination: {tag_spec} ---")
            continue
        tag_key_string = " AND ".join(sorted(valid_tags_in_spec))
        print(f"\n--- Processing Tag Combination: '{tag_key_string}' ---")
        if len(valid_tags_in_spec) == 1:
            filter_dict = {"tags": valid_tags_in_spec[0]}
        else:
            and_conditions = [{"tags": tag} for tag in valid_tags_in_spec]
            filter_dict = {"$and": and_conditions}
    else:
        print(
            f"\n--- Skipping Invalid Specification Type: {type(tag_spec)} ({tag_spec}) ---"
        )
        continue

    # Initialize results for this tag/combination
    best_metric_value_for_tag = -float("inf")
    best_run_details_for_tag = None
    runs_checked_count = 0
    runs_with_metric_count = 0

    # Fetch Runs using the constructed filter
    try:
        print(f"  Applying filter: {filter_dict}")
        # Add ordering by creation time descending to potentially prioritize newer runs
        # if multiple runs match and have the same best score later
        runs = api.runs(path=runs_path, filters=filter_dict, order="-created_at")
    except Exception as e:
        print(f"  Error fetching runs for spec '{tag_key_string}': {e}. Skipping.")
        best_run_results[tag_key_string] = {"status": "error_fetching"}
        continue

    if not runs:
        print(f"  No runs found matching specification: '{tag_key_string}'.")
        best_run_results[tag_key_string] = {"status": "no_runs_found"}
        continue

    # Iterate through the found runs
    for run in runs:
        runs_checked_count += 1
        found_metric_for_this_run = False
        current_run_metric_val = None
        current_run_metric_used = None

        for metric_name_to_try in metric_names_priority:
            potential_val = run.summary.get(metric_name_to_try)
            is_usable = False
            if potential_val is not None:
                if isinstance(potential_val, (int, float)) and not pd.isna(
                    potential_val
                ):
                    is_usable = True

            if is_usable:
                current_run_metric_val = float(potential_val)
                current_run_metric_used = metric_name_to_try
                found_metric_for_this_run = True
                if not hasattr(run, "_counted_for_metric"):
                    runs_with_metric_count += 1
                    run._counted_for_metric = True
                break  # Found highest priority metric for *this* run

        if found_metric_for_this_run:
            # Compare with the best metric found *so far* for this tag_spec
            if current_run_metric_val > best_metric_value_for_tag:
                best_metric_value_for_tag = current_run_metric_val
                # Store details, including the summary timestamp (_timestamp)
                best_run_details_for_tag = {
                    "run_name": run.name,
                    "run_id": run.id,
                    "created_at": run.created_at,  # Start time
                    "last_updated_ts": run.summary.get(
                        "_timestamp"
                    ),  # <-- ADDED: Timestamp of last summary update (Unix timestamp)
                    "metric_value": best_metric_value_for_tag,
                    "metric_name_used": current_run_metric_used,
                    "run_state": run.state,
                }
                # Note: No need to print intermediate bests unless debugging

    # Store results for this tag_spec using the string key
    print(
        f"  Checked {runs_checked_count} runs matching '{tag_key_string}'. Found {runs_with_metric_count} runs with any target metric."
    )
    if best_run_details_for_tag:
        best_run_results[tag_key_string] = {
            "status": "found_best",
            "details": best_run_details_for_tag,
        }
        # Format timestamp for printing here
        last_updated_ts = best_run_details_for_tag.get("last_updated_ts")
        last_updated_str = "N/A"
        if last_updated_ts:
            try:
                last_updated_dt = datetime.fromtimestamp(last_updated_ts)
                last_updated_str = last_updated_dt.strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                last_updated_str = str(last_updated_ts)  # Fallback

        print(
            f"  Best run selected for '{tag_key_string}': {best_run_details_for_tag['run_name']} "
            f"(Created: {best_run_details_for_tag['created_at']}, Last Updated: {last_updated_str}) "  # Updated print
            f"with value {best_run_details_for_tag['metric_value']:.4f}"
        )
    elif runs_checked_count > 0 and runs_with_metric_count == 0:
        print(
            f"  No runs with the specified metrics ({metric_names_priority}) found for '{tag_key_string}'."
        )
        best_run_results[tag_key_string] = {"status": "no_valid_metrics"}
    elif runs_checked_count > 0 and runs_with_metric_count > 0:
        print(
            f"  Runs with metrics found for '{tag_key_string}', but couldn't determine a best one."
        )
        best_run_results[tag_key_string] = {"status": "no_best_determined"}
    # 'no_runs_found' was handled earlier


# --- Aggregate and Save the Best Run Results ---
print("\n--- Aggregating Best Run Results ---")

best_runs_list = []

# Iterate using the string keys generated earlier
for tag_key, result_info in best_run_results.items():
    status = result_info.get("status")
    if status == "found_best":
        details = result_info.get("details")
        if details:
            details_with_tag = details.copy()
            details_with_tag["tag_specification"] = tag_key
            best_runs_list.append(details_with_tag)
        else:
            print(
                f"Warning: Tag spec '{tag_key}' has status 'found_best' but no details found."
            )
    # Print statuses for skipped/error cases (same as before)
    elif status == "error_fetching":
        print(f"Skipping spec '{tag_key}': Error during run fetching.")
    elif status == "no_runs_found":
        print(f"Skipping spec '{tag_key}': No runs found matching this spec.")
    elif status == "no_valid_metrics":
        print(
            f"Skipping spec '{tag_key}': Runs found, but none had the required metrics."
        )
    elif status == "no_best_determined":
        print(
            f"Skipping spec '{tag_key}': Runs with metrics found, but couldn't select a best."
        )
    else:
        print(f"Skipping spec '{tag_key}': Unknown status '{status}'.")


if best_runs_list:
    best_runs_df = pd.DataFrame(best_runs_list)

    # Convert time columns to datetime objects
    try:
        best_runs_df["created_at"] = pd.to_datetime(best_runs_df["created_at"])
    except Exception as e:
        print(
            f"Warning: Could not convert 'created_at' column to datetime objects: {e}"
        )
        best_runs_df["created_at"] = pd.NaT  # Fill with Not-a-Time on error

    try:
        # Convert Unix timestamp (seconds since epoch) to datetime
        best_runs_df["last_updated"] = pd.to_datetime(
            best_runs_df["last_updated_ts"], unit="s", errors="coerce"
        )  # Use 'coerce' to handle potential errors/None
        # Optional: Localize to UTC then convert to your timezone
        # best_runs_df['last_updated'] = best_runs_df['last_updated'].dt.tz_localize('UTC').dt.tz_convert('America/New_York')
    except Exception as e:
        print(
            f"Warning: Could not convert 'last_updated_ts' column to datetime objects: {e}"
        )
        best_runs_df["last_updated"] = pd.NaT  # Fill with Not-a-Time on error

    # Reorder columns, adding 'last_updated'
    column_order = [
        "tag_specification",
        "metric_value",
        "metric_name_used",
        "run_name",
        "run_id",
        "created_at",  # Start time
        "last_updated",  # Time of last summary update
        "run_state",
    ]
    # Keep only existing columns and drop the raw timestamp column if desired
    existing_columns = [col for col in column_order if col in best_runs_df.columns]
    best_runs_df = best_runs_df[existing_columns]  # Select and order

    # Sort by the tag specification string
    best_runs_df = best_runs_df.sort_values(by="tag_specification").reset_index(
        drop=True
    )

    print("\n--- Best Run per Tag Specification ---")
    pd.set_option("display.max_rows", len(best_runs_df) + 10)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 220)  # Increase width further
    print(best_runs_df)

    try:

        output_filename = os.path.join("runs", "best_runs_per_tag.csv")

        # Save to CSV with formatted dates
        best_runs_df.to_csv(
            output_filename,
            index=False,
            float_format="%.4f",
            date_format="%Y-%m-%d %H:%M:%S",  # Applies to all datetime columns
        )
        print(f"\nBest run results saved to {output_filename}")
    except Exception as e:
        print(f"\nError saving results to CSV: {e}")

else:
    print(
        "\nNo best runs could be determined for any of the specified tag specifications."
    )

print("\n--- Script Finished ---")
