# --- START OF FILE extract_wandb_info.py ---

import wandb
import pandas as pd
import sys
import yaml
from datetime import datetime  # Import datetime for timestamp conversion
import os  # Import os for path joining, directory creation, and basename

# --- Configuration ---
# User-provided values
entity = "nvanutrecht"
project = "Honors Capstone"
metric_names_priority = [
    "eval/mean_reward",  # Try this metric first
    "return/test",  # If the first isn't found, try this one
]
# --- Define Potential Config Keys for Environment ID ---
# List of keys to check in run.config, in order of preference
ENV_ID_CONFIG_KEYS = ["env_id", "env", "environment_id", "environment"]

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
            exp_names = list(config["experiments"].keys())
            if exp_names and filepath.endswith("sac.yaml"):
                exp_names.pop(0)
            specs.extend(exp_names)

        elif spec_type == "environments" and "environments" in config:
            env_names = list(config["environments"].keys())
            if base_tags and isinstance(base_tags, list):
                for env in env_names:
                    specs.append([env] + base_tags)
            else:
                print(
                    f"Warning: 'base_tags' missing or not a list for {filepath}. Adding env names only."
                )
                specs.extend(env_names)

        else:
            print(f"Warning: Expected key ('{spec_type}') not found in {filepath}")

    except FileNotFoundError:
        print(f"Warning: Configuration file not found: {filepath}")
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file {filepath}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred loading {filepath}: {e}")
    return specs


# Load specifications
tags_to_process.extend(
    load_and_transform_specs("./hyperparameters/sac.yaml", "experiments")
)
tags_to_process.extend(
    load_and_transform_specs(
        "./hyperparameters/airl.yaml", "environments", base_tags=["airl", "normal_env"]
    )
)
tags_to_process.extend(
    load_and_transform_specs(
        "./hyperparameters/gail.yaml", "environments", base_tags=["gail", "normal_env"]
    )
)
tags_to_process.extend(
    load_and_transform_specs(
        "./hyperparameters/bc.yaml", "environments", base_tags=["bc", "normal_env"]
    )
)

print(f"\nGenerated {len(tags_to_process)} tag specifications to process:")
# for spec in tags_to_process: print(f"  - {spec}") # Optional


# --- Derived Configuration ---
runs_path_base = f"{entity}/{project}"  # Base path used in api.runs()
# Define the desired output date format string
OUTPUT_DATE_FORMAT = "%Y%m%d-%H%M"

# --- Initialization ---
best_run_results = {}
api = None

# --- Authentication and API Initialization ---
try:
    api = wandb.Api(timeout=60)
    print(f"\nSuccessfully connected to Wandb API.")
    print(f"Processing project: {runs_path_base}")
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

    # Determine Filter Logic Based on Type
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
        runs = api.runs(path=runs_path_base, filters=filter_dict, order="-created_at")
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
                break

        if found_metric_for_this_run:
            if current_run_metric_val > best_metric_value_for_tag:
                best_metric_value_for_tag = current_run_metric_val

                # Get XML Basename Safely
                xml_basename = None
                full_xml_path = run.config.get("xml_file")
                if full_xml_path and isinstance(full_xml_path, str):
                    try:
                        xml_basename = os.path.basename(full_xml_path)
                    except Exception as e:
                        print(
                            f"  Warning: Could not get basename for XML path '{full_xml_path}': {e}"
                        )
                        xml_basename = full_xml_path

                # --- Get Environment ID from Config Safely ---
                env_id = None
                for key in ENV_ID_CONFIG_KEYS:
                    env_id = run.config.get(key)
                    if env_id:  # Found a non-empty value
                        break  # Stop checking keys
                if not env_id:  # If still None after checking all keys
                    env_id = "N/A"  # Default value if not found

                # Store details, including env_id and run path
                best_run_details_for_tag = {
                    "run_name": run.name,
                    "run_id": run.id,
                    "env_id": env_id,  # <-- ADDED Env ID
                    "run_path": f"{entity}/{project}/{run.id}",
                    "xml_file": xml_basename,
                    "created_at": run.created_at,
                    "last_updated_ts": run.summary.get("_timestamp"),
                    "metric_value": best_metric_value_for_tag,
                    "metric_name_used": current_run_metric_used,
                    "run_state": run.state,
                }

    # Store results for this tag_spec
    print(
        f"  Checked {runs_checked_count} runs matching '{tag_key_string}'. Found {runs_with_metric_count} runs with any target metric."
    )
    if best_run_details_for_tag:
        best_run_results[tag_key_string] = {
            "status": "found_best",
            "details": best_run_details_for_tag,
        }
        # Temporarily format dates for the print statement
        created_at_str_print = str(best_run_details_for_tag["created_at"])
        try:
            created_dt_print = pd.to_datetime(best_run_details_for_tag["created_at"])
            if pd.notna(created_dt_print):
                created_at_str_print = created_dt_print.strftime(OUTPUT_DATE_FORMAT)
        except Exception:
            pass

        last_updated_ts = best_run_details_for_tag.get("last_updated_ts")
        last_updated_str_print = "N/A"
        if last_updated_ts:
            try:
                last_updated_dt_print = datetime.fromtimestamp(last_updated_ts)
                last_updated_str_print = last_updated_dt_print.strftime(
                    OUTPUT_DATE_FORMAT
                )
            except Exception:
                last_updated_str_print = str(last_updated_ts)

        xml_file_str = best_run_details_for_tag.get("xml_file", "N/A")
        if not xml_file_str:
            xml_file_str = "N/A"

        # Get env_id for printing
        env_id_str = best_run_details_for_tag.get("env_id", "N/A")

        print(
            f"  Best run selected for '{tag_key_string}': {best_run_details_for_tag['run_name']} "
            f"(Env: {env_id_str}, XML: {xml_file_str}, Created: {created_at_str_print}, Last Updated: {last_updated_str_print}) "  # Added Env to print
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


# --- Aggregate and Save the Best Run Results ---
print("\n--- Aggregating Best Run Results ---")

best_runs_list = []

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

    # Convert original time columns to datetime objects
    try:
        best_runs_df["created_at_dt"] = pd.to_datetime(
            best_runs_df["created_at"], errors="coerce"
        )
    except Exception:
        best_runs_df["created_at_dt"] = pd.NaT
    try:
        best_runs_df["last_updated_dt"] = pd.to_datetime(
            best_runs_df["last_updated_ts"], unit="s", errors="coerce"
        )
    except KeyError:
        best_runs_df["last_updated_dt"] = pd.NaT
    except Exception:
        best_runs_df["last_updated_dt"] = pd.NaT

    # Create STRING columns with the desired format
    if "created_at_dt" in best_runs_df.columns:
        best_runs_df["created_at_str"] = best_runs_df["created_at_dt"].apply(
            lambda x: x.strftime(OUTPUT_DATE_FORMAT) if pd.notna(x) else "N/A"
        )
    else:
        best_runs_df["created_at_str"] = "N/A"
    if "last_updated_dt" in best_runs_df.columns:
        best_runs_df["last_updated_str"] = best_runs_df["last_updated_dt"].apply(
            lambda x: x.strftime(OUTPUT_DATE_FORMAT) if pd.notna(x) else "N/A"
        )
    else:
        best_runs_df["last_updated_str"] = "N/A"

    # Define final column order including env_id
    column_order = [
        "tag_specification",
        "env_id",  # <-- ADDED column
        "xml_file",
        "metric_value",
        "metric_name_used",
        "run_name",
        "run_id",
        "run_path",
        "created_at_str",
        "last_updated_str",
        "run_state",
    ]
    existing_columns = [col for col in column_order if col in best_runs_df.columns]
    best_runs_df_final = best_runs_df[existing_columns]

    # Sort by the tag specification string
    best_runs_df_final = best_runs_df_final.sort_values(
        by="tag_specification"
    ).reset_index(drop=True)

    print("\n--- Best Run per Tag Specification ---")
    pd.set_option("display.max_rows", len(best_runs_df_final) + 10)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 320)  # Increase width further
    print(best_runs_df_final)

    try:
        output_dir = "runs"
        os.makedirs(output_dir, exist_ok=True)
        output_filename = os.path.join(output_dir, "best_runs_per_tag.csv")
        best_runs_df_final.to_csv(
            output_filename,
            index=False,
            float_format="%.4f",
        )
        print(f"\nBest run results saved to {output_filename}")
    except Exception as e:
        print(f"\nError saving results to CSV: {e}")

else:
    print(
        "\nNo best runs could be determined for any of the specified tag specifications."
    )

print("\n--- Script Finished ---")
# --- END OF FILE extract_wandb_info.py ---
