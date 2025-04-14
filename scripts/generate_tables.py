import pandas as pd
import json
from pathlib import Path
import os
import re  # For regex parsing and filename sanitization

# --- Configuration ---
runs_base_dir = "runs"  # The top-level directory containing experiment folders
output_precision = 2  # Number of decimal places for mean/std
nan_replacement = "-"  # How to display missing results

# --- Formatting Maps ---
# Define how algorithm names should appear in the table index
ALGO_NAME_MAP = {
    "sac": "SAC",
    "dagger": "DAgger",  # Note the specific capitalization
    "airl": "AIRL",
    "gail": "GAIL",
    "bc": "BC",
    # Add other algorithms here if needed, preserving case if input differs
}


# Function to capitalize the first letter of environment names
def format_env_name(env_name):
    if not env_name:
        return ""
    return env_name.capitalize()


# --- NEW: Function to format modification details ---
def format_modification_details(mod_string):
    """
    Formats modification strings like 'torso_mass_p20' into 'Torso Mass +20%'.
    """
    if not mod_string or mod_string.lower() == "standard":
        return "Standard"  # Return early for standard case

    # Regex to capture: (property_part)_(p|m)(digits)
    # Example: torso_mass_p20 -> group1='torso_mass', group2='p', group3='20'
    match = re.match(r"^(.*)_([pm])(\d+)$", mod_string, re.IGNORECASE)

    if match:
        property_raw = match.group(1)
        modifier = match.group(2).lower()  # Ensure modifier is lowercase
        value = match.group(3)

        # Format property: Replace underscores, title case
        property_formatted = property_raw.replace("_", " ").title()

        # Determine sign
        sign = "+" if modifier == "p" else "-"

        return f"{property_formatted} {sign}{value}\%"
    else:
        # Fallback if the pattern doesn't match: Capitalize and replace underscores
        print(
            f"  INFO: Modification string '{mod_string}' did not match p/m pattern. Using fallback formatting."
        )
        return mod_string.replace("_", " ").capitalize()


# ---------------------

# --- Data Storage Initialization ---
standard_table_data = {}
modified_tables_data = {}

# --- File Discovery ---
base_path = Path(runs_base_dir)
json_files = list(base_path.rglob("results.json"))

if not json_files:
    print(f"ERROR: No 'results.json' files found anywhere within '{runs_base_dir}'.")
    exit()

print(f"Found {len(json_files)} results.json files.")

# --- Processing Loop ---
for json_path in json_files:
    print(f"Processing: {json_path}")
    try:
        # --- Load JSON Data ---
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # --- Extract Required Data from JSON ---
        algo_raw = data.get("algo")
        xml_path_str = data.get("xml_file")
        mean_ret = data.get("mean_return")
        std_ret = data.get("std_return")

        # --- Data Validation ---
        if not all([algo_raw, xml_path_str, mean_ret is not None, std_ret is not None]):
            print(f"  WARNING: Skipping - Missing required keys in {json_path}")
            continue

        # --- Format Algorithm Name ---
        formatted_algo_name = ALGO_NAME_MAP.get(algo_raw.lower(), algo_raw.upper())

        # --- Extract Information from XML Path ---
        xml_path = Path(xml_path_str)
        xml_filename_stem = xml_path.stem
        split_stem = xml_filename_stem.split("_", 1)
        base_env_raw = split_stem[0]
        formatted_base_env = format_env_name(base_env_raw)

        # Determine standard/modified and get RAW modification details
        is_standard = False
        raw_modification_details = "standard"  # Default

        if xml_filename_stem.lower() == base_env_raw.lower():
            is_standard = True
        else:
            prefix_to_remove = base_env_raw.lower() + "_"
            if xml_filename_stem.lower().startswith(prefix_to_remove):
                raw_modification_details = xml_filename_stem[len(prefix_to_remove) :]
            elif len(split_stem) > 1:  # Fallback 1
                raw_modification_details = split_stem[1]
            else:  # Fallback 2
                raw_modification_details = xml_filename_stem

        # --- Format Modification Details (NEW STEP) ---
        # This applies only when it's *not* standard, otherwise uses default "Standard"
        if not is_standard:
            formatted_modification = format_modification_details(
                raw_modification_details
            )
        else:
            # For standard envs, we don't need a modification column name
            pass  # No specific modification formatting needed

        # --- Format Output String ---
        try:
            mean_str = f"{float(mean_ret):.{output_precision}f}"
            std_str = f"{float(std_ret):.{output_precision}f}"
            cell_value = f"{mean_str} \\pm {std_str}"  # Use LaTeX +/- symbol
        except (ValueError, TypeError) as e:
            print(
                f"  WARNING: Skipping - Error formatting mean/std return (must be numeric) in {json_path}: {e}"
            )
            continue

        # --- Store Data in Appropriate Dictionary using FORMATTED names ---
        if is_standard:
            standard_table_data.setdefault(formatted_algo_name, {})[
                formatted_base_env
            ] = cell_value
        else:
            # Use formatted_base_env and formatted_modification for modified tables
            modified_tables_data.setdefault(formatted_base_env, {})
            modified_tables_data[formatted_base_env].setdefault(
                formatted_algo_name, {}
            )[formatted_modification] = cell_value

    except json.JSONDecodeError:
        print(f"  ERROR: Could not decode JSON from {json_path}")
    except IOError as e:
        print(f"  ERROR: Could not read file {json_path}: {e}")
    except Exception as e:
        print(f"  ERROR: An unexpected error occurred processing {json_path}: {e}")


# --- Generate Tables ---

print("\n" + "=" * 50)
print("Generating LaTeX Tables...")
print("=" * 50 + "\n")

# 1. Standard Environments Table
if not standard_table_data:
    print("No data found for the standard environments table.")
else:
    print("--- Standard Environments Table (LaTeX Output) ---")
    standard_df = pd.DataFrame.from_dict(standard_table_data, orient="index")
    standard_df = standard_df.sort_index(axis=0)  # Sort rows (formatted algos)
    standard_df = standard_df.sort_index(axis=1)  # Sort columns (formatted base envs)
    standard_df = standard_df.fillna(nan_replacement)

    try:
        std_latex = standard_df.to_latex(
            index=True,
            index_names=True,
            na_rep=nan_replacement,
            caption="Algorithm Performance on Standard Environments (Mean $\\pm$ Std Return)",
            label="tab:perf_standard",
            escape=False,
            bold_rows=False,
        )
        print(std_latex)
        tex_filename_std = os.path.join(
            "assets", "tables", "summary_standard_results.tex"
        )
        with open(tex_filename_std, "w", encoding="utf-8") as f:
            f.write(std_latex)
        print(f"\nStandard table LaTeX code saved to {tex_filename_std}")
        print(
            "Remember to \\usepackage{booktabs} and potentially \\usepackage{amsmath} (for \\pm) in your LaTeX document."
        )
    except Exception as e:
        print(f"\nError exporting standard table to LaTeX: {e}")


# 2. Modified Environments Tables (One per base_env)
if not modified_tables_data:
    print("\nNo data found for any modified environments.")
else:
    print("\n--- Modified Environments Tables (LaTeX Output) ---")
    sorted_formatted_base_envs = sorted(modified_tables_data.keys())

    for formatted_base_env in sorted_formatted_base_envs:
        mod_data = modified_tables_data[formatted_base_env]
        print(f"\n--- {formatted_base_env} Modified Environments ---")
        if not mod_data:
            continue

        # Create DataFrame with formatted algo names as index & formatted modifications as columns
        mod_df = pd.DataFrame.from_dict(mod_data, orient="index")
        mod_df = mod_df.sort_index(axis=0)  # Sort rows (formatted algos)
        # Sort columns (formatted modifications) - Note: pandas sorts strings alphabetically
        # Custom sorting might be needed if alpha sort isn't desired (e.g., -20% before +10%)
        mod_df = mod_df.sort_index(axis=1)
        mod_df = mod_df.fillna(nan_replacement)

        # --- Export Modified Table to LaTeX ---
        safe_base_env_name = re.sub(r"[^\w-]", "", formatted_base_env)
        tex_filename_mod = os.path.join(
            "assets", "tables", f"summary_modified_{safe_base_env_name}_results.tex"
        )
        try:
            mod_latex = mod_df.to_latex(
                index=True,
                index_names=True,
                na_rep=nan_replacement,
                caption=f"Algorithm Performance on Modified {formatted_base_env} Environments (Mean $\\pm$ Std Return)",
                label=f"tab:perf_mod_{safe_base_env_name.lower()}",
                escape=False,  # Allow \pm
                bold_rows=False,
            )
            print(mod_latex)
            with open(tex_filename_mod, "w", encoding="utf-8") as f:
                f.write(mod_latex)
            print(
                f"\n{formatted_base_env} modified table LaTeX code saved to {tex_filename_mod}"
            )
            print(
                "Remember to \\usepackage{booktabs} and potentially \\usepackage{amsmath} (for \\pm) in your LaTeX document."
            )
        except Exception as e:
            print(
                f"\nError exporting modified table LaTeX for {formatted_base_env}: {e}"
            )

print("\n" + "=" * 50)
print("Script finished.")
print("=" * 50)
