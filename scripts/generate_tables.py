import pandas as pd
import json
from pathlib import Path
import os
import re  # For regex parsing and filename sanitization

# --- Configuration ---
runs_base_dir = "runs"  # The top-level directory containing experiment folders
output_precision = 2  # Number of decimal places for mean/std
nan_replacement = "-"  # How to display missing results
output_dir = Path("assets") / "tables"  # Directory to save LaTeX tables

# Ensure output directory exists
output_dir.mkdir(parents=True, exist_ok=True)


# --- Formatting Maps ---
# Define how algorithm names should appear in the table index
ALGO_NAME_MAP = {
    "sac": "Expert SAC",
    "modified_sac": "Modified SAC",  # Assuming this might exist based on original code
    "dagger": "DAgger",
    "airl": "AIRL",
    "gail": "GAIL",
    "bc": "BC",
    # Add other algorithms here if needed, preserving case if input differs
}

# --- Utility Functions ---


# Function to capitalize the first letter of environment names
def format_env_name(env_name):
    """Capitalizes the first letter of a string."""
    if not env_name:
        return ""
    return env_name.capitalize()


# --- NEW: Function to format modification details ---
def format_modification_details(mod_string):
    """
    Formats modification strings like 'ant_gear_m20' into 'Ant Gear -20%'.
    Handles patterns like entity_property_[p|m]amount.
    """
    if not mod_string or mod_string.lower() == "standard":
        return "Standard"  # Should not happen with current logic, but good fallback

    # Regex to capture: (entity)_(property)_(p|m)(digits)
    # Example: ant_gear_m20 -> group1='ant', group2='gear', group3='m', group4='20'
    # Example: ant_torso_mass_p20 -> group1='ant', group2='torso_mass', group3='p', group4='20'
    match = re.match(r"^([a-zA-Z]+)_(.*)_([pm])(\d+)$", mod_string, re.IGNORECASE)

    if match:
        entity_raw = match.group(1)
        property_raw = match.group(2)
        modifier = match.group(3).lower()  # Ensure modifier is lowercase
        value = match.group(4)

        # Format entity and property
        entity_formatted = entity_raw.capitalize()
        property_formatted = property_raw.replace("_", " ").title()

        # Determine sign
        sign = "+" if modifier == "p" else "-"

        return f"{entity_formatted} {property_formatted} {sign}{value}\%"
    else:
        # Fallback if the pattern doesn't match: Capitalize and replace underscores
        print(
            f"  INFO: Modification string '{mod_string}' did not match entity_property_p/m_amount pattern. Using fallback formatting."
        )
        return mod_string.replace("_", " ").capitalize()


# --- Parsing XML Filename to get Env, Version, and Modification ---
def parse_xml_filename(xml_path_str):
    """
    Parses an XML filename (like Ant-v5_ant_gear_m20.xml or Ant-v5_ant.xml)
    to extract base environment name, full environment ID, and modification suffix.
    Returns (base_env_name, full_env_id, modification_suffix, is_standard).
    """
    xml_path = Path(xml_path_str)
    xml_filename_stem = xml_path.stem  # e.g., "Ant-v5_ant_gear_m20" or "Ant-v5_ant"

    # Try to match the pattern BaseEnv-vX_suffix
    match = re.match(r"^([a-zA-Z]+-v\d+)_?(.*)$", xml_filename_stem)

    if match:
        full_env_id = match.group(1)  # e.g., "Ant-v5"
        suffix = match.group(2)  # e.g., "ant_gear_m20" or "ant"

        # Extract base environment name from full_env_id (e.g., "Ant" from "Ant-v5")
        base_env_name_match = re.match(r"^([a-zA-Z]+)-v\d+$", full_env_id)
        if base_env_name_match:
            base_env_name = base_env_name_match.group(1)  # e.g., "Ant"
        else:
            # Fallback: use the full_env_id as the base name if pattern doesn't match
            print(
                f"  WARNING: Could not extract base name from '{full_env_id}'. Using full ID."
            )
            base_env_name = full_env_id

        # Determine if it's standard: suffix matches the lowercase base env name
        is_standard = (
            suffix.lower() == base_env_name.lower() or suffix == ""
        )  # "" handles potential BaseEnv-vX.xml case

        modification_suffix = (
            suffix if not is_standard else "standard"
        )  # Store the raw suffix if modified

        return base_env_name, full_env_id, modification_suffix, is_standard
    else:
        # Fallback for filenames that don't match BaseEnv-vX_suffix (e.g., just 'MyEnv.xml')
        print(
            f"  WARNING: XML filename '{xml_filename_stem}' did not match BaseEnv-vX_suffix pattern. Using full stem as base env name."
        )
        return (
            xml_filename_stem,
            xml_filename_stem,
            "standard",
            True,
        )  # Treat as standard


# ---------------------

# --- Data Storage Initialization ---
# standard_table_data: {formatted_algo: {formatted_full_env_id: cell_value}}
standard_table_data = {}

# modified_tables_data: {formatted_base_env_name: {formatted_algo: {formatted_modification: cell_value}}}
modified_tables_data = {}

# --- File Discovery ---
base_path = Path(runs_base_dir)
# json_files = list(base_path.rglob("results.json")) # This finds *all* results.json
# Let's be more specific: Find results.json only within algo subdirectories
json_files = list(base_path.rglob("*/results.json"))


if not json_files:
    print(
        f"ERROR: No 'results.json' files found anywhere within '{runs_base_dir}' in algorithm subdirectories."
    )
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
        xml_path_str = data.get("xml_file")  # Expecting the full path from the run
        mean_ret = data.get("mean_return")
        std_ret = data.get("std_return")

        # --- Data Validation ---
        if not all([algo_raw, xml_path_str, mean_ret is not None, std_ret is not None]):
            print(f"  WARNING: Skipping - Missing required keys in {json_path}")
            continue

        # --- Parse Environment Details from XML Path ---
        base_env_name, full_env_id, raw_modification_suffix, is_standard = (
            parse_xml_filename(xml_path_str)
        )

        # --- Format Names for Tables ---
        formatted_algo_name = ALGO_NAME_MAP.get(algo_raw.lower(), algo_raw.upper())
        formatted_base_env_name = format_env_name(base_env_name)
        formatted_full_env_id = (
            full_env_id  # Use the full ID for standard table column header if needed
        )

        # --- Format Modification Details (for modified tables) ---
        # This applies only when it's *not* standard
        if not is_standard:
            formatted_modification = format_modification_details(
                raw_modification_suffix
            )
        # No specific formatted_modification needed for standard envs

        # --- Format Output String ---
        try:
            # Check if mean_ret/std_ret are list/tuple and take the first element if needed
            if isinstance(mean_ret, (list, tuple)):
                mean_ret = mean_ret[0]
            if isinstance(std_ret, (list, tuple)):
                std_ret = std_ret[0]

            mean_str = f"{float(mean_ret):.{output_precision}f}"
            std_str = f"{float(std_ret):.{output_precision}f}"
            cell_value = f"{mean_str} \\pm {std_str}"  # Use LaTeX +/- symbol
        except (ValueError, TypeError, IndexError) as e:
            print(
                f"  WARNING: Skipping - Error formatting mean/std return (must be numeric, list/tuple supported) in {json_path}: {e}"
            )
            continue

        # --- Store Data in Appropriate Dictionary ---
        if is_standard:
            # Standard table: Algo (row) vs Full Env ID (col)
            standard_table_data.setdefault(formatted_algo_name, {})[
                formatted_full_env_id
            ] = cell_value
        else:
            # Modified tables: Grouped by Base Env Name, Algo (row) vs Formatted Modification (col)
            modified_tables_data.setdefault(formatted_base_env_name, {})
            modified_tables_data[formatted_base_env_name].setdefault(
                formatted_algo_name, {}
            )
            modified_tables_data[formatted_base_env_name][formatted_algo_name][
                formatted_modification
            ] = cell_value

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
    # standard_table_data: {formatted_algo: {formatted_full_env_id: cell_value}}
    standard_df = pd.DataFrame.from_dict(standard_table_data, orient="index")
    standard_df = standard_df.sort_index(axis=0)  # Sort rows (formatted algos)
    standard_df = standard_df.sort_index(
        axis=1
    )  # Sort columns (formatted full env IDs)
    standard_df = standard_df.fillna(nan_replacement)

    try:
        # Replace '_' in column names for LaTeX if necessary (pandas does some escaping, but _ needs care)
        # Or, just keep them as is if they look ok (Ant-v5_ant is probably fine)
        # standard_df.columns = [col.replace('_', '\_') for col in standard_df.columns] # Example if needed

        std_latex = standard_df.to_latex(
            index=True,
            index_names=True,
            na_rep=nan_replacement,
            caption="Algorithm Performance on Standard Environments (Mean $\\pm$ Std Return)",
            label="tab:perf_standard",
            escape=False,  # Allow \pm and potentially other LaTeX symbols in data
            bold_rows=False,
            column_format="l"
            + "c" * len(standard_df.columns),  # Auto-generate column format
        )
        print(std_latex)
        tex_filename_std = output_dir / "summary_standard_results.tex"
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
    # modified_tables_data: {formatted_base_env_name: {formatted_algo: {formatted_modification: cell_value}}}
    sorted_formatted_base_envs = sorted(modified_tables_data.keys())

    for formatted_base_env in sorted_formatted_base_envs:
        mod_data = modified_tables_data[
            formatted_base_env
        ]  # This is {formatted_algo: {formatted_modification: cell_value}}
        print(f"\n--- {formatted_base_env} Modified Environments ---")
        if not mod_data:
            print(f"  No data for {formatted_base_env} modified environments.")
            continue

        # Create DataFrame with formatted algo names as index & formatted modifications as columns
        mod_df = pd.DataFrame.from_dict(mod_data, orient="index")
        mod_df = mod_df.sort_index(axis=0)  # Sort rows (formatted algos)
        # Sort columns (formatted modifications) - pandas sorts strings alphabetically
        # This means '+20%' comes before '-20%'. If specific sorting (-50, -20, +20, +50) is needed,
        # you'd need a custom sort key function applied to mod_df.columns.
        mod_df = mod_df.sort_index(axis=1)
        mod_df = mod_df.fillna(nan_replacement)

        # Ensure column names don't contain problematic LaTeX characters like % or +
        # Pandas to_latex escapes '%' but '+' is usually safe outside math mode.
        # Our format_modification_details puts a '\%' which is already LaTeX escaped.
        # Let's check if any column names need explicit escaping.
        # Column names will be like 'Ant Gear -20\%'
        mod_df.columns = [
            col.replace("%", "\\%") for col in mod_df.columns
        ]  # Escape '%' if not already

        # --- Export Modified Table to LaTeX ---
        safe_base_env_name = re.sub(
            r"[^\w-]", "", formatted_base_env
        ).lower()  # Sanitize for filename, make lowercase
        tex_filename_mod = (
            output_dir / f"summary_modified_{safe_base_env_name}_results.tex"
        )
        try:
            mod_latex = mod_df.to_latex(
                index=True,
                index_names=True,
                na_rep=nan_replacement,
                caption=f"Algorithm Performance on Modified {formatted_base_env} Environments (Mean $\\pm$ Std Return)",
                label=f"tab:perf_mod_{safe_base_env_name}",  # Use safe name for label
                escape=False,  # Allow \pm, \% etc.
                bold_rows=False,
                column_format="l"
                + "c" * len(mod_df.columns),  # Auto-generate column format
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
