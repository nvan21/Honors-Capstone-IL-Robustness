import pandas as pd
import json
from pathlib import Path
import os
import re  # For regex parsing and filename sanitization
import numpy as np  # Needed for np.nan, np.isclose

# --- Configuration ---
runs_base_dir = "runs"  # The top-level directory containing experiment folders
output_precision = 2  # Number of decimal places for mean/std
nan_replacement = "-"  # How to display missing results
output_dir = Path("assets") / "tables"  # Directory to save LaTeX tables
combined_latex_file = (
    output_dir / "all_results_combined.tex"
)  # Name for the combined LaTeX file

# Ensure output directory exists
output_dir.mkdir(parents=True, exist_ok=True)

# Set to True to print detailed parsing and data storage info
DEBUG_MODE = False

# --- Formatting Maps ---
# Define how algorithm names should appear in the table index
ALGO_NAME_MAP = {
    "sac": "Expert SAC",
    "modified_sac": "Modified SAC",  # Assuming this might exist
    "dagger": "DAgger",
    "airl": "AIRL",
    "gail": "GAIL",
    "bc": "BC",
    # Add other algorithms here if needed, preserving case if input differs
}

# --- Utility Functions ---


def format_env_name_for_caption(base_env_name):
    """Formats the base environment name for captions (e.g., 'Ant' -> 'Ant')."""
    if not base_env_name:
        return ""
    return (
        base_env_name.capitalize()
    )  # Simple capitalization is usually sufficient for captions


# --- Function to format modification details ---
def format_modification_details(raw_suffix):
    """
    Formats raw modification suffixes like 'ant_gear_m20' or 'pusher_goal_shift'
    into human-readable strings like 'Gear -20%' or 'Goal Shift'.
    Removes the assumed entity prefix (like 'ant_', 'pusher_') and formats the rest.
    Handles entity_property_[p|m]amount pattern for specific formatting.
    """
    if DEBUG_MODE:
        print(f"    Formatting raw suffix: '{raw_suffix}'")

    if not raw_suffix or raw_suffix.lower() == "standard":
        # This function is primarily for modified suffixes, but this is a safe check
        print(
            f"  WARNING: format_modification_details called with unexpected input: '{raw_suffix}'"
        )
        return raw_suffix.replace("_", " ").capitalize()  # Fallback formatting

    # Try to split off the entity prefix (e.g., "ant_", "pusher_")
    parts = raw_suffix.split("_", 1)
    if len(parts) < 2:
        # If no underscore, treat the whole thing as the modification detail
        modification_part = raw_suffix
        if DEBUG_MODE:
            print(
                f"      No underscore found in raw suffix, treating whole as modification: '{modification_part}'"
            )
    else:
        # The part after the first underscore is the modification detail
        modification_part = parts[1]
        if DEBUG_MODE:
            print(
                f"      Split suffix into entity='{parts[0]}' and modification_part='{modification_part}'"
            )

    # Priority 1: Match the property_[p|m]amount pattern on the modification_part
    # Regex: (property)_(p|m)(digits)
    match_p_m = re.match(r"^(.*)_([pm])(\d+)$", modification_part, re.IGNORECASE)

    if match_p_m:
        property_raw = match_p_m.group(1)  # e.g., "gear", "torso_mass"
        modifier = match_p_m.group(2).lower()
        value = match_p_m.group(3)

        # Format property
        property_formatted = property_raw.replace("_", " ").title()

        # Determine sign
        sign = "+" if modifier == "p" else "-"

        formatted_string = f"{property_formatted} {sign}{value}\%"
        if DEBUG_MODE:
            print(
                f"      Matched p/m pattern on modification part, formatted as: '{formatted_string}'"
            )
        return formatted_string
    else:
        # Fallback for other modification patterns like 'goal_shift'
        # Just format the modification_part directly
        formatted_string = modification_part.replace("_", " ").capitalize()
        if DEBUG_MODE:
            print(
                f"      Did not match p/m pattern on modification part, used fallback: '{formatted_string}'"
            )
        return formatted_string


# --- Parsing XML Filename to get Env Details ---
def parse_xml_filename(xml_path_str):
    """
    Parses an XML filename (like runs/Ant-v5_ant/Ant-v5_ant_gear_m20.xml or runs/pusher_goal_shift/pusher_goal_shift.xml)
    to extract base environment name for grouping, and the identifier for the table column.
    Returns (base_env_name_for_grouping, identifier_for_table_column, is_standard).
    """
    xml_path = Path(xml_path_str)
    xml_filename_stem = (
        xml_path.stem
    )  # e.g., "Ant-v5_ant_gear_m20", "Ant-v5_ant", "pusher_goal_shift"

    if DEBUG_MODE:
        print(f"    Parsing xml_file stem: '{xml_filename_stem}'")

    # Try matching the BaseEnv-vX structure first, as it's common
    # Regex: (BaseEnv)-v(Version)_(Suffix, optional, non-greedy match)
    match_base_vx = re.match(r"^([a-zA-Z]+)-v(\d+)_?(.*)$", xml_filename_stem)

    if match_base_vx:
        base_raw = match_base_vx.group(1)  # e.g., "Ant"
        version = match_base_vx.group(2)  # e.g., "5"
        suffix = match_base_vx.group(
            3
        )  # e.g., "ant", "ant_gear_m20", or "" if just "Ant-v5.xml"

        base_env_name_formatted = base_raw.capitalize()  # e.g., "Ant"
        full_env_id = f"{base_env_name_formatted}-v{version}"  # e.g., "Ant-v5"

        # Determine if it's standard: suffix is empty or matches lowercase base name
        is_standard = suffix.lower() == base_raw.lower() or suffix == ""

        if DEBUG_MODE:
            print(
                f"      Matched BaseEnv-vX pattern: base_raw='{base_raw}', version='{version}', suffix='{suffix}'"
            )
            print(f"      is_standard based on BaseEnv-vX: {is_standard}")

        if is_standard:
            # Standard env column header is the full ID (e.g., Ant-v5)
            identifier = full_env_id
        else:
            # Modified env column identifier is the raw suffix (e.g., ant_gear_m20)
            # Base name for grouping is base_env_name_formatted (e.g., Ant)
            identifier = (
                suffix  # Suffix is the raw modification detail, needs formatting later
            )
            if DEBUG_MODE:
                print(
                    f"      Identified as modified, raw suffix for formatting: '{identifier}'"
                )

        if DEBUG_MODE:
            print(
                f"      Result: base_env_for_grouping='{base_env_name_formatted}', identifier_for_table_column='{identifier}', is_standard={is_standard}"
            )
        return base_env_name_formatted, identifier, is_standard

    else:
        # Fallback for filenames that do NOT match BaseEnv-vX_... pattern (e.g., "pusher_goal_shift")
        # Assume the first part before '_' is the base environment name for grouping.
        # Assume the entire stem is the raw modification suffix for the identifier.
        parts = xml_filename_stem.split("_", 1)
        base_raw = parts[0]  # e.g., "pusher"
        # suffix = parts[1] if len(parts) > 1 else "" # Not strictly needed here

        base_env_name_formatted = base_raw.capitalize()  # e.g., "Pusher"
        raw_modification_suffix = (
            xml_filename_stem  # Use the whole stem as the raw suffix for formatting
        )

        # In this fallback, we assume it's a modification UNLESS
        # the stem *is* just the base name itself (e.g. "pusher" or "ant").
        # This handles cases like "pusher.xml" or "ant.xml" if they exist as standard.
        is_standard_simple = xml_filename_stem.lower() == base_raw.lower()

        if DEBUG_MODE:
            print(f"      Did NOT match BaseEnv-vX pattern.")
            print(
                f"      Fallback split: base_raw='{base_raw}', raw_modification_suffix='{raw_modification_suffix}'"
            )
            print(f"      is_standard based on simple name match: {is_standard_simple}")

        if is_standard_simple:
            # Simple standard name (e.g. "pusher")
            identifier = base_env_name_formatted  # Column header is "Pusher"
            is_standard = True  # Explicitly set standard flag
        else:
            # Modification that didn't match BaseEnv-vX (e.g. "pusher_goal_shift")
            identifier = raw_modification_suffix  # Identifier is the raw suffix, needs formatting later
            is_standard = False  # Explicitly set standard flag
            if DEBUG_MODE:
                print(
                    f"      Identified as modified, raw suffix for formatting: '{identifier}'. Treating as modified '{base_env_name_for_grouping}'."
                )

        if DEBUG_MODE:
            print(
                f"      Result: base_env_for_grouping='{base_env_name_formatted}', identifier_for_table_column='{identifier}', is_standard={is_standard}"
            )
        return base_env_name_formatted, identifier, is_standard


# --- Function to extract numeric mean from cell value string ---
def extract_mean(cell_value_str):
    """
    Extracts the numeric mean from a string like '123.45 $\pm$ 6.78'.
    Returns float, or -inf if the string is the nan_replacement or not parseable.
    """
    if not isinstance(
        cell_value_str, str
    ):  # Handle potential None or other types before == nan_replacement
        return float("-inf")

    if cell_value_str.strip() == nan_replacement:
        return float("-inf")  # Use negative infinity so it's never the max

    # Remove LaTeX bolding if present for parsing
    cleaned_str = cell_value_str.replace("\\textbf{", "").replace("}", "")

    # Split by ' $\pm$ ' or just ' \pm ' (less strict) or other common separators
    # Regex to capture the first number before the +/- like symbol
    match = re.match(r"^\s*(-?\d+\.?\d*)\s*[$]?\\?pm", cleaned_str)

    if match:
        try:
            return float(match.group(1))
        except (ValueError, TypeError):
            if DEBUG_MODE:
                print(
                    f"    extract_mean: Failed to convert '{match.group(1)}' to float from string '{cell_value_str}'."
                )
            return float("-inf")  # Return negative infinity if parsing fails
    else:
        if DEBUG_MODE:
            print(
                f"    extract_mean: String '{cell_value_str}' did not match mean pattern."
            )
        return float("-inf")  # Return negative infinity if pattern doesn't match


# --- Function to apply bolding to maximums in a DataFrame ---
def bold_max_in_dataframe(df, nan_val=nan_replacement):
    """
    Modifies the DataFrame in place to bold the string value(s)
    corresponding to the maximum numeric mean in each column.
    """
    if df.empty:
        return df

    for col in df.columns:
        # Extract numeric means for this column
        col_means = df[col].apply(extract_mean)

        # Find the maximum mean value, ignoring -inf (our representation for NaN/unparseable)
        # Use a mask to exclude -inf when finding the max
        valid_means = col_means[col_means > float("-inf")]

        if valid_means.empty:
            # No valid numeric data in this column
            if DEBUG_MODE:
                print(f"    Column '{col}' has no valid numeric data for bolding.")
            continue

        max_mean = valid_means.max()

        if DEBUG_MODE:
            print(f"    Column '{col}': Max mean found = {max_mean}")

        # Find indices where the mean equals the maximum
        # Use np.isclose to handle floating point comparisons
        indices_to_bold = col_means[np.isclose(col_means, max_mean)].index

        if DEBUG_MODE:
            print(f"    Column '{col}': Indices to bold = {list(indices_to_bold)}")

        # Bold the original string values at these indices
        for idx in indices_to_bold:
            original_value = df.loc[idx, col]
            # Ensure we don't try to bold the nan_replacement itself if somehow indexed
            # Also ensure the original value is indeed the one corresponding to the max mean
            if original_value != nan_val and np.isclose(
                extract_mean(original_value), max_mean
            ):
                df.loc[idx, col] = f"\\textbf{{{original_value}}}"
            elif DEBUG_MODE:
                print(
                    f"      Skipping bold for '{original_value}' at index {idx}, column {col} (either nan or mean mismatch)."
                )

    return df  # Modified DataFrame


# --- Function to wrap tabular in adjustbox ---
def wrap_with_adjustbox(latex_table_string):
    """
    Wraps the tabular environment within an adjustbox for width control.
    Assumes the string contains one \begin{tabular}...\end{tabular} block
    and potentially \begin{table}...\end{table} around it.
    """
    # Find the tabular environment
    # This regex is simple and might fail for complex tabular environments.
    # A more robust parser would be needed for full LaTeX parsing.
    # This looks for \begin{tabular} and matches everything non-greedily until \end{tabular}.
    # Added \s* to account for potential whitespace before the opening brace of the arguments
    tabular_match = re.search(
        r"(\\begin{tabular}\s*{.*?}.*?\\end{tabular})", latex_table_string, re.DOTALL
    )

    if not tabular_match:
        print(
            "  WARNING: Could not find \\begin{tabular}...\\end{tabular} to wrap with adjustbox."
        )
        return latex_table_string  # Return original if not found

    tabular_block = tabular_match.group(1)

    # Find the table environment if it exists.
    # Look for \begin{table} followed by anything (\s\S)*? non-greedily, then the tabular block,
    # then anything (\s\S)*? non-greedily until \end{table}.
    table_match = re.search(
        r"(\\begin{table}.*?)(\s*\n*)("
        + re.escape(tabular_block)
        + r")(\s*\n*)(.*?\\end{table})",
        latex_table_string,
        re.DOTALL,
    )

    if table_match:
        # Table environment found, insert adjustbox inside
        before_tabular = table_match.group(
            1
        )  # stuff from \begin{table} up to just before \begin{tabular}
        whitespace_before = table_match.group(2)  # optional whitespace/newlines
        whitespace_after = table_match.group(4)  # optional whitespace/newlines
        after_tabular = table_match.group(
            5
        )  # stuff from just after \end{tabular} to \end{table}

        wrapped_block = f"""{whitespace_before}\\begin{{adjustbox}}{{max width=\\textwidth, center}}
{tabular_block}
\\end{{adjustbox}}{whitespace_after}"""  # Maintain original whitespace around tabular

        # Reconstruct the table with the wrapped tabular
        return f"{before_tabular}{wrapped_block}{after_tabular}"
    else:
        # No table environment, just wrap the tabular block directly
        if DEBUG_MODE:
            print(
                "  WARNING: Found tabular but no wrapping table environment. Wrapping just tabular."
            )

        wrapped_block = f"""\\begin{{adjustbox}}{{max width=\\textwidth, center}}
{tabular_block}
\\end{{adjustbox}}"""

        # Replace the original tabular block with the wrapped one
        return latex_table_string.replace(tabular_block, wrapped_block)


# ---------------------

# --- Data Storage Initialization ---
# standard_table_data: {formatted_algo: {full_standard_env_id_or_simple_name: cell_value}}
standard_table_data = {}

# modified_tables_data: {base_env_name_formatted: {formatted_algo: {formatted_modification: cell_value}}}
modified_tables_data = {}

# --- File Discovery ---
base_path = Path(runs_base_dir)
# Find results.json within any immediate subdirectory of the environment directories
json_files = list(base_path.rglob("*/results.json"))


if not json_files:
    print(
        f"ERROR: No 'results.json' files found anywhere within '{runs_base_dir}' in algorithm subdirectories."
    )
    # Don't exit immediately, print message and proceed if data structures are empty

print(f"Found {len(json_files)} results.json files.")

# --- Processing Loop ---
for json_path in json_files:
    # The json_path will be like runs/Ant-v5_ant/sac/results.json
    # The xml_file path inside results.json is like runs/Ant-v5_ant/Ant-v5_ant.xml
    print(f"Processing: {json_path}")
    try:
        # --- Load JSON Data ---
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # --- Extract Required Data from JSON ---
        algo_raw = data.get("algo")
        xml_file_path_in_json = data.get(
            "xml_file"
        )  # This is the key from the json file
        mean_ret = data.get("mean_return")
        std_ret = data.get("std_return")

        # --- Data Validation ---
        if not all(
            [algo_raw, xml_file_path_in_json, mean_ret is not None, std_ret is not None]
        ):
            print(
                f"  WARNING: Skipping - Missing required keys (algo, xml_file, mean_return, std_return) in {json_path}"
            )
            continue
        if not isinstance(xml_file_path_in_json, str):
            print(
                f"  WARNING: Skipping - 'xml_file' key in {json_path} is not a string."
            )
            continue

        # --- Parse Environment Details from XML Path string ---
        # We use the xml_file path found *inside* the results.json
        try:
            base_env_name_for_grouping, identifier_for_table_column, is_standard = (
                parse_xml_filename(xml_file_path_in_json)
            )
        except Exception as parse_e:
            print(
                f"  ERROR: Could not parse xml_file path '{xml_file_path_in_json}' from {json_path}: {parse_e}"
            )
            continue  # Skip this result if parsing fails

        # --- Format Names for Tables ---
        formatted_algo_name = ALGO_NAME_MAP.get(algo_raw.lower(), algo_raw.upper())

        # --- Format Output String ---
        try:
            # Check if mean_ret/std_ret are list/tuple and take the first element if needed
            if isinstance(mean_ret, (list, tuple)):
                mean_ret = mean_ret[0]
            if isinstance(std_ret, (list, tuple)):
                std_ret = std_ret[0]

            mean_str = f"{float(mean_ret):.{output_precision}f}"
            std_str = f"{float(std_ret):.{output_precision}f}"
            # Enclose \pm in $ for LaTeX math mode
            cell_value = f"{mean_str} $\\pm$ {std_str}"
        except (ValueError, TypeError, IndexError) as e:
            print(
                f"  WARNING: Skipping - Error formatting mean/std return (must be numeric, list/tuple supported) in {json_path}: {e}"
            )
            continue

        # --- Store Data in Appropriate Dictionary ---
        if DEBUG_MODE:
            print(
                f"  Storing data: Algo='{formatted_algo_name}', BaseEnv='{base_env_name_for_grouping}', Identifier='{identifier_for_table_column}', IsStandard={is_standard}"
            )
            print(f"  Cell value: '{cell_value}'")

        if is_standard:
            # Standard table: Algo (row) vs Full Standard Env ID or simple name (col)
            # identifier_for_table_column is like "Ant-v5" or "Pusher"
            standard_table_data.setdefault(formatted_algo_name, {})[
                identifier_for_table_column
            ] = cell_value
        else:
            # Modified tables: Grouped by Base Env Name, Algo (row) vs Formatted Modification (col)
            # base_env_name_for_grouping is like "Ant" or "Pusher"
            # identifier_for_table_column is the raw suffix like "ant_gear_m20" or "pusher_goal_shift"
            raw_modification_suffix = identifier_for_table_column
            # The base_env_name_for_grouping is NOT used in format_modification_details anymore
            formatted_modification = format_modification_details(
                raw_modification_suffix
            )

            modified_tables_data.setdefault(
                base_env_name_for_grouping, {}
            )  # Key is like "Ant" or "Pusher"
            modified_tables_data[base_env_name_for_grouping].setdefault(
                formatted_algo_name, {}
            )
            modified_tables_data[base_env_name_for_grouping][formatted_algo_name][
                formatted_modification
            ] = cell_value

    except json.JSONDecodeError:
        print(f"  ERROR: Could not decode JSON from {json_path}")
    except IOError as e:
        print(f"  ERROR: Could not read file {json_path}: {e}")
    except Exception as e:
        print(f"  ERROR: An unexpected error occurred processing {json_path}: {e}")

# --- Print final data structures in debug mode ---
if DEBUG_MODE:
    print("\n--- Final Data Structures ---")
    print("standard_table_data:")
    print(json.dumps(standard_table_data, indent=2))
    print("\nmodified_tables_data:")
    print(json.dumps(modified_tables_data, indent=2))
    print("-----------------------------\n")


# --- Function to extract numeric mean from cell value string ---
def extract_mean(cell_value_str):
    """
    Extracts the numeric mean from a string like '123.45 $\pm$ 6.78'.
    Returns float, or -inf if the string is the nan_replacement or not parseable.
    Handles strings potentially wrapped in \textbf{}.
    """
    if not isinstance(cell_value_str, str):
        return float("-inf")

    cleaned_str = cell_value_str.strip()

    if cleaned_str == nan_replacement:
        return float("-inf")  # Use negative infinity so it's never the max

    # Remove LaTeX bolding if present before parsing
    cleaned_str = cleaned_str.replace("\\textbf{", "").replace("}", "")

    # Regex to capture the first number before the +/- like symbol
    # Allows for negative numbers, decimals, optional $ and optional \ before pm
    match = re.match(r"^\s*(-?\d+\.?\d*)\s*[$]?\\?pm", cleaned_str)

    if match:
        try:
            return float(match.group(1))
        except (ValueError, TypeError):
            if DEBUG_MODE:
                print(
                    f"    extract_mean: Failed to convert '{match.group(1)}' to float from string '{cell_value_str}'."
                )
            return float("-inf")  # Return negative infinity if parsing fails
    else:
        if DEBUG_MODE:
            print(
                f"    extract_mean: String '{cell_value_str}' did not match mean pattern after cleaning."
            )
        return float("-inf")  # Return negative infinity if pattern doesn't match


# --- Function to apply bolding to maximums in a DataFrame ---
def bold_max_in_dataframe(df, nan_val=nan_replacement):
    """
    Modifies the DataFrame in place to bold the string value(s)
    corresponding to the maximum numeric mean in each column.
    """
    if df.empty:
        return df

    for col in df.columns:
        # Extract numeric means for this column
        col_means = df[col].apply(extract_mean)

        # Find the maximum mean value, ignoring -inf (our representation for NaN/unparseable)
        # Use a mask to exclude -inf when finding the max
        valid_means = col_means[col_means > float("-inf")]

        if valid_means.empty:
            # No valid numeric data in this column
            if DEBUG_MODE:
                print(f"    Column '{col}' has no valid numeric data for bolding.")
            continue

        max_mean = valid_means.max()

        # Handle edge case where max_mean is -inf (e.g., all cells were non-numeric)
        if max_mean <= float("-inf"):
            if DEBUG_MODE:
                print(f"    Column '{col}': Max mean is -inf, skipping bolding.")
            continue

        if DEBUG_MODE:
            print(f"    Column '{col}': Max mean found = {max_mean}")

        # Find indices where the mean equals the maximum using the *original* string value
        indices_to_bold = [
            idx
            for idx, val in df[col].items()
            if val != nan_val and np.isclose(extract_mean(val), max_mean)
        ]

        if DEBUG_MODE:
            print(f"    Column '{col}': Indices to bold = {list(indices_to_bold)}")

        # Bold the original string values at these indices
        for idx in indices_to_bold:
            original_value = df.loc[idx, col]
            # Double check we don't re-bold if it somehow already is (shouldn't happen with this script)
            # and ensure it's not the nan_replacement
            if original_value != nan_val and not original_value.strip().startswith(
                "\\textbf{"
            ):
                df.loc[idx, col] = f"\\textbf{{{original_value}}}"
            elif DEBUG_MODE:
                print(
                    f"      Skipping bold for '{original_value}' at index {idx}, column {col} (either nan or already bold)."
                )

    return df  # Modified DataFrame


# --- Function to wrap tabular in adjustbox ---
def wrap_with_adjustbox(latex_table_string):
    """
    Wraps the tabular environment within an adjustbox for width control.
    Assumes the string contains one \begin{tabular}...\end{tabular} block
    and potentially \begin{table}...\end{table} around it.
    """
    # Find the tabular environment
    # This regex is simple and might fail for complex tabular environments.
    # A more robust parser would be needed for full LaTeX parsing.
    # This looks for \begin{tabular} and matches everything non-greedily until \end{tabular}.
    # Added \s* to account for potential whitespace before the opening brace of the arguments
    # Added \n*? to account for newlines
    tabular_match = re.search(
        r"(\\begin{tabular}\s*{.*?}\s*\n*.*?\\end{tabular})",
        latex_table_string,
        re.DOTALL,
    )

    if not tabular_match:
        print(
            "  WARNING: Could not find \\begin{tabular}...\\end{tabular} to wrap with adjustbox."
        )
        return latex_table_string  # Return original if not found

    tabular_block = tabular_match.group(1)

    # Find the table environment if it exists.
    # Look for \begin{table} followed by anything (\s\S)*? non-greedily, then the tabular block,
    # then anything (\s\S)*? non-greedily until \end{table}.
    table_match = re.search(
        r"(\\begin{table}.*?)(\s*\n*)("
        + re.escape(tabular_block)
        + r")(\s*\n*)(.*?\\end{table})",
        latex_table_string,
        re.DOTALL,
    )

    if table_match:
        # Table environment found, insert adjustbox inside
        before_tabular = table_match.group(
            1
        )  # stuff from \begin{table} up to just before \begin{tabular}
        whitespace_before = table_match.group(2)  # optional whitespace/newlines
        whitespace_after = table_match.group(4)  # optional whitespace/newlines
        after_tabular = table_match.group(
            5
        )  # stuff from just after \end{tabular} to \end{table}

        wrapped_block = f"""{whitespace_before}\\begin{{adjustbox}}{{max width=\\textwidth, center}}
{tabular_block}
\\end{{adjustbox}}{whitespace_after}"""  # Maintain original whitespace around tabular

        # Reconstruct the table with the wrapped tabular
        return f"{before_tabular}{wrapped_block}{after_tabular}"
    else:
        # No table environment, just wrap the tabular block directly
        if DEBUG_MODE:
            print(
                "  WARNING: Found tabular but no wrapping table environment. Wrapping just tabular."
            )

        wrapped_block = f"""\\begin{{adjustbox}}{{max width=\\textwidth, center}}
{tabular_block}
\\end{{adjustbox}}"""

        # Replace the original tabular block with the wrapped one
        return latex_table_string.replace(tabular_block, wrapped_block)


# ---------------------

# --- Data Storage Initialization ---
# standard_table_data: {formatted_algo: {full_standard_env_id_or_simple_name: cell_value}}
standard_table_data = {}

# modified_tables_data: {base_env_name_formatted: {formatted_algo: {formatted_modification: cell_value}}}
modified_tables_data = {}

# --- File Discovery ---
base_path = Path(runs_base_dir)
# Find results.json within any immediate subdirectory of the environment directories
json_files = list(base_path.rglob("*/results.json"))


if not json_files:
    print(
        f"ERROR: No 'results.json' files found anywhere within '{runs_base_dir}' in algorithm subdirectories."
    )
    # Don't exit immediately, print message and proceed if data structures are empty

print(f"Found {len(json_files)} results.json files.")

# --- Processing Loop ---
for json_path in json_files:
    # The json_path will be like runs/Ant-v5_ant/sac/results.json
    # The xml_file path inside results.json is like runs/Ant-v5_ant/Ant-v5_ant.xml
    print(f"Processing: {json_path}")
    try:
        # --- Load JSON Data ---
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # --- Extract Required Data from JSON ---
        algo_raw = data.get("algo")
        xml_file_path_in_json = data.get(
            "xml_file"
        )  # This is the key from the json file
        mean_ret = data.get("mean_return")
        std_ret = data.get("std_return")

        # --- Data Validation ---
        if not all(
            [algo_raw, xml_file_path_in_json, mean_ret is not None, std_ret is not None]
        ):
            print(
                f"  WARNING: Skipping - Missing required keys (algo, xml_file, mean_return, std_return) in {json_path}"
            )
            continue
        if not isinstance(xml_file_path_in_json, str):
            print(
                f"  WARNING: Skipping - 'xml_file' key in {json_path} is not a string."
            )
            continue

        # --- Parse Environment Details from XML Path string ---
        # We use the xml_file path found *inside* the results.json
        try:
            base_env_name_for_grouping, identifier_for_table_column, is_standard = (
                parse_xml_filename(xml_file_path_in_json)
            )
        except Exception as parse_e:
            print(
                f"  ERROR: Could not parse xml_file path '{xml_file_path_in_json}' from {json_path}: {parse_e}"
            )
            continue  # Skip this result if parsing fails

        # --- Format Names for Tables ---
        formatted_algo_name = ALGO_NAME_MAP.get(algo_raw.lower(), algo_raw.upper())

        # --- Format Output String ---
        try:
            # Check if mean_ret/std_ret are list/tuple and take the first element if needed
            if isinstance(mean_ret, (list, tuple)):
                mean_ret = mean_ret[0]
            if isinstance(std_ret, (list, tuple)):
                std_ret = std_ret[0]

            mean_str = f"{float(mean_ret):.{output_precision}f}"
            std_str = f"{float(std_ret):.{output_precision}f}"
            # Enclose \pm in $ for LaTeX math mode
            cell_value = f"{mean_str} $\\pm$ {std_str}"
        except (ValueError, TypeError, IndexError) as e:
            print(
                f"  WARNING: Skipping - Error formatting mean/std return (must be numeric, list/tuple supported) in {json_path}: {e}"
            )
            continue

        # --- Store Data in Appropriate Dictionary ---
        if DEBUG_MODE:
            print(
                f"  Storing data: Algo='{formatted_algo_name}', BaseEnv='{base_env_name_for_grouping}', Identifier='{identifier_for_table_column}', IsStandard={is_standard}"
            )
            print(f"  Cell value: '{cell_value}'")

        if is_standard:
            # Standard table: Algo (row) vs Full Standard Env ID or simple name (col)
            # identifier_for_table_column is like "Ant-v5" or "Pusher"
            standard_table_data.setdefault(formatted_algo_name, {})[
                identifier_for_table_column
            ] = cell_value
        else:
            # Modified tables: Grouped by Base Env Name, Algo (row) vs Formatted Modification (col)
            # base_env_name_for_grouping is like "Ant" or "Pusher"
            # identifier_for_table_column is the raw suffix like "ant_gear_m20" or "pusher_goal_shift"
            raw_modification_suffix = identifier_for_table_column
            # The base_env_name_for_grouping is NOT used in format_modification_details anymore
            formatted_modification = format_modification_details(
                raw_modification_suffix
            )

            modified_tables_data.setdefault(
                base_env_name_for_grouping, {}
            )  # Key is like "Ant" or "Pusher"
            modified_tables_data[base_env_name_for_grouping].setdefault(
                formatted_algo_name, {}
            )
            modified_tables_data[base_env_name_for_grouping][formatted_algo_name][
                formatted_modification
            ] = cell_value

    except json.JSONDecodeError:
        print(f"  ERROR: Could not decode JSON from {json_path}")
    except IOError as e:
        print(f"  ERROR: Could not read file {json_path}: {e}")
    except Exception as e:
        print(f"  ERROR: An unexpected error occurred processing {json_path}: {e}")

# --- Print final data structures in debug mode ---
if DEBUG_MODE:
    print("\n--- Final Data Structures ---")
    print("standard_table_data:")
    print(json.dumps(standard_table_data, indent=2))
    print("\nmodified_tables_data:")
    print(json.dumps(modified_tables_data, indent=2))
    print("-----------------------------\n")


# --- Function to extract numeric mean from cell value string ---
def extract_mean(cell_value_str):
    """
    Extracts the numeric mean from a string like '123.45 $\pm$ 6.78'.
    Returns float, or -inf if the string is the nan_replacement or not parseable.
    Handles strings potentially wrapped in \textbf{}.
    """
    if not isinstance(cell_value_str, str):
        return float("-inf")

    cleaned_str = cell_value_str.strip()

    if cleaned_str == nan_replacement:
        return float("-inf")  # Use negative infinity so it's never the max

    # Remove LaTeX bolding if present before parsing
    cleaned_str = cleaned_str.replace("\\textbf{", "").replace("}", "")

    # Regex to capture the first number before the +/- like symbol
    # Allows for negative numbers, decimals, optional $ and optional \ before pm
    match = re.match(r"^\s*(-?\d+\.?\d*)\s*[$]?\\?pm", cleaned_str)

    if match:
        try:
            return float(match.group(1))
        except (ValueError, TypeError):
            if DEBUG_MODE:
                print(
                    f"    extract_mean: Failed to convert '{match.group(1)}' to float from string '{cell_value_str}'."
                )
            return float("-inf")  # Return negative infinity if parsing fails
    else:
        if DEBUG_MODE:
            print(
                f"    extract_mean: String '{cell_value_str}' did not match mean pattern after cleaning."
            )
        return float("-inf")  # Return negative infinity if pattern doesn't match


# --- Function to apply bolding to maximums in a DataFrame ---
def bold_max_in_dataframe(df, nan_val=nan_replacement):
    """
    Modifies the DataFrame in place to bold the string value(s)
    corresponding to the maximum numeric mean in each column.
    """
    if df.empty:
        return df

    for col in df.columns:
        # Extract numeric means for this column
        col_means = df[col].apply(extract_mean)

        # Find the maximum mean value, ignoring -inf (our representation for NaN/unparseable)
        # Use a mask to exclude -inf when finding the max
        valid_means = col_means[col_means > float("-inf")]

        if valid_means.empty:
            # No valid numeric data in this column
            if DEBUG_MODE:
                print(f"    Column '{col}' has no valid numeric data for bolding.")
            continue

        max_mean = valid_means.max()

        # Handle edge case where max_mean is -inf (e.g., all cells were non-numeric)
        if max_mean <= float("-inf"):
            if DEBUG_MODE:
                print(f"    Column '{col}': Max mean is -inf, skipping bolding.")
            continue

        if DEBUG_MODE:
            print(f"    Column '{col}': Max mean found = {max_mean}")

        # Find indices where the mean equals the maximum using the *original* string value
        indices_to_bold = [
            idx
            for idx, val in df[col].items()
            if val != nan_val and np.isclose(extract_mean(val), max_mean)
        ]

        if DEBUG_MODE:
            print(f"    Column '{col}': Indices to bold = {list(indices_to_bold)}")

        # Bold the original string values at these indices
        for idx in indices_to_bold:
            original_value = df.loc[idx, col]
            # Double check we don't re-bold if it somehow already is (shouldn't happen with this script)
            # and ensure it's not the nan_replacement
            if original_value != nan_val and not original_value.strip().startswith(
                "\\textbf{"
            ):
                df.loc[idx, col] = f"\\textbf{{{original_value}}}"
            elif DEBUG_MODE:
                print(
                    f"      Skipping bold for '{original_value}' at index {idx}, column {col} (either nan or already bold)."
                )

    return df  # Modified DataFrame


# --- Function to wrap tabular in adjustbox ---
def wrap_with_adjustbox(latex_table_string):
    """
    Wraps the tabular environment within an adjustbox for width control.
    Assumes the string contains one \begin{tabular}...\end{tabular} block
    and potentially \begin{table}...\end{table} around it.
    """
    # Find the tabular environment
    # This regex is simple and might fail for complex tabular environments.
    # A more robust parser would be needed for full LaTeX parsing.
    # This looks for \begin{tabular} and matches everything non-greedily until \end{tabular}.
    # Added \s* to account for potential whitespace before the opening brace of the arguments
    # Added \n*? to account for newlines
    tabular_match = re.search(
        r"(\\begin{tabular}\s*{.*?}\s*\n*.*?\\end{tabular})",
        latex_table_string,
        re.DOTALL,
    )

    if not tabular_match:
        print(
            "  WARNING: Could not find \\begin{tabular}...\\end{tabular} to wrap with adjustbox."
        )
        return latex_table_string  # Return original if not found

    tabular_block = tabular_match.group(1)

    # Find the table environment if it exists.
    # Look for \begin{table} followed by anything (\s\S)*? non-greedily, then the tabular block,
    # then anything (\s\S)*? non-greedily until \end{table}.
    table_match = re.search(
        r"(\\begin{table}.*?)(\s*\n*)("
        + re.escape(tabular_block)
        + r")(\s*\n*)(.*?\\end{table})",
        latex_table_string,
        re.DOTALL,
    )

    if table_match:
        # Table environment found, insert adjustbox inside
        before_tabular = table_match.group(
            1
        )  # stuff from \begin{table} up to just before \begin{tabular}
        whitespace_before = table_match.group(2)  # optional whitespace/newlines
        whitespace_after = table_match.group(4)  # optional whitespace/newlines
        after_tabular = table_match.group(
            5
        )  # stuff from just after \end{tabular} to \end{table}

        wrapped_block = f"""{whitespace_before}\\begin{{adjustbox}}{{max width=\\textwidth, center}}
{tabular_block}
\\end{{adjustbox}}{whitespace_after}"""  # Maintain original whitespace around tabular

        # Reconstruct the table with the wrapped tabular
        return f"{before_tabular}{wrapped_block}{after_tabular}"
    else:
        # No table environment, just wrap the tabular block directly
        if DEBUG_MODE:
            print(
                "  WARNING: Found tabular but no wrapping table environment. Wrapping just tabular."
            )

        wrapped_block = f"""\\begin{{adjustbox}}{{max width=\\textwidth, center}}
{tabular_block}
\\end{{adjustbox}}"""

        # Replace the original tabular block with the wrapped one
        return latex_table_string.replace(tabular_block, wrapped_block)


# ---------------------

# --- Data Storage Initialization ---
# standard_table_data: {formatted_algo: {full_standard_env_id_or_simple_name: cell_value}}
standard_table_data = {}

# modified_tables_data: {base_env_name_formatted: {formatted_algo: {formatted_modification: cell_value}}}
modified_tables_data = {}

# --- File Discovery ---
base_path = Path(runs_base_dir)
# Find results.json within any immediate subdirectory of the environment directories
json_files = list(base_path.rglob("*/results.json"))


if not json_files:
    print(
        f"ERROR: No 'results.json' files found anywhere within '{runs_base_dir}' in algorithm subdirectories."
    )
    # Don't exit immediately, print message and proceed if data structures are empty

print(f"Found {len(json_files)} results.json files.")

# --- Processing Loop ---
for json_path in json_files:
    # The json_path will be like runs/Ant-v5_ant/sac/results.json
    # The xml_file path inside results.json is like runs/Ant-v5_ant/Ant-v5_ant.xml
    print(f"Processing: {json_path}")
    try:
        # --- Load JSON Data ---
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # --- Extract Required Data from JSON ---
        algo_raw = data.get("algo")
        xml_file_path_in_json = data.get(
            "xml_file"
        )  # This is the key from the json file
        mean_ret = data.get("mean_return")
        std_ret = data.get("std_return")

        # --- Data Validation ---
        if not all(
            [algo_raw, xml_file_path_in_json, mean_ret is not None, std_ret is not None]
        ):
            print(
                f"  WARNING: Skipping - Missing required keys (algo, xml_file, mean_return, std_return) in {json_path}"
            )
            continue
        if not isinstance(xml_file_path_in_json, str):
            print(
                f"  WARNING: Skipping - 'xml_file' key in {json_path} is not a string."
            )
            continue

        # --- Parse Environment Details from XML Path string ---
        # We use the xml_file path found *inside* the results.json
        try:
            base_env_name_for_grouping, identifier_for_table_column, is_standard = (
                parse_xml_filename(xml_file_path_in_json)
            )
        except Exception as parse_e:
            print(
                f"  ERROR: Could not parse xml_file path '{xml_file_path_in_json}' from {json_path}: {parse_e}"
            )
            continue  # Skip this result if parsing fails

        # --- Format Names for Tables ---
        formatted_algo_name = ALGO_NAME_MAP.get(algo_raw.lower(), algo_raw.upper())

        # --- Format Output String ---
        try:
            # Check if mean_ret/std_ret are list/tuple and take the first element if needed
            if isinstance(mean_ret, (list, tuple)):
                mean_ret = mean_ret[0]
            if isinstance(std_ret, (list, tuple)):
                std_ret = std_ret[0]

            mean_str = f"{float(mean_ret):.{output_precision}f}"
            std_str = f"{float(std_ret):.{output_precision}f}"
            # Enclose \pm in $ for LaTeX math mode
            cell_value = f"{mean_str} $\\pm$ {std_str}"
        except (ValueError, TypeError, IndexError) as e:
            print(
                f"  WARNING: Skipping - Error formatting mean/std return (must be numeric, list/tuple supported) in {json_path}: {e}"
            )
            continue

        # --- Store Data in Appropriate Dictionary ---
        if DEBUG_MODE:
            print(
                f"  Storing data: Algo='{formatted_algo_name}', BaseEnv='{base_env_name_for_grouping}', Identifier='{identifier_for_table_column}', IsStandard={is_standard}"
            )
            print(f"  Cell value: '{cell_value}'")

        if is_standard:
            # Standard table: Algo (row) vs Full Standard Env ID or simple name (col)
            # identifier_for_table_column is like "Ant-v5" or "Pusher"
            standard_table_data.setdefault(formatted_algo_name, {})[
                identifier_for_table_column
            ] = cell_value
        else:
            # Modified tables: Grouped by Base Env Name, Algo (row) vs Formatted Modification (col)
            # base_env_name_for_grouping is like "Ant" or "Pusher"
            # identifier_for_table_column is the raw suffix like "ant_gear_m20" or "pusher_goal_shift"
            raw_modification_suffix = identifier_for_table_column
            # The base_env_name_for_grouping is NOT used in format_modification_details anymore
            formatted_modification = format_modification_details(
                raw_modification_suffix
            )

            modified_tables_data.setdefault(
                base_env_name_for_grouping, {}
            )  # Key is like "Ant" or "Pusher"
            modified_tables_data[base_env_name_for_grouping].setdefault(
                formatted_algo_name, {}
            )
            modified_tables_data[base_env_name_for_grouping][formatted_algo_name][
                formatted_modification
            ] = cell_value

    except json.JSONDecodeError:
        print(f"  ERROR: Could not decode JSON from {json_path}")
    except IOError as e:
        print(f"  ERROR: Could not read file {json_path}: {e}")
    except Exception as e:
        print(f"  ERROR: An unexpected error occurred processing {json_path}: {e}")

# --- Print final data structures in debug mode ---
if DEBUG_MODE:
    print("\n--- Final Data Structures ---")
    print("standard_table_data:")
    print(json.dumps(standard_table_data, indent=2))
    print("\nmodified_tables_data:")
    print(json.dumps(modified_tables_data, indent=2))
    print("-----------------------------\n")

# --- Collect all table LaTeX strings ---
all_tables_latex_content = []

# 1. Standard Environments Table
if not standard_table_data:
    print("No data found for the standard environments table.")
else:
    print("--- Generating Standard Environments Table String ---")
    standard_df = pd.DataFrame.from_dict(standard_table_data, orient="index")
    standard_df = standard_df.sort_index(axis=0)  # Sort rows (formatted algos)
    standard_df = standard_df.sort_index(
        axis=1
    )  # Sort columns (full standard env IDs/names, e.g., Ant-v5, Pusher)
    standard_df = standard_df.fillna(nan_replacement)

    # Apply bolding to the DataFrame
    if DEBUG_MODE:
        print("  Applying bolding to standard DataFrame...")
    bold_max_in_dataframe(standard_df, nan_val=nan_replacement)

    try:
        std_latex = standard_df.to_latex(
            index=True,
            index_names=True,
            na_rep=nan_replacement,
            caption="Algorithm Performance on Standard Environments (Mean $\\pm$ Std Return)",
            label="tab:perf_standard",
            escape=False,  # Allow $ \pm $ in cells, and \% etc. \textbf{} is safe with escape=False
            bold_rows=False,  # We are doing custom bolding per cell
            column_format="l"
            + "c" * len(standard_df.columns),  # Auto-generate column format
        )
        # Wrap the generated LaTeX string
        wrapped_std_latex = wrap_with_adjustbox(std_latex)
        all_tables_latex_content.append(f"% --- Standard Environment Results ---")
        all_tables_latex_content.append(wrapped_std_latex)
        print("Standard table string collected and wrapped.")
    except Exception as e:
        print(f"\nError generating standard table LaTeX string: {e}")


# 2. Modified Environments Tables (One per base_env)
if not modified_tables_data:
    print("\nNo data found for any modified environments.")
else:
    print("\n--- Generating Modified Environments Table Strings ---")
    # modified_tables_data: {base_env_name_formatted: {formatted_algo: {formatted_modification: cell_value}}}
    sorted_formatted_base_envs = sorted(modified_tables_data.keys())

    for base_env_name_for_grouping in sorted_formatted_base_envs:
        mod_data = modified_tables_data[
            base_env_name_for_grouping
        ]  # This is {formatted_algo: {formatted_modification: cell_value}}
        print(f"--- Processing {base_env_name_for_grouping} Modified Environments ---")
        if not mod_data:
            print(f"  No data for {base_env_name_for_grouping} modified environments.")
            continue

        # Create DataFrame with formatted algo names as index & formatted modifications as columns
        mod_df = pd.DataFrame.from_dict(mod_data, orient="index")
        mod_df = mod_df.sort_index(axis=0)  # Sort rows (formatted algos)
        # Sort columns (formatted modifications). Default string sort is used.
        mod_df = mod_df.sort_index(axis=1)
        mod_df = mod_df.fillna(nan_replacement)

        # Apply bolding to the DataFrame
        if DEBUG_MODE:
            print(
                f"  Applying bolding to '{base_env_name_for_grouping}' modified DataFrame..."
            )
        bold_max_in_dataframe(mod_df, nan_val=nan_replacement)

        # --- Generate Modified Table LaTeX String ---
        # Sanitize base env name for label (not strictly needed in combined raw output, but good practice)
        safe_base_env_name = re.sub(r"[^\w-]", "", base_env_name_for_grouping).lower()

        # Use the base environment name for the caption
        caption_base_env = format_env_name_for_caption(base_env_name_for_grouping)

        try:
            mod_latex = mod_df.to_latex(
                index=True,
                index_names=True,
                na_rep=nan_replacement,
                caption=f"Algorithm Performance on Modified {caption_base_env} Environments (Mean $\\pm$ Std Return)",
                label=f"tab:perf_mod_{safe_base_env_name}",  # Use safe name for label
                escape=False,  # Allow $ \pm $ in cells, and \% etc. \textbf{} is safe with escape=False
                bold_rows=False,  # We are doing custom bolding per cell
                column_format="l"
                + "c" * len(mod_df.columns),  # Auto-generate column format
            )
            # Wrap the generated LaTeX string
            wrapped_mod_latex = wrap_with_adjustbox(mod_latex)
            all_tables_latex_content.append(
                f"\n% --- Modified {base_env_name_for_grouping} Results ---"
            )
            all_tables_latex_content.append(wrapped_mod_latex)
            print(
                f"{base_env_name_for_grouping} modified table string collected and wrapped."
            )
        except Exception as e:
            print(
                f"\nError generating modified table LaTeX string for {base_env_name_for_grouping}: {e}"
            )

# --- Write Combined LaTeX File ---

print(f"\nWriting all table LaTeX code to {combined_latex_file}")

if all_tables_latex_content:
    try:
        with open(combined_latex_file, "w", encoding="utf-8") as f:
            # Join the collected strings. Add an extra newline between tables for readability in the output file.
            # Also add a helpful comment at the top
            f.write("% This file contains raw LaTeX table code.\n")
            f.write(
                "% To use these tables, include \\usepackage{adjustbox} in your main LaTeX document preamble.\n"
            )
            f.write(
                "% You will also likely need \\usepackage{booktabs}, \\usepackage{amsmath}, and \\usepackage{array}.\n"
            )
            f.write("% Adjust \\textwidth in the adjustbox options if needed.\n\n")
            f.write("\n\n".join(all_tables_latex_content))
        print(f"Combined LaTeX table code saved to {combined_latex_file}")
        print(
            "\nRemember to include \\usepackage{adjustbox} in your main LaTeX document preamble."
        )
        print("You may also need booktabs, amsmath, and array.")
        print("You can now copy the content of this file into your main .tex file.")
    except Exception as e:
        print(f"\nError writing combined LaTeX file {combined_latex_file}: {e}")
else:
    print("No table content was generated, combined file will not be created.")


print("\n" + "=" * 50)
print("Script finished.")
print("=" * 50)
