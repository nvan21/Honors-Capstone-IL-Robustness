import pandas as pd
import json
from pathlib import Path
import os
import re  # For regex parsing and filename sanitization
import numpy as np  # Needed for np.nan, np.arange
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker  # For formatting axes if needed
import matplotlib.cm as cm  # For colormaps

# --- Configuration ---
runs_base_dir = "runs"  # The top-level directory containing experiment folders
nan_replacement_val = np.nan  # Use NaN for missing numeric data for plotting
# Directory to save the generated plots
plot_output_dir = Path("assets") / "plots"

# Plotting Style and Aesthetics
plt.style.use("seaborn-v0_8-whitegrid")  # Corrected style name
plt.rcParams["figure.dpi"] = 100
plt.rcParams["savefig.dpi"] = 300
# --- Font Changes ---
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [
    "DejaVu Sans",
    "Arial",
    "Helvetica",
    "sans-serif",
]  # Specify preferred fonts
plt.rcParams["font.size"] = 10
plt.rcParams["axes.labelsize"] = 11
# --- Title Changes ---
plt.rcParams["axes.titlesize"] = 14  # Increased size
plt.rcParams["axes.titleweight"] = "bold"  # Make title bold
plt.rcParams["xtick.labelsize"] = 9
plt.rcParams["ytick.labelsize"] = 9
plt.rcParams["legend.fontsize"] = 9
plt.rcParams["legend.title_fontsize"] = 10
# Increase space between groups multiplier for modified plots
mod_plot_width_multiplier = 0.45


# Ensure output directory exists
plot_output_dir.mkdir(parents=True, exist_ok=True)

# Set to True to print detailed parsing and data storage info
DEBUG_MODE = False

# --- Formatting Maps ---
# Define how algorithm names should appear in the plot legends
ALGO_NAME_MAP = {
    "sac": "Expert SAC",
    "modified_sac": "Modified SAC",
    "dagger": "DAgger",
    "airl": "AIRL",
    "gail": "GAIL",
    "bc": "BC",
    # Add other algorithms here if needed, preserving case if input differs
}

# Define distinct colors for algorithms (add more if needed)
DISTINCT_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#aec7e8",
    "#ffbb78",
    "#98df8a",
    "#ff9896",
    "#c5b0d5",
    "#c49c94",
    "#f7b6d2",
    "#c7c7c7",
    "#dbdb8d",
    "#9edae5",
]

# --- Utility Functions ---


def format_env_name_for_caption(base_env_name):
    """Formats the base environment name for plot titles/filenames."""
    if not base_env_name:
        return ""
    if base_env_name.lower() == "invpend":
        return "Inverted Pendulum"
    return base_env_name.capitalize()


def format_modification_details(raw_suffix):
    """Formats raw modification suffixes into human-readable strings."""
    if DEBUG_MODE:
        print(f"    Formatting raw suffix: '{raw_suffix}'")
    if not raw_suffix or raw_suffix.lower() == "standard":
        print(
            f"  WARNING: format_modification_details called with unexpected input: '{raw_suffix}'"
        )
        return raw_suffix.replace("_", " ").capitalize()

    parts = raw_suffix.split("_", 1)
    modification_part = parts[1] if len(parts) > 1 else raw_suffix
    match_p_m = re.match(r"^(.*)_([pm])(\d+)$", modification_part, re.IGNORECASE)

    if match_p_m:
        property_raw = match_p_m.group(1)
        modifier = match_p_m.group(2).lower()
        value = match_p_m.group(3)
        property_formatted = property_raw.replace("_", " ").title()
        sign = "+" if modifier == "p" else "-"
        return f"{property_formatted} {sign}{value}%"
    else:
        return modification_part.replace("_", " ").capitalize()


# --- Parsing XML Filename to get Env Details ---
def parse_xml_filename(xml_path_str):
    """Parses an XML filename."""
    xml_path = Path(xml_path_str)
    xml_filename_stem = xml_path.stem
    if DEBUG_MODE:
        print(f"    Parsing xml_file stem: '{xml_filename_stem}'")

    match_base_vx = re.match(r"^([a-zA-Z]+)-v(\d+)_?(.*)$", xml_filename_stem)
    if match_base_vx:
        base_raw, version, suffix = match_base_vx.groups()
        base_env_name_formatted = base_raw.capitalize()
        full_env_id = f"{base_env_name_formatted}-v{version}"
        is_standard = suffix.lower() == base_raw.lower() or suffix == ""
        identifier = full_env_id if is_standard else suffix
        base_env_for_grouping = base_env_name_formatted
        return base_env_for_grouping, identifier, is_standard
    else:
        parts = xml_filename_stem.split("_", 1)
        base_raw = parts[0]
        base_env_name_for_grouping = base_raw.capitalize()
        is_standard_simple = xml_filename_stem.lower() == base_raw.lower()
        identifier = (
            base_env_name_for_grouping if is_standard_simple else xml_filename_stem
        )
        is_standard = is_standard_simple
        return base_env_name_for_grouping, identifier, is_standard


# --- Function to extract numeric mean and std ---
def extract_mean_std(data):
    """Safely extracts mean and std return, returning np.nan if missing/invalid."""
    mean_ret = data.get("mean_return")
    std_ret = data.get("std_return")
    if isinstance(mean_ret, (list, tuple)):
        mean_ret = mean_ret[0] if mean_ret else None
    if isinstance(std_ret, (list, tuple)):
        std_ret = std_ret[0] if std_ret else None
    try:
        mean_val = float(mean_ret) if mean_ret is not None else np.nan
    except (ValueError, TypeError):
        mean_val = np.nan
    try:
        std_val = float(std_ret) if std_ret is not None else np.nan
    except (ValueError, TypeError):
        std_val = np.nan
    return mean_val, std_val


# ---------------------

# --- Data Storage Initialization ---
standard_plot_data = {}
modified_plot_data = {}

# --- File Discovery ---
base_path = Path(runs_base_dir)
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
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        algo_raw = data.get("algo")
        xml_file_path_in_json = data.get("xml_file")
        if not all([algo_raw, xml_file_path_in_json]):
            continue
        if not isinstance(xml_file_path_in_json, str):
            continue
        base_env_name_for_grouping, identifier, is_standard = parse_xml_filename(
            xml_file_path_in_json
        )
        mean_val, std_val = extract_mean_std(data)
        if np.isnan(mean_val):
            continue
        if np.isnan(std_val):
            std_val = 0.0
        formatted_algo_name = ALGO_NAME_MAP.get(algo_raw.lower(), algo_raw.upper())
        result_dict = {"mean": mean_val, "std": std_val}
        if is_standard:
            standard_plot_data.setdefault(formatted_algo_name, {})[
                identifier
            ] = result_dict
        else:
            formatted_modification = format_modification_details(identifier)
            modified_plot_data.setdefault(base_env_name_for_grouping, {})
            modified_plot_data[base_env_name_for_grouping].setdefault(
                formatted_algo_name, {}
            )
            modified_plot_data[base_env_name_for_grouping][formatted_algo_name][
                formatted_modification
            ] = result_dict
    except json.JSONDecodeError:
        print(f"  ERROR: Could not decode JSON from {json_path}")
    except IOError as e:
        print(f"  ERROR: Could not read file {json_path}: {e}")
    except Exception as e:
        print(f"  ERROR: An unexpected error occurred processing {json_path}: {e}")

# --- Generate Plots ---
print("\n" + "=" * 50)
print("Generating Matplotlib Bar Charts...")
print("=" * 50 + "\n")

# 1. Standard Environments Plot (with Legend)
if not standard_plot_data:
    print("No data found for the standard environments plot.")
else:
    print("--- Generating Standard Environments Plot ---")
    try:
        all_algos = sorted(standard_plot_data.keys())
        all_envs_set = set().union(*(d.keys() for d in standard_plot_data.values()))
        all_envs = sorted(list(all_envs_set))
        if all_envs:
            n_envs, n_algos = len(all_envs), len(all_algos)
            bar_width = max(0.1, 0.7 / n_algos)
            index = np.arange(n_envs)
            fig, ax = plt.subplots(figsize=(max(8, n_envs * n_algos * 0.25), 6))
            algo_colors = {
                algo: DISTINCT_COLORS[i % len(DISTINCT_COLORS)]
                for i, algo in enumerate(all_algos)
            }
            for i, algo in enumerate(all_algos):
                means = [
                    standard_plot_data.get(algo, {}).get(env, {}).get("mean", np.nan)
                    for env in all_envs
                ]
                stds = [
                    standard_plot_data.get(algo, {}).get(env, {}).get("std", 0)
                    for env in all_envs
                ]
                bar_positions = index + i * bar_width - (bar_width * (n_algos - 1) / 2)
                ax.bar(
                    bar_positions,
                    means,
                    bar_width,
                    yerr=stds,
                    label=algo,
                    capsize=3,
                    color=algo_colors[algo],
                    edgecolor="black",
                    linewidth=0.5,
                )
            ax.set_ylabel("Mean Return")
            ax.set_xlabel("Standard Environment")
            ax.set_title(
                "Algorithm Performance on Standard Environments"
            )  # Title font set by rcParams
            ax.set_xticks(index)
            ax.set_xticklabels(all_envs, rotation=45, ha="right")
            ax.legend(
                title="Algorithms",
                bbox_to_anchor=(1.04, 1),
                loc="upper left",
                frameon=True,
            )
            ax.grid(True, axis="y", linestyle="-", alpha=0.6)
            ax.axhline(0, color="grey", linewidth=0.8)
            plt.tight_layout(rect=[0, 0.03, 0.88, 0.97])
            plot_filename = plot_output_dir / "standard_envs_performance.png"
            plt.savefig(plot_filename)
            print(f"Standard environments plot saved to {plot_filename}")
            plt.close(fig)
        else:
            print("  No standard environments found in data.")
    except Exception as e:
        print(f"\nError generating standard environments plot: {e}")

# 2. Modified Environments Plots (No Legend, More Space)
if not modified_plot_data:
    print("\nNo data found for any modified environments.")
else:
    print("\n--- Generating Modified Environments Plots ---")
    sorted_formatted_base_envs = sorted(modified_plot_data.keys())
    for base_env_name_for_grouping in sorted_formatted_base_envs:
        caption_base_env = format_env_name_for_caption(base_env_name_for_grouping)
        print(f"\n--- Generating Plot for Modified '{caption_base_env}' ---")
        env_data = modified_plot_data[base_env_name_for_grouping]
        if not env_data:
            print(f"  No data for '{caption_base_env}' modified environments.")
            continue
        try:
            all_algos = sorted(env_data.keys())
            all_mods_set = set().union(*(d.keys() for d in env_data.values()))
            all_mods = sorted(list(all_mods_set))
            if not all_mods:
                print(f"  No modifications found for '{caption_base_env}'.")
                continue
            n_mods, n_algos = len(all_mods), len(all_algos)
            bar_width = max(0.1, 0.7 / n_algos)
            index = np.arange(n_mods)
            fig_width = max(
                8, n_mods * n_algos * mod_plot_width_multiplier
            )  # Wider figure
            fig, ax = plt.subplots(figsize=(fig_width, 6))
            algo_colors = {
                algo: DISTINCT_COLORS[i % len(DISTINCT_COLORS)]
                for i, algo in enumerate(all_algos)
            }
            for i, algo in enumerate(all_algos):
                means = [
                    env_data.get(algo, {}).get(mod, {}).get("mean", np.nan)
                    for mod in all_mods
                ]
                stds = [
                    env_data.get(algo, {}).get(mod, {}).get("std", 0)
                    for mod in all_mods
                ]
                bar_positions = index + i * bar_width - (bar_width * (n_algos - 1) / 2)
                ax.bar(
                    bar_positions,
                    means,
                    bar_width,
                    yerr=stds,
                    label=algo,
                    capsize=3,
                    color=algo_colors[algo],
                    edgecolor="black",
                    linewidth=0.5,
                )
            ax.set_ylabel("Mean Return")
            ax.set_xlabel("Modification")
            ax.set_title(
                f"Algorithm Performance on Modified {caption_base_env} Environments"
            )  # Title font set by rcParams
            ax.set_xticks(index)
            ax.set_xticklabels(all_mods, rotation=45, ha="right")
            ax.grid(True, axis="y", linestyle="-", alpha=0.6)
            ax.axhline(0, color="grey", linewidth=0.8)
            plt.tight_layout(rect=[0, 0.03, 0.97, 0.97])  # Use more width
            safe_base_env_name = re.sub(
                r"[^\w-]", "", base_env_name_for_grouping
            ).lower()
            plot_filename = (
                plot_output_dir / f"modified_{safe_base_env_name}_performance.png"
            )
            plt.savefig(plot_filename)
            print(f"Modified '{caption_base_env}' plot saved to {plot_filename}")
            plt.close(fig)
        except Exception as e:
            print(
                f"\nError generating plot for modified '{base_env_name_for_grouping}': {e}"
            )

print("\n" + "=" * 50)
print("Script finished.")
print("=" * 50)
