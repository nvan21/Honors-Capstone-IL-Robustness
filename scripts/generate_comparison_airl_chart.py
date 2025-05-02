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
nan_replacement_val = np.nan  # Use NaN for missing numeric data
# Directory to save the generated plots
plot_output_dir = Path("assets") / "plots"
# Specific algorithms to compare
ALGO_TARGET = (
    "Modified SAC"  # The algorithm whose performance is evaluated relative to baseline
)
ALGO_BASELINE = "AIRL"  # The algorithm used as the baseline
# Percentage formatting
percentage_precision = 0  # Decimal places for percentage labels

# Plotting Style and Aesthetics
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.dpi"] = 100
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"]
plt.rcParams["font.size"] = 10
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["axes.titlesize"] = 14  # Title size
plt.rcParams["axes.titleweight"] = "bold"  # Title bold
plt.rcParams["xtick.labelsize"] = 9
plt.rcParams["ytick.labelsize"] = 9
plt.rcParams["legend.fontsize"] = 9
plt.rcParams["legend.title_fontsize"] = 10


# Ensure output directory exists
plot_output_dir.mkdir(parents=True, exist_ok=True)

# Set to True to print detailed parsing and data storage info
DEBUG_MODE = False

# --- Formatting Maps ---
ALGO_NAME_MAP = {
    "sac": "Expert SAC",
    "modified_sac": "Modified SAC",
    "dagger": "DAgger",
    "airl": "AIRL",
    "gail": "GAIL",
    "bc": "BC",
}
TARGET_ALGO_NAME_FORMATTED = ALGO_NAME_MAP.get(
    ALGO_TARGET.lower().replace(" ", "_"), ALGO_TARGET
)
BASELINE_ALGO_NAME_FORMATTED = ALGO_NAME_MAP.get(
    ALGO_BASELINE.lower().replace(" ", "_"), ALGO_BASELINE
)


# --- Utility Functions ---
def format_env_name_for_caption(base_env_name):
    """Formats the base environment name."""
    if not base_env_name:
        return ""
    if base_env_name.lower() == "invpend":
        return "Inverted Pendulum"
    return base_env_name.capitalize()


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


def extract_mean_std(data):
    """Safely extracts mean return, returning np.nan if missing/invalid."""
    mean_ret = data.get("mean_return")
    if isinstance(mean_ret, (list, tuple)):
        mean_ret = mean_ret[0] if mean_ret else None
    try:
        mean_val = float(mean_ret) if mean_ret is not None else np.nan
    except (ValueError, TypeError):
        mean_val = np.nan
    return mean_val


# ---------------------

# --- Data Storage Initialization ---
standard_means_data = {}

# --- File Discovery ---
base_path = Path(runs_base_dir)
json_files = list(base_path.rglob("*/results.json"))
if not json_files:
    print(f"ERROR: No 'results.json' files found.")
    exit()
print(f"Found {len(json_files)} results.json files.")

# --- Processing Loop ---
for json_path in json_files:
    try:
        algo_dir_name = json_path.parent.name.lower()
        is_relevant_algo = (
            ALGO_TARGET.lower().replace(" ", "_") in algo_dir_name
            or ALGO_BASELINE.lower().replace(" ", "_") in algo_dir_name
        )
        if not is_relevant_algo:
            continue
    except Exception:
        pass

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
        if is_standard:
            mean_val = extract_mean_std(data)
            if np.isnan(mean_val):
                print(f"  WARNING: Skipping - Invalid mean in {json_path}")
                continue
            formatted_algo_name = ALGO_NAME_MAP.get(algo_raw.lower(), algo_raw.upper())
            if formatted_algo_name in [
                TARGET_ALGO_NAME_FORMATTED,
                BASELINE_ALGO_NAME_FORMATTED,
            ]:
                standard_means_data.setdefault(formatted_algo_name, {})[
                    identifier
                ] = mean_val
                if DEBUG_MODE:
                    print(
                        f"  Stored Standard Mean: Algo='{formatted_algo_name}', Env='{identifier}', Mean={mean_val}"
                    )
    except json.JSONDecodeError:
        print(f"  ERROR: Could not decode JSON from {json_path}")
    except IOError as e:
        print(f"  ERROR: Could not read file {json_path}: {e}")
    except Exception as e:
        print(f"  ERROR: An unexpected error occurred processing {json_path}: {e}")


# --- Generate Comparison Plot ---

print("\n" + "=" * 50)
print(
    f"Generating Comparison Bar Chart: {TARGET_ALGO_NAME_FORMATTED} / {BASELINE_ALGO_NAME_FORMATTED} on Standard Environments"
)
print("=" * 50 + "\n")

target_data = standard_means_data.get(TARGET_ALGO_NAME_FORMATTED, {})
baseline_data = standard_means_data.get(BASELINE_ALGO_NAME_FORMATTED, {})

if not target_data or not baseline_data:
    print(
        f"ERROR: Could not find standard env data for both '{TARGET_ALGO_NAME_FORMATTED}' and '{BASELINE_ALGO_NAME_FORMATTED}'."
    )
else:
    common_envs = sorted(list(set(target_data.keys()) & set(baseline_data.keys())))
    if not common_envs:
        print(f"ERROR: No common standard environments found for both algorithms.")
    else:
        ratios = []
        envs_plotted = []
        print("Calculating ratios for common standard environments:")
        for env in common_envs:
            target_mean = target_data.get(env)
            baseline_mean = baseline_data.get(env)

            if target_mean is None or baseline_mean is None:
                print(f"  Skipping '{env}': Missing data.")
                continue
            if abs(baseline_mean) < 1e-9:  # Avoid division by zero
                print(
                    f"  Skipping '{env}': Baseline score is near zero ({baseline_mean:.2f}). Ratio undefined."
                )
                continue

            # --- Ratio Calculation Logic ---
            if target_mean >= 0 and baseline_mean > 0:
                # Standard case: both positive
                ratio = target_mean / baseline_mean
                calc_type = "pos/pos"
            elif target_mean < 0 and baseline_mean < 0:
                # Both negative: Use min(abs)/max(abs) to show recovery fraction
                # Numerator is the one CLOSER to zero (min abs value)
                # Denominator is the one FURTHER from zero (max abs value)
                abs_target = abs(target_mean)
                abs_baseline = abs(baseline_mean)
                ratio = min(abs_target, abs_baseline) / max(abs_target, abs_baseline)
                calc_type = "neg/neg (min|val|/max|val|)"
                # If target is better (closer to zero, smaller abs), ratio < 1
                # If baseline is better (closer to zero, smaller abs), ratio < 1
                # This makes the interpretation consistent: how much of the better negative score does the worse score achieve?
                # A ratio closer to 1 means the performance difference is smaller.
            elif target_mean >= 0 and baseline_mean < 0:
                # Target positive, Baseline negative: HUGE improvement, assign large positive ratio?
                # Or maybe cap it? Let's represent as a large value (e.g., 2.0 or 200%) for plotting.
                ratio = 2.0  # Represents >100% improvement crossing zero
                calc_type = "pos/neg"
                print(
                    f"  INFO '{env}': Target score positive ({target_mean:.2f}), baseline negative ({baseline_mean:.2f}). Plotting ratio as 200%."
                )
            elif target_mean < 0 and baseline_mean >= 0:
                # Target negative, Baseline positive/zero: HUGE drop, assign small/zero ratio?
                ratio = 0.0  # Represents <0% performance (dropped below zero baseline)
                calc_type = "neg/pos"
                print(
                    f"  INFO '{env}': Target score negative ({target_mean:.2f}), baseline non-negative ({baseline_mean:.2f}). Plotting ratio as 0%."
                )
            else:
                # Should not happen given previous checks, but as fallback:
                ratio = np.nan
                calc_type = "other/invalid"

            if np.isnan(ratio):
                print(f"  Skipping '{env}': Ratio calculation resulted in NaN.")
                continue

            ratios.append(ratio)
            envs_plotted.append(env)
            print(
                f"  - {env}: Target={target_mean:.2f}, Baseline={baseline_mean:.2f}. Ratio ({calc_type}) = {ratio:.3f}"
            )

        if not envs_plotted:
            print("ERROR: No environments remained after calculating ratios.")
        else:
            try:
                n_envs = len(envs_plotted)
                index = np.arange(n_envs)
                fig, ax = plt.subplots(figsize=(max(6, n_envs * 1.2), 5))

                # Plot ratios as percentages
                percentages_to_plot = [r * 100 for r in ratios]
                bars = ax.bar(
                    index,
                    percentages_to_plot,
                    color="#1f77b4",
                    edgecolor="black",
                    linewidth=0.5,
                )

                ax.set_ylabel(
                    f"{TARGET_ALGO_NAME_FORMATTED} Performance (% of {BASELINE_ALGO_NAME_FORMATTED})"
                )
                ax.set_xlabel("Standard Environment")
                ax.set_title(
                    f"Relative Performance: {TARGET_ALGO_NAME_FORMATTED} vs {BASELINE_ALGO_NAME_FORMATTED}"
                )
                ax.set_xticks(index)
                ax.set_xticklabels(envs_plotted, rotation=45, ha="right")
                ax.grid(True, axis="y", linestyle="-", alpha=0.6)
                # Add 100% line for reference (where target == baseline for positive scores)
                ax.axhline(
                    100,
                    color="red",
                    linestyle="--",
                    linewidth=1,
                    label="Perfect Performance Recovery (100%)",
                )  # UPDATED LABEL

                # Format y-axis as percentage
                ax.yaxis.set_major_formatter(mticker.PercentFormatter())

                # Add percentage annotations on bars
                for bar in bars:
                    height = bar.get_height()
                    ax.annotate(
                        f"{height:.{percentage_precision}f}%",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3 if height >= 0 else -12),
                        textcoords="offset points",
                        ha="center",
                        va="bottom" if height >= 0 else "top",
                        fontsize=8,
                    )

                plt.tight_layout()

                plot_filename = (
                    plot_output_dir
                    / f"std_perf_ratio_{ALGO_TARGET.lower().replace(' ','_')}_vs_{ALGO_BASELINE.lower().replace(' ','_')}.png"
                )
                plt.savefig(plot_filename)
                print(f"\nComparison plot saved to {plot_filename}")
                plt.close(fig)

            except Exception as e:
                print(f"\nError generating comparison plot: {e}")

print("\n" + "=" * 50)
print("Script finished.")
print("=" * 50)
