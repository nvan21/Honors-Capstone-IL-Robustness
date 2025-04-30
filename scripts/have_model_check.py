import os
import yaml


def is_within_directory(target_path, parent_directory):
    """
    Checks if target_path is inside or is the same as parent_directory.

    Handles relative paths and determines containment based on the
    normalized absolute paths.

    Args:
        target_path (str): The path to check.
        parent_directory (str): The path of the directory to check against.

    Returns:
        bool: True if target_path is within or same as parent_directory,
              False otherwise.
    """
    if not target_path or not parent_directory:
        return False  # Cannot compare empty paths

    # 1. Normalize both paths to absolute paths
    #    os.path.abspath resolves '.' and '..' and makes the path absolute.
    try:
        abs_target = os.path.abspath(target_path)
        abs_parent = os.path.abspath(parent_directory)
    except Exception as e:
        # Handle potential errors during path normalization if needed
        print(
            f"Warning: Could not normalize path '{target_path}' or '{parent_directory}': {e}"
        )
        return False  # Treat un-normalizable paths as not comparable or external

    # 2. Use os.path.commonpath
    #    Find the longest common path prefix. If the target is inside
    #    the parent (or is the parent itself), the common path will
    #    be exactly the parent's absolute path.
    #    This handles cases like '/home/user/project' vs '/home/user/project_data' correctly.
    common = os.path.commonpath([abs_parent, abs_target])

    # 3. Compare the common path with the parent's absolute path
    return common == abs_parent


def find_external_dirs(directory_list, project_dir):
    """
    Filters a list of directories, returning only those not inside project_dir.

    Args:
        directory_list (list): A list of directory path strings.
        project_dir (str): The path to the project directory to compare against.

    Returns:
        list: A list of directory paths (from the original list) that are
              determined to NOT be within the project_dir.
    """
    external_directories = []
    normalized_project_dir = os.path.abspath(project_dir)  # Normalize once

    print(f"Project Directory (Normalized): {normalized_project_dir}")
    print("-" * 30)

    for dir_path in directory_list:
        print(f"Checking '{dir_path}'...")
        if is_within_directory(dir_path, normalized_project_dir):
            print(f"  -> Inside or same as project directory.")
        else:
            print(f"  -> External.")
            external_directories.append(dir_path)  # Keep the original path string

    return external_directories


# --- Example Usage ---
if __name__ == "__main__":
    # --- Configuration ---

    # Define the path to your project directory
    # This can be relative (like below) or absolute (e.g., "/path/to/your/project")
    MY_PROJECT_DIR = "."  # Example: current working directory IS the project dir

    # List of directory paths you want to check
    with open("test.yaml", "r") as f:
        DIRS_TO_CHECK = list(yaml.safe_load(f).keys())

    for dir in DIRS_TO_CHECK:
        print(f"Path: {dir}       Exists: {os.path.exists(dir)}")
