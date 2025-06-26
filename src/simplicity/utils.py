import os

import tomli

from simplicity.settings import Settings


def get_project_root():
    """
    Recursively search for a .project-root file starting from the current file's directory
    and moving up the directory tree until found or reaching the filesystem root.

    Returns:
        str: Path to the directory containing the .project-root file, or the original
             directory if .project-root is not found.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))

    while True:
        # Check if .project-root exists in current directory
        project_root_file = os.path.join(current_dir, ".project-root")
        if os.path.exists(project_root_file):
            return current_dir

        # Move up one directory
        parent_dir = os.path.dirname(current_dir)

        # If we've reached the filesystem root, stop searching
        if parent_dir == current_dir:
            # Return the original directory if .project-root not found
            return os.path.dirname(os.path.abspath(__file__))

        current_dir = parent_dir


def get_settings_from_project_root():
    project_root = get_project_root()
    settings_file = os.path.join(project_root, "config.toml")
    if not os.path.exists(settings_file):
        raise FileNotFoundError(f"Settings file not found at {settings_file}")
    with open(settings_file, "rb") as f:
        return Settings.model_validate(tomli.load(f))
