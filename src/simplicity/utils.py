import logging
import os
import re

import tomli
from pydantic_ai.usage import Usage

from simplicity.settings import Settings
from simplicity.structure import LLMUsage, SimpOutput, SimpTaskOutput

logger = logging.getLogger(__name__)


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


def match_link(text: str) -> list[tuple[int, int, list[str]]]:
    """
    Find substrings like "[5bc707, 798902, 2c18c5, 85ec8c]" with one or more hash values.
    
    Args:
        text: The input text to search for hash patterns
        
    Returns:
        list[tuple[int, int, list[str]]]: List of tuples containing:
            - start position of the link (position of '[')
            - end position of the link (position of ']')  
            - list of hash strings found within the brackets
    """
    # Pattern to match brackets containing one or more comma-separated hash values
    # Hash pattern: 6 hexadecimal characters (a-f, A-F, 0-9)
    pattern = r'\[([a-fA-F0-9]{6}(?:,\s*[a-fA-F0-9]{6})*)\]'
    
    results = []
    for match in re.finditer(pattern, text):
        start_pos = match.start()
        end_pos = match.end()
        
        # Extract the content inside brackets and split by comma
        hash_content = match.group(1)
        hashes = [h.strip() for h in hash_content.split(',')]
        
        results.append((start_pos, end_pos, hashes))
    
    return results

def calc_usage(usage: Usage, config_name: str) -> SimpTaskOutput:
    if usage.request_tokens is None or usage.response_tokens is None:
        logger.error("Usage is None for model: %s", config_name)
    return SimpTaskOutput(
        data=[
            SimpOutput(
                d=LLMUsage(
                    kind="llm_usage",
                    input_tokens=usage.request_tokens,
                    output_tokens=usage.response_tokens,
                    config_name=config_name,
                )
            )
        ]
    )