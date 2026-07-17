"""Shared helpers: repository paths and config loading.

Keeping all path logic in one place means every script and module resolves files
relative to the repository root, regardless of the current working directory.
"""
from __future__ import annotations

import json
import os

# Repository root is the parent of the directory that holds this file (src/).
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CONFIG_DIR = os.path.join(REPO_ROOT, "configs")
DATA_DIR = os.path.join(REPO_ROOT, "data")
OUTPUT_DIR = os.path.join(REPO_ROOT, "outputs")
IMAGE_DIR = os.path.join(REPO_ROOT, "docs", "images")

DEFAULT_CONFIG_PATH = os.path.join(CONFIG_DIR, "data_config.json")


def load_config(path: str = DEFAULT_CONFIG_PATH) -> dict:
    """Load the ground-truth generative configuration."""
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def ensure_dirs() -> None:
    """Create output directories if they do not already exist."""
    for directory in (DATA_DIR, OUTPUT_DIR, IMAGE_DIR):
        os.makedirs(directory, exist_ok=True)


def write_json(obj: dict, filename: str) -> str:
    """Write a JSON artifact to the outputs directory and return its path."""
    ensure_dirs()
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2)
    return path
