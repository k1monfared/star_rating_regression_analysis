"""Generate and commit the synthetic review-attempt dataset.

Usage:
    python scripts/generate_data.py

Writes:
    data/businesses.csv
    data/review_attempts.csv

Data is fully synthetic and reproducible from the fixed seed in
configs/data_config.json. run_demo.py calls the same code path, so running this
script is only needed if you want the data without the analysis.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_generation import build_dataset, save_dataset  # noqa: E402


def main() -> None:
    cfg, businesses, attempts = build_dataset()
    paths = save_dataset(businesses, attempts)

    n = len(attempts)
    completed = int(attempts["completed"].sum())
    print("Synthetic data generated (seed = {}).".format(cfg["seed"]))
    print("  businesses:      {:>7,}  ->  {}".format(len(businesses), paths["businesses"]))
    print("  review attempts: {:>7,}  ->  {}".format(n, paths["attempts"]))
    print("  overall completion rate: {:.3f}".format(completed / n))
    print("  completed reviews:       {:,}".format(completed))


if __name__ == "__main__":
    main()
