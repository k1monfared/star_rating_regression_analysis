"""Regenerate all figures from the committed data without refitting from scratch.

Usage:
    python scripts/generate_figures.py

This is a convenience wrapper. It rebuilds the deterministic dataset, refits the
models, and re-renders docs/images/*.png. For the full pipeline (data, JSON
artifacts, summary, and figures) use scripts/run_demo.py instead.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import analysis, figures  # noqa: E402
from src.data_generation import build_dataset  # noqa: E402
from src.rdd import build_rdd_frame, estimate_rdd  # noqa: E402


def main() -> None:
    cfg, businesses, attempts = build_dataset()
    prepared = analysis.prepare(attempts)
    raw = analysis.raw_descriptives(prepared)
    completion = analysis.fit_completion(prepared)
    length = analysis.fit_length(prepared)
    rdd_frame = build_rdd_frame(cfg, businesses)
    rdd_result = estimate_rdd(cfg, rdd_frame)

    paths = figures.generate_all(raw, completion, length, rdd_frame, rdd_result)
    print("Figures written:")
    for name, path in paths.items():
        print("  {}: {}".format(name, os.path.relpath(path)))


if __name__ == "__main__":
    main()
