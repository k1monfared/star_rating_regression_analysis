"""All committed figures for the analysis.

Figures are written to docs/images as PNG. Every number plotted comes from the
actual model run passed in by scripts/run_demo.py, nothing is hard-coded.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless backend, no display required
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .utils import IMAGE_DIR, ensure_dirs

# Neutral, colorblind-friendly palette. No brand colors.
_RAW = "#9aa0a6"       # muted grey for raw / unadjusted
_ADJ = "#1f6feb"       # blue for adjusted
_TRUTH = "#d1495b"     # red for injected ground truth
_ACCENT = "#2f9e44"    # green accent

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 130,
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.axisbelow": True,
    }
)


def _stars():
    return [1, 2, 3, 4, 5]


def fig_completion_ushape(raw: dict, completion: dict, path: str) -> str:
    """Completion rate by star: raw vs g-computed adjusted curve."""
    stars = _stars()
    raw_rate = [raw[s]["completion_rate"] for s in stars]
    adj_curve = completion["adjusted_categorical"]["star_curve"]
    adj_rate = [adj_curve[s] for s in stars]

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(stars, raw_rate, "o-", color=_RAW, linewidth=2, markersize=8,
            label="Raw completion rate (unadjusted)")
    ax.plot(stars, adj_rate, "s-", color=_ADJ, linewidth=2, markersize=8,
            label="Adjusted (controls for confounders)")
    ax.set_xlabel("Intended star rating")
    ax.set_ylabel("Probability of completing the review")
    ax.set_title("Review completion is U-shaped in the intended star rating")
    ax.set_xticks(stars)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_length_pattern(raw: dict, length: dict, path: str) -> str:
    """Review length by star: raw vs g-computed adjusted curve."""
    stars = _stars()
    raw_len = [raw[s]["mean_length"] for s in stars]
    adj_curve = length["ols_adjusted_categorical"]["star_curve_words"]
    adj_len = [adj_curve[s] for s in stars]

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(stars, raw_len, "o-", color=_RAW, linewidth=2, markersize=8,
            label="Raw mean length (unadjusted)")
    ax.plot(stars, adj_len, "s-", color=_ADJ, linewidth=2, markersize=8,
            label="Adjusted (controls for confounders)")
    ax.set_xlabel("Intended star rating")
    ax.set_ylabel("Review length (words)")
    ax.set_title("Reviews are shorter at the extremes, longer in the middle")
    ax.set_xticks(stars)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_adjusted_by_star(completion: dict, length: dict, path: str) -> str:
    """Stakeholder view: the two adjusted curves, completion and length by star.

    Adjusted (confounder-controlled) results only, no raw or injected-truth
    overlay. This is what a product or UX lead would read off the models: a
    completion U-shape and an inverted-U in length.
    """
    stars = _stars()
    comp_curve = completion["adjusted_categorical"]["star_curve"]
    comp_rate = [100.0 * comp_curve[s] for s in stars]
    len_curve = length["ols_adjusted_categorical"]["star_curve_words"]
    adj_len = [len_curve[s] for s in stars]

    fig, (axc, axl) = plt.subplots(1, 2, figsize=(10.4, 4.4))

    axc.plot(stars, comp_rate, "o-", color=_ADJ, linewidth=2.2, markersize=9)
    for s, y in zip(stars, comp_rate):
        axc.annotate("{:.1f}%".format(y), (s, y), textcoords="offset points",
                     xytext=(0, 9), ha="center", fontsize=9, color=_ADJ)
    axc.set_xlabel("Intended star rating")
    axc.set_ylabel("Adjusted completion rate")
    axc.set_title("Completion is U-shaped: middle ratings leak")
    axc.set_xticks(stars)
    axc.set_ylim(min(comp_rate) - 8, max(comp_rate) + 8)
    axc.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: "{:.0f}%".format(v)))

    axl.plot(stars, adj_len, "s-", color=_ACCENT, linewidth=2.2, markersize=9)
    for s, y in zip(stars, adj_len):
        axl.annotate("{:.0f}".format(y), (s, y), textcoords="offset points",
                     xytext=(0, 9), ha="center", fontsize=9, color=_ACCENT)
    axl.set_xlabel("Intended star rating")
    axl.set_ylabel("Adjusted review length (words)")
    axl.set_title("Length is inverted-U: extremes say the least")
    axl.set_xticks(stars)
    axl.set_ylim(min(adj_len) - 8, max(adj_len) + 8)

    fig.suptitle("Adjusted results by star rating (confounder-controlled)",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_confounding(raw: dict, path: str) -> str:
    """Why the raw pattern is biased: confounders vary systematically by star."""
    stars = _stars()
    share_mobile = [raw[s]["share_mobile"] for s in stars]
    exp_z = [raw[s]["mean_experience_z"] for s in stars]

    fig, ax1 = plt.subplots(figsize=(7.2, 4.6))
    ax1.bar([s - 0.0 for s in stars], share_mobile, width=0.55, color=_RAW,
            alpha=0.8, label="Share on mobile")
    ax1.set_xlabel("Intended star rating")
    ax1.set_ylabel("Share of attempts on mobile", color="#5f6368")
    ax1.set_xticks(stars)
    ax1.set_ylim(0, max(share_mobile) * 1.25)

    ax2 = ax1.twinx()
    ax2.plot(stars, exp_z, "o-", color=_ACCENT, linewidth=2, markersize=8,
             label="Mean user experience (z)")
    ax2.set_ylabel("Mean user experience (standardized)", color=_ACCENT)
    ax2.grid(False)

    ax1.set_title("Confounders vary by star: extremes skew mobile and less experienced")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="upper center")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_rdd(rdd_frame: pd.DataFrame, rdd_result: dict, path: str) -> str:
    """Binned RDD plot around the half-star rounding threshold."""
    bw = rdd_result["bandwidth"]
    band = rdd_frame[np.abs(rdd_frame["running"]) <= bw].copy()

    # Bin the running variable for a clean scatter.
    nbins = 20
    band["bin"] = pd.cut(band["running"], bins=nbins)
    grouped = band.groupby("bin", observed=True).agg(
        x=("running", "mean"), y=("log_weekly_attempts", "mean")
    ).dropna()

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    left = grouped[grouped["x"] < 0]
    right = grouped[grouped["x"] >= 0]
    ax.scatter(left["x"], left["y"], color=_RAW, s=35, label="Below threshold (rounds down)")
    ax.scatter(right["x"], right["y"], color=_ADJ, s=35, label="At/above threshold (rounds up)")

    # Fitted local-linear lines on each side.
    for side, color in ((left, _RAW), (right, _ADJ)):
        if len(side) >= 2:
            coeffs = np.polyfit(side["x"], side["y"], 1)
            xs = np.linspace(side["x"].min(), side["x"].max(), 50)
            ax.plot(xs, np.polyval(coeffs, xs), color=color, linewidth=2)

    ax.axvline(0.0, color=_TRUTH, linestyle="--", linewidth=1.5, alpha=0.8)
    ax.set_xlabel("Distance of true average to nearest half-star threshold")
    ax.set_ylabel("Log weekly review attempts")
    ax.set_title(
        "RDD: injected jump {:.3f}, estimated {:.3f}".format(
            rdd_result["injected_jump"], rdd_result["estimated_jump"]
        )
    )
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def generate_all(raw: dict, completion: dict, length: dict, rdd_frame, rdd_result) -> dict:
    """Produce every figure and return the mapping of names to paths."""
    ensure_dirs()
    paths = {
        "adjusted_by_star": fig_adjusted_by_star(
            completion, length, f"{IMAGE_DIR}/adjusted_by_star.png"
        ),
        "completion_ushape": fig_completion_ushape(
            raw, completion, f"{IMAGE_DIR}/completion_ushape.png"
        ),
        "length_pattern": fig_length_pattern(
            raw, length, f"{IMAGE_DIR}/length_pattern.png"
        ),
        "confounding": fig_confounding(raw, f"{IMAGE_DIR}/confounding.png"),
        "rdd": fig_rdd(rdd_frame, rdd_result, f"{IMAGE_DIR}/rdd_halfstar.png"),
    }
    return paths
