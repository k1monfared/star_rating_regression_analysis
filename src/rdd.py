"""Optional regression-discontinuity extension: half-star rounding.

Business pages display the average rating rounded to the nearest half star. Two
businesses with almost identical true averages (say 3.24 and 3.26) are shown 3.0
and 3.5 stars respectively. If the displayed rating drives downstream behavior,
we should see a jump in outcomes exactly at each rounding threshold, even though
the underlying quality is essentially continuous.

This is the classic design behind published work on how coarse displayed ratings
causally affect business outcomes. Here we inject a small, known jump into a
synthetic downstream outcome (weekly review-attempt volume, on a log scale) at
the rounding thresholds and recover it with a local-linear RDD.

This extension is deliberately lightweight and secondary to the completion and
length analysis. It exists to show the design and that we can recover the
injected discontinuity, not to carry the main business conclusions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def build_rdd_frame(cfg: dict, businesses: pd.DataFrame) -> pd.DataFrame:
    """Create a business-level downstream outcome with an injected rounding jump.

    The running variable is the true average rating recentered at its distance to
    the nearest half-star rounding threshold. Crossing a threshold from below adds
    a known jump to the log downstream outcome.
    """
    rcfg = cfg["rdd"]
    rng = np.random.default_rng(cfg["seed"] + 1)

    true_avg = businesses["business_true_avg_rating"].to_numpy()

    # Rounding thresholds sit at the midpoints between displayed half-star values:
    # 1.25, 1.75, 2.25, ... A true average is rounded UP once it passes a midpoint.
    nearest_threshold = np.round((true_avg - 0.25) * 2.0) / 2.0 + 0.25
    running = true_avg - nearest_threshold  # distance to threshold, in (-0.25, 0.25]
    above = (running >= 0).astype(int)      # displayed rating rounded up at threshold

    log_outcome = (
        3.0
        + rcfg["slope"] * running
        + rcfg["true_jump"] * above
        + rng.normal(0.0, rcfg["noise_sd"], size=len(true_avg))
    )

    return pd.DataFrame(
        {
            "business_id": businesses["business_id"].to_numpy(),
            "true_avg": true_avg,
            "running": running,
            "above": above,
            "log_weekly_attempts": log_outcome,
        }
    )


def estimate_rdd(cfg: dict, frame: pd.DataFrame) -> dict:
    """Local-linear RDD estimate of the jump within the configured bandwidth."""
    bw = cfg["rdd"]["bandwidth"]
    band = frame[np.abs(frame["running"]) <= bw].copy()

    # Local linear regression with separate slopes on each side of the threshold.
    model = smf.ols("log_weekly_attempts ~ above + running + above:running", data=band).fit()

    jump = float(model.params["above"])
    ci = model.conf_int().loc["above"]

    return {
        "formula": "log_weekly_attempts ~ above + running + above:running",
        "bandwidth": bw,
        "n_in_band": int(len(band)),
        "injected_jump": float(cfg["rdd"]["true_jump"]),
        "estimated_jump": jump,
        "estimated_jump_ci": [float(ci[0]), float(ci[1])],
        "std_err": float(model.bse["above"]),
        "p_value": float(model.pvalues["above"]),
        "recovered_within_ci": bool(ci[0] <= cfg["rdd"]["true_jump"] <= ci[1]),
    }
