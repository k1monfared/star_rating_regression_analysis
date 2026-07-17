"""Synthetic review-attempt data generator.

This module builds a synthetic dataset of review ATTEMPTS on a fictional local
business review and discovery platform. A review attempt is a session in which a
user has opened the review composer for a business with an intended star rating.
Two outcomes are recorded:

    completed     : did the user finish and submit the review (0 / 1)
    review_length : word count of the submitted review (only when completed)

The data is generated from a fully documented ground-truth model (see
configs/data_config.json). The relationships we deliberately inject are:

    1. Completion probability is U-shaped in the intended star rating: users who
       intend to give 1 or 5 stars complete more often than users who intend to
       give 2, 3, or 4 stars. Encoded as a POSITIVE coefficient on (star - 3)^2.

    2. Review length is inverted-U in the intended star rating: reviews at the
       extremes (1 and 5) are SHORTER, reviews in the middle run longer. Encoded
       as a NEGATIVE coefficient on (star - 3)^2.

    3. Confounding is real, not cosmetic. The intended star rating is correlated
       with device (mobile users skew to extreme ratings), user experience
       (experienced users skew to moderate ratings), and the business average
       rating. Those same variables independently drive completion and length.
       As a result the RAW (unadjusted) star pattern is biased, and only a
       regression that controls for the confounders recovers the injected truth.

Everything is synthetic and clearly labeled. No real users, businesses, or
platform data are involved.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import DATA_DIR, ensure_dirs, load_config


def _softmax_rows(logits: np.ndarray) -> np.ndarray:
    """Row-wise softmax that is numerically stable."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _sample_categorical_rows(probs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Draw one categorical outcome per row given a matrix of row probabilities.

    Uses the inverse-CDF (Gumbel-free) trick: compare a single uniform per row to
    the cumulative probabilities. Returns 0-based class indices.
    """
    cumulative = np.cumsum(probs, axis=1)
    draws = rng.random(size=(probs.shape[0], 1))
    return (draws > cumulative).sum(axis=1)


def generate_businesses(cfg: dict, rng: np.random.Generator) -> pd.DataFrame:
    """Create the business table with a true average rating and displayed stars."""
    n = cfg["n_businesses"]
    b = cfg["business"]
    cats = cfg["categories"]

    category = rng.choice(len(cats), size=n, p=cfg["category_probs"])
    true_avg = rng.normal(b["avg_rating_mean"], b["avg_rating_sd"], size=n)
    true_avg = np.clip(true_avg, b["avg_rating_min"], b["avg_rating_max"])
    review_count = np.rint(
        rng.lognormal(b["review_count_lognormal_mu"], b["review_count_lognormal_sigma"], size=n)
    ).astype(int) + 1

    # Displayed stars: the true average rounded to the nearest half star. This is
    # the running variable used later by the optional RDD extension.
    displayed_stars = np.round(true_avg * 2.0) / 2.0

    return pd.DataFrame(
        {
            "business_id": np.arange(n),
            "business_category": [cats[i] for i in category],
            "business_true_avg_rating": np.round(true_avg, 3),
            "business_displayed_stars": displayed_stars,
            "business_review_count": review_count,
        }
    )


def generate_attempts(cfg: dict, businesses: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Create the review-attempt table with completion and length outcomes."""
    n = cfg["n_attempts"]
    cats = cfg["categories"]
    cat_index = {c: i for i, c in enumerate(cats)}

    # --- Assign each attempt to a business -------------------------------------
    biz_idx = rng.integers(0, cfg["n_businesses"], size=n)
    biz = businesses.iloc[biz_idx].reset_index(drop=True)

    # --- User features (confounder: experience) --------------------------------
    u = cfg["user"]
    review_count = np.rint(
        rng.lognormal(u["review_count_lognormal_mu"], u["review_count_lognormal_sigma"], size=n)
    ).astype(int)
    tenure_days = np.rint(
        rng.lognormal(u["tenure_days_lognormal_mu"], u["tenure_days_lognormal_sigma"], size=n)
    ).astype(int) + 1

    # Experience as a standardized log-review-count. This is the modeling feature
    # that the analysis will control for.
    log_reviews = np.log1p(review_count)
    experience_z = (log_reviews - log_reviews.mean()) / log_reviews.std()

    elite_logit = u["elite_base_logit"] + u["elite_experience_coef"] * experience_z
    user_is_elite = (rng.random(n) < _sigmoid(elite_logit)).astype(int)

    # --- Context features (confounder: device, prompt) -------------------------
    c = cfg["context"]
    device_mobile = (rng.random(n) < c["mobile_prob"]).astype(int)

    prompt_logit = (
        c["prompt_base_logit"]
        + c["prompt_experience_coef"] * experience_z
        + c["prompt_mobile_coef"] * device_mobile
    )
    session_prompted = (rng.random(n) < _sigmoid(prompt_logit)).astype(int)

    # --- Intended star rating (the treatment of interest) ----------------------
    # Confounders shift the star a user intends to give, which is precisely what
    # entangles star with completion and length.
    sd = cfg["star_distribution"]
    base = np.array(sd["base_log_weights"])          # shape (5,)
    mob = np.array(sd["mobile_shift"])
    exp = np.array(sd["experience_shift"])
    avg = np.array(sd["business_avg_shift"])

    avg_centered = (biz["business_true_avg_rating"].to_numpy() - 3.0)

    logits = (
        base[None, :]
        + device_mobile[:, None] * mob[None, :]
        + experience_z[:, None] * exp[None, :]
        + avg_centered[:, None] * avg[None, :]
    )
    star_probs = _softmax_rows(logits)
    intended_star = _sample_categorical_rows(star_probs, rng) + 1  # map 0..4 -> 1..5

    # --- Derived design terms --------------------------------------------------
    star_centered = intended_star - 3.0
    star_u = star_centered ** 2  # U-shape / inverted-U driver
    cat_arr = biz["business_category"].to_numpy()
    log_biz_reviews = np.log1p(biz["business_review_count"].to_numpy())

    # --- Completion outcome (logistic, TRUE U-shape) ---------------------------
    cm = cfg["completion_model"]
    comp_cat_effect = np.array([cm["category_effects"][cat] for cat in cat_arr])
    eta_c = (
        cm["intercept"]
        + cm["star_u_coef"] * star_u
        + cm["experience_coef"] * experience_z
        + cm["mobile_coef"] * device_mobile
        + cm["prompt_coef"] * session_prompted
        + cm["elite_coef"] * user_is_elite
        + cm["business_logreviews_coef"] * (log_biz_reviews - log_biz_reviews.mean())
        + comp_cat_effect
    )
    p_complete = _sigmoid(eta_c)
    completed = (rng.random(n) < p_complete).astype(int)

    # --- Length outcome (negative binomial, TRUE inverted-U) -------------------
    lm = cfg["length_model"]
    len_cat_effect = np.array([lm["category_effects"][cat] for cat in cat_arr])
    log_mu = (
        lm["log_intercept"]
        + lm["star_u_coef"] * star_u
        + lm["experience_coef"] * experience_z
        + lm["mobile_coef"] * device_mobile
        + lm["prompt_coef"] * session_prompted
        + lm["elite_coef"] * user_is_elite
        + len_cat_effect
    )
    mu = np.exp(log_mu)
    length_full = _sample_negative_binomial(mu, lm["nb_dispersion"], rng)
    length_full = np.maximum(length_full, lm["min_words"])

    # Length is only observed when the review is completed.
    review_length = np.where(completed == 1, length_full, np.nan)

    df = pd.DataFrame(
        {
            "attempt_id": np.arange(n),
            "business_id": biz["business_id"].to_numpy(),
            "business_category": cat_arr,
            "business_true_avg_rating": biz["business_true_avg_rating"].to_numpy(),
            "business_displayed_stars": biz["business_displayed_stars"].to_numpy(),
            "business_review_count": biz["business_review_count"].to_numpy(),
            "intended_star": intended_star.astype(int),
            "user_review_count": review_count,
            "user_tenure_days": tenure_days,
            "user_experience_z": np.round(experience_z, 4),
            "user_is_elite": user_is_elite,
            "device_mobile": device_mobile,
            "session_prompted": session_prompted,
            "completed": completed,
            "review_length": review_length,
        }
    )
    return df


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _sample_negative_binomial(mu: np.ndarray, dispersion: float, rng: np.random.Generator) -> np.ndarray:
    """Sample negative binomial counts parameterized by mean mu and dispersion r.

    Uses the Gamma-Poisson mixture: draw a Gamma-distributed rate with shape r and
    scale mu / r, then draw a Poisson with that rate. Variance = mu + mu^2 / r.
    """
    shape = dispersion
    scale = mu / dispersion
    gamma_rate = rng.gamma(shape=shape, scale=scale)
    return rng.poisson(gamma_rate)


def build_dataset(config_path: str | None = None):
    """Generate businesses and attempts and return both frames."""
    cfg = load_config(config_path) if config_path else load_config()
    rng = np.random.default_rng(cfg["seed"])
    businesses = generate_businesses(cfg, rng)
    attempts = generate_attempts(cfg, businesses, rng)
    return cfg, businesses, attempts


def save_dataset(businesses: pd.DataFrame, attempts: pd.DataFrame) -> dict:
    """Write both tables to the committed data directory."""
    ensure_dirs()
    biz_path = f"{DATA_DIR}/businesses.csv"
    att_path = f"{DATA_DIR}/review_attempts.csv"
    businesses.to_csv(biz_path, index=False)
    attempts.to_csv(att_path, index=False)
    return {"businesses": biz_path, "attempts": att_path}
