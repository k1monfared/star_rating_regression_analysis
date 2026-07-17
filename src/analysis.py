"""Regression analysis of the review-creation funnel.

Two outcomes are modeled:

    completion : logistic regression of whether a review attempt is submitted.
                 Star rating enters both as a categorical factor and, in a second
                 specification, as a quadratic (star - 3)^2 term to capture the
                 U-shape with a single interpretable coefficient.

    length     : regression of review word count for completed reviews. Reported
                 as OLS on log(length) for interpretability and, as the model that
                 matches the data-generating process, a negative binomial GLM.

For every outcome we fit a NAIVE model (star only) and an ADJUSTED model (star
plus confounder controls) so we can show that controlling changes the estimates.
Predicted curves by star are produced by g-computation: fix the star for every
row, predict, and average, which marginalizes over the observed covariate
distribution.

This is an observational analysis. The recovered numbers are ASSOCIATIONS. See
the caveats section of the summary and README for what a causal claim would need.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Controls shared by the adjusted completion and length specifications.
_CONTROLS = (
    "user_experience_z + device_mobile + session_prompted + user_is_elite "
    "+ biz_logreviews_c + C(business_category)"
)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived modeling columns used across specifications."""
    out = df.copy()
    out["star_c"] = out["intended_star"] - 3.0
    out["star_u"] = out["star_c"] ** 2
    log_biz = np.log1p(out["business_review_count"])
    out["biz_logreviews_c"] = log_biz - log_biz.mean()
    completed_mask = out["completed"] == 1
    out["log_length"] = np.nan
    out.loc[completed_mask, "log_length"] = np.log(out.loc[completed_mask, "review_length"])
    return out


def _coef_table(result) -> dict:
    """Extract coefficients, standard errors, CIs, and p-values as plain dicts."""
    conf = result.conf_int()
    conf.columns = ["ci_low", "ci_high"]
    table = {}
    for name in result.params.index:
        table[name] = {
            "coef": float(result.params[name]),
            "std_err": float(result.bse[name]),
            "ci_low": float(conf.loc[name, "ci_low"]),
            "ci_high": float(conf.loc[name, "ci_high"]),
            "p_value": float(result.pvalues[name]),
        }
    return table


def _star_curve_logit(result, df: pd.DataFrame) -> dict:
    """G-computation completion probability by star (average adjusted prediction)."""
    curve = {}
    work = df.copy()
    for star in range(1, 6):
        work["intended_star"] = star
        work["star_c"] = star - 3.0
        work["star_u"] = (star - 3.0) ** 2
        preds = result.predict(work)
        curve[star] = float(np.mean(preds))
    return curve


def _star_curve_length(result, df: pd.DataFrame, log_scale: bool) -> dict:
    """G-computation predicted review length by star (completed reviews only)."""
    curve = {}
    work = df.copy()
    for star in range(1, 6):
        work["intended_star"] = star
        work["star_c"] = star - 3.0
        work["star_u"] = (star - 3.0) ** 2
        preds = result.predict(work)
        preds = np.exp(preds) if log_scale else np.asarray(preds)
        curve[star] = float(np.mean(preds))
    return curve


# --------------------------------------------------------------------------- #
# Completion models
# --------------------------------------------------------------------------- #

def fit_completion(df: pd.DataFrame) -> dict:
    """Fit naive, adjusted-categorical, and adjusted-quadratic completion models."""
    results = {}

    # Naive: star only, categorical.
    naive = smf.logit("completed ~ C(intended_star)", data=df).fit(disp=False)

    # Adjusted: star categorical + controls.
    adj_cat = smf.logit(f"completed ~ C(intended_star) + {_CONTROLS}", data=df).fit(disp=False)

    # Adjusted quadratic: single U-shape coefficient + controls.
    adj_quad = smf.logit(f"completed ~ star_u + {_CONTROLS}", data=df).fit(disp=False)

    # Naive quadratic for a clean naive-vs-adjusted U-coefficient comparison.
    naive_quad = smf.logit("completed ~ star_u", data=df).fit(disp=False)

    # Average marginal effects of the controls in the adjusted categorical model.
    margeff = adj_cat.get_margeff(at="overall")
    me_frame = margeff.summary_frame()
    ame = {}
    for name, row in me_frame.iterrows():
        ame[str(name)] = {
            "dydx": float(row["dy/dx"]),
            "std_err": float(row["Std. Err."]),
            "p_value": float(row["Pr(>|z|)"]),
        }

    results["naive"] = {
        "formula": "completed ~ C(intended_star)",
        "coefficients": _coef_table(naive),
        "pseudo_r2": float(naive.prsquared),
        "aic": float(naive.aic),
        "log_likelihood": float(naive.llf),
        "n": int(naive.nobs),
        "star_curve": _star_curve_logit(naive, df),
    }
    results["adjusted_categorical"] = {
        "formula": f"completed ~ C(intended_star) + {_CONTROLS}",
        "coefficients": _coef_table(adj_cat),
        "pseudo_r2": float(adj_cat.prsquared),
        "aic": float(adj_cat.aic),
        "log_likelihood": float(adj_cat.llf),
        "n": int(adj_cat.nobs),
        "star_curve": _star_curve_logit(adj_cat, df),
        "average_marginal_effects": ame,
    }
    results["adjusted_quadratic"] = {
        "formula": f"completed ~ star_u + {_CONTROLS}",
        "coefficients": _coef_table(adj_quad),
        "pseudo_r2": float(adj_quad.prsquared),
        "aic": float(adj_quad.aic),
        "log_likelihood": float(adj_quad.llf),
        "n": int(adj_quad.nobs),
        "star_u_coef": float(adj_quad.params["star_u"]),
        "star_u_ci": [float(adj_quad.conf_int().loc["star_u", 0]),
                      float(adj_quad.conf_int().loc["star_u", 1])],
    }
    results["naive_quadratic"] = {
        "formula": "completed ~ star_u",
        "star_u_coef": float(naive_quad.params["star_u"]),
        "star_u_ci": [float(naive_quad.conf_int().loc["star_u", 0]),
                      float(naive_quad.conf_int().loc["star_u", 1])],
    }
    return results


# --------------------------------------------------------------------------- #
# Length models (completed reviews only)
# --------------------------------------------------------------------------- #

def fit_length(df: pd.DataFrame) -> dict:
    """Fit naive and adjusted length models: OLS on log length and NB GLM."""
    completed = df[df["completed"] == 1].copy()
    results = {}

    # OLS on log length.
    ols_naive = smf.ols("log_length ~ C(intended_star)", data=completed).fit()
    ols_adj = smf.ols(f"log_length ~ C(intended_star) + {_CONTROLS}", data=completed).fit()
    ols_quad = smf.ols(f"log_length ~ star_u + {_CONTROLS}", data=completed).fit()
    ols_naive_quad = smf.ols("log_length ~ star_u", data=completed).fit()

    # Negative binomial GLM (matches the data-generating process, log link).
    nb_adj = smf.glm(
        f"review_length ~ star_u + {_CONTROLS}",
        data=completed,
        family=sm.families.NegativeBinomial(alpha=1.0 / 6.0),
    ).fit()

    results["ols_naive"] = {
        "formula": "log_length ~ C(intended_star)",
        "coefficients": _coef_table(ols_naive),
        "r2": float(ols_naive.rsquared),
        "n": int(ols_naive.nobs),
        "star_curve_words": _star_curve_length(ols_naive, completed, log_scale=True),
    }
    results["ols_adjusted_categorical"] = {
        "formula": f"log_length ~ C(intended_star) + {_CONTROLS}",
        "coefficients": _coef_table(ols_adj),
        "r2": float(ols_adj.rsquared),
        "n": int(ols_adj.nobs),
        "star_curve_words": _star_curve_length(ols_adj, completed, log_scale=True),
    }
    results["ols_adjusted_quadratic"] = {
        "formula": f"log_length ~ star_u + {_CONTROLS}",
        "star_u_coef": float(ols_quad.params["star_u"]),
        "star_u_ci": [float(ols_quad.conf_int().loc["star_u", 0]),
                      float(ols_quad.conf_int().loc["star_u", 1])],
        "r2": float(ols_quad.rsquared),
    }
    results["ols_naive_quadratic"] = {
        "formula": "log_length ~ star_u",
        "star_u_coef": float(ols_naive_quad.params["star_u"]),
        "star_u_ci": [float(ols_naive_quad.conf_int().loc["star_u", 0]),
                      float(ols_naive_quad.conf_int().loc["star_u", 1])],
    }
    results["negbin_adjusted_quadratic"] = {
        "formula": f"review_length ~ star_u + {_CONTROLS} (NegativeBinomial, log link)",
        "star_u_coef": float(nb_adj.params["star_u"]),
        "star_u_ci": [float(nb_adj.conf_int().loc["star_u", 0]),
                      float(nb_adj.conf_int().loc["star_u", 1])],
        "aic": float(nb_adj.aic),
        "star_curve_words": _star_curve_length(nb_adj, completed, log_scale=False),
    }
    return results


def raw_descriptives(df: pd.DataFrame) -> dict:
    """Unadjusted completion rate and mean length by star, for the plots and README."""
    by_star = {}
    for star in range(1, 6):
        sub = df[df["intended_star"] == star]
        completed_sub = sub[sub["completed"] == 1]
        by_star[star] = {
            "n_attempts": int(len(sub)),
            "completion_rate": float(sub["completed"].mean()),
            "mean_length": float(completed_sub["review_length"].mean()),
            "median_length": float(completed_sub["review_length"].median()),
            "share_mobile": float(sub["device_mobile"].mean()),
            "mean_experience_z": float(sub["user_experience_z"].mean()),
        }
    return by_star
