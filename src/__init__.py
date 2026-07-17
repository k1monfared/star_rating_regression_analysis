"""Synthetic star-rating review-funnel analysis package.

Modules:
    data_generation : builds the synthetic review-attempt dataset from a documented
                      ground-truth generative model.
    analysis        : logistic and count regressions for completion and length,
                      naive vs adjusted, marginal effects.
    rdd             : optional half-star rounding regression-discontinuity extension.
    figures         : all committed figures.
    utils           : shared helpers (config loading, paths).
"""
