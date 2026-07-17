# Data dictionary and generation notes

This file documents the two committed tables and how `src/data_generation.py`
builds them from the fixed seed and parameters in `configs/data_config.json`.

There are two committed tables.

## `review_attempts.csv`

One row per review ATTEMPT: a session in which a user opened the review composer
for a business with an intended star rating. This is the unit of analysis.

| Column | Type | Meaning |
| --- | --- | --- |
| `attempt_id` | int | Unique attempt identifier. |
| `business_id` | int | Business the attempt targets (joins to `businesses.csv`). |
| `business_category` | string | One of restaurants, shopping, nightlife, beauty, services, health. |
| `business_true_avg_rating` | float | Latent true average rating of the business (1 to 5). |
| `business_displayed_stars` | float | True average rounded to the nearest half star (what a user sees). |
| `business_review_count` | int | Number of prior reviews for the business (popularity proxy). |
| `intended_star` | int | Star rating the user intends to give (1 to 5). The treatment of interest. |
| `user_review_count` | int | Reviews the user has written before (raw experience). |
| `user_tenure_days` | int | Days since the user joined. |
| `user_experience_z` | float | Standardized log(1 + user_review_count), the modeling feature for experience. |
| `user_is_elite` | int | 1 if the user is an elite/highly active contributor. |
| `device_mobile` | int | 1 if the attempt happened on mobile, 0 on desktop. |
| `session_prompted` | int | 1 if the composer was opened from an ML nudge (for example "you visited X, write a review"). |
| `completed` | int | 1 if the review was submitted, 0 if abandoned. First outcome. |
| `review_length` | float | Word count of the submitted review, empty when `completed` is 0. Second outcome. |

## `businesses.csv`

One row per business. Used for context features and for the RDD extension.

| Column | Type | Meaning |
| --- | --- | --- |
| `business_id` | int | Unique business identifier. |
| `business_category` | string | Business category. |
| `business_true_avg_rating` | float | Latent true average rating (1 to 5). |
| `business_displayed_stars` | float | True average rounded to nearest half star. |
| `business_review_count` | int | Number of prior reviews. |

## How the data is generated

Generation order is deliberate so that confounding is real. Confounders are drawn
first, then they shift the intended star rating, then both the star and the
confounders drive the two outcomes.

1. Businesses get a true average rating, a category, and a review count. The
   displayed stars are the true average rounded to the nearest half star.
2. Each attempt draws user experience (log review count, standardized), tenure,
   elite status, device (about 62% mobile), and whether the session was prompted.
   Prompts are more likely for experienced and mobile users.
3. The intended star rating is drawn from a J-shaped base distribution (mass at 4
   and 5, typical of review platforms) that is shifted by confounders:
   - mobile users skew toward the extremes (1 and 5),
   - experienced users skew toward moderate ratings (3 and 4),
   - higher business average pulls the intended star up.
   This is what entangles the star with the confounders.
4. Completion is a Bernoulli draw from a logistic model whose linear predictor
   includes a POSITIVE coefficient on `(star - 3)^2` (the injected U-shape:
   completion highest at 1 and 5) plus effects of experience (+), mobile (-),
   prompt (+), elite (+), business popularity (+), and category.
5. Review length (for the generated full length) is a negative binomial count
   with a log mean that includes a NEGATIVE coefficient on `(star - 3)^2` (the
   injected inverted-U: shorter at the extremes) plus experience (+), mobile (-),
   prompt (+), elite (+), and category. Length is only recorded when completed.

## Injected ground truth (what the analysis should recover)

| Quantity | Truth | Location |
| --- | --- | --- |
| Completion U-shape coefficient on `(star-3)^2` (logit) | +0.200 | `completion_model.star_u_coef` |
| Length U-shape coefficient on `(star-3)^2` (log mean) | -0.075 | `length_model.star_u_coef` |
| Mobile effect on completion (logit) | -0.700 | `completion_model.mobile_coef` |
| Experience effect on completion (logit) | +0.450 | `completion_model.experience_coef` |
| Prompt effect on completion (logit) | +0.850 | `completion_model.prompt_coef` |
| Mobile effect on length (log mean) | -0.240 | `length_model.mobile_coef` |
| RDD half-star rounding jump (log outcome) | +0.080 | `rdd.true_jump` |

## Why this is realistic

- Review platforms see J-shaped rating distributions with mass at the top.
- Strong emotions (very good or very bad experiences) motivate people to finish a
  review, and those reactions are often quick and short. Middling experiences are
  harder to articulate, so people abandon more but write longer when they do.
- Mobile sessions are quicker and more likely to be abandoned and shorter, and
  quick emotional reactions skew to the extremes: a real confounder.
- Experienced contributors are more measured (moderate ratings) and more likely
  to finish and to write at length: another real confounder.

Because the confounders move both the star and the outcomes, the raw pattern by
star is biased, and only a model that controls for them recovers the injected
truth. That is the point of the exercise.
