"""Published 5,000-agent KS comparisons against the weeks 1–44 buyer panel.

Both historical columns used seed 42 and independent uniform ±10% jitter.
Scores come from each release's eval_report.json except for the 60-profile
average units, evaluated retrospectively with the 150-profile release's
protocol. That protocol draws unit jitter after the six original attributes.
"""

HISTORICAL_SOURCES = {
    "averaged_60": "26e5e96e1bdc05434a83525949df370b8c293316",
    "own_jitter_150": "1d58e2aac18630be91a5b39c0248442aaaf52029",
}

HISTORICAL_KS = {
    "weekly_buy_rate": {"averaged_60": 0.6995105151729004, "own_jitter_150": 0.8384},
    "purchase_cycle_weeks": {"averaged_60": 0.6977105151729005, "own_jitter_150": 0.8372},
    "avg_units": {"averaged_60": 0.4323181369089626, "own_jitter_150": 0.6317181369089626},
    "avg_price_paid": {"averaged_60": 0.771930698659139, "own_jitter_150": 0.9552338743824982},
    "promo_share": {"averaged_60": 0.6390101623147495, "own_jitter_150": 0.7786101623147494},
    "coupon_rate": {"averaged_60": 0.6107203952011291, "own_jitter_150": 0.9778746647847565},
    "brand_loyalty": {"averaged_60": 0.535772759350741, "own_jitter_150": 0.747172759350741},
}
