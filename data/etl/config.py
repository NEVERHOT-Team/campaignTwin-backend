"""CampaignTwin demo configuration for toilet tissue."""

import re

CATEGORY = "toilet_tissue"
PRODUCT_PATH = ("GROCERY", "BATH TISSUES", "TOILET TISSUE")
# Complete Journey campaign 18 spans eight complete retail weeks in the 2017 panel.
BACKTEST_CAMPAIGN_ID = 18
BACKTEST_START_WEEK = 45
BACKTEST_END_WEEK = 52
MANUFACTURERS = {751: "our_brand", 764: "leader", 595: "peer", 69: "private_label"}
ROLE_PRICE_TIERS = {"our_brand": "mid", "leader": "high", "peer": "mid", "private_label": "low"}
SEED = 42
ELASTICITY_PRIOR_LEAST_PROMO = -0.8
ELASTICITY_PRIOR_MOST_PROMO = -2.8
INCOME_ELASTICITY_FACTORS = {"low": 1.2, "middle": 1.0, "high": 0.85}
INCOME_ELASTICITY_NOTE = (
    "Annual USD income is low below 35,000, middle from 35,000 to below 75,000, "
    "and high from 75,000. Apply the factor once to each persona's archetype base prior. "
    "Unclassified income retains the base prior. The factors are modelling defaults. "
    "Maier et al. (2025), §4.3, report lower purchase intent under budget pressure. "
    "https://arxiv.org/html/2510.08338v2#S4.SS3"
)
DISCOUNT_CAP_PCT = 50.0
MIN_SEGMENT_HOUSEHOLDS = 50

DEFAULTS = {
    "usd_thb": 35.1,
    "reference_margin_pct": 40.0,
    "margin_floor_pct": 10.0,
    "platform_fee_pct": 7.0,
    "payment_fee_pct": 3.0,
    "shipping_cost_per_order": 25.0,
    "media_cost_per_reach_thb": 0.32,
}
CURRENCY = "THB"
PANEL_WEEKS = {"campaign": 4, "post": 2}
CHANNELS = ("Shopee", "LINE")


def income_group(band: str | None) -> str | None:
    """Map Complete Journey annual USD income bands to the elasticity prior groups."""
    if not isinstance(band, str):
        return None
    if band == "Under 15K":
        return "low"
    match = re.fullmatch(r"(\d+)(?:-\d+K|K\+)", band)
    if not match:
        return None
    floor_k = int(match[1])
    if floor_k < 35:
        return "low"
    return "middle" if floor_k < 75 else "high"
