"""CampaignTwin demo configuration for toilet tissue."""

CATEGORY = "toilet_tissue"
PRODUCT_PATH = ("GROCERY", "BATH TISSUES", "TOILET TISSUE")
MANUFACTURERS = {751: "our_brand", 764: "leader", 595: "peer", 69: "private_label"}
ROLE_PRICE_TIERS = {"our_brand": "mid", "leader": "high", "peer": "mid", "private_label": "low"}
SEED = 42
ELASTICITY_PRIOR_LEAST_PROMO = -0.8
ELASTICITY_PRIOR_MOST_PROMO = -2.8
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
