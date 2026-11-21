"""Default message and mechanic affinities from weighted, aggregate persona profiles."""

import numpy as np
import pandas as pd

ARCHETYPES = ("loyal", "deal_hunter", "waiter", "explorer", "indifferent")
MECHANICS = ("percent_off", "voucher", "bundle", "free_shipping", "flash_sale", "coins_cashback")
MESSAGE_STYLES = ("price", "value", "urgency", "family", "premium")
APPEAL_KEYS = ("archetype", "mechanic", "message_style")


def build_appeal_fallback(personas: list[dict]) -> pd.DataFrame:
    profiles = pd.json_normalize(personas)
    rows = []
    for archetype in ARCHETYPES:
        group = profiles.loc[profiles["archetype"].eq(archetype)]
        features = group[
            [
                "promo_share",
                "brand_loyalty",
                "traits.impulsiveness",
                "traits.risk_aversion",
                "traits.time_preference",
            ]
        ]
        promo, loyalty, impulse, risk, immediacy = np.average(
            features, axis=0, weights=group["weight"]
        )
        styles = {
            "price": promo,
            "value": (1 - promo + risk) / 2,
            "urgency": promo + (1 - promo) * (impulse + immediacy) / 2,
            "family": (loyalty + risk + 1 - promo) / 3,
            "premium": (1 - promo + 1 - risk) / 2,
        }
        mechanics = {
            "percent_off": promo,
            "voucher": (promo + risk) / 2,
            "bundle": (loyalty + risk) / 2,
            "free_shipping": (1 - promo + risk) / 2,
            "flash_sale": (promo + impulse) / 2,
            "coins_cashback": (promo + 1 - immediacy) / 2,
        }
        # Move the specified design preferences halfway toward maximum affinity.
        preferred = {
            "deal_hunter": ("price", "urgency"),
            "loyal": ("value", "family"),
            "waiter": ("urgency",),
        }
        for style in preferred.get(archetype, ()):
            styles[style] = (1 + styles[style]) / 2
        if archetype == "waiter":
            mechanics["flash_sale"] = (1 + max(promo, mechanics["flash_sale"])) / 2
        for mechanic in MECHANICS:
            for style in MESSAGE_STYLES:
                appeal = 2 * (0.6 * styles[style] + 0.4 * mechanics[mechanic]) - 1
                rows.append((archetype, mechanic, style, round(appeal, 6), "default"))
    return pd.DataFrame(rows, columns=[*APPEAL_KEYS, "appeal", "label"])
