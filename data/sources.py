"""Public source assets; no credentials or household data in this registry."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SourceAsset:
    dataset: str
    filename: str
    url: str
    source_page: str
    license: str
    md5: str | None = None

    @property
    def key(self) -> str:
        return f"{self.dataset}/{self.filename}"


CJ_BASE = "https://raw.githubusercontent.com/bradleyboehmke/completejourney/master/"
CJ_PAGE = "https://bradleyboehmke.github.io/completejourney/"
TWIN_BASE = "https://huggingface.co/datasets/LLM-Digital-Twin/Twin-2K-500/resolve/main/"
TWIN_PAGE = "https://huggingface.co/datasets/LLM-Digital-Twin/Twin-2K-500"
CJ_TABLES = {
    "transactions": "transactions.rds",
    "promotions": "promotions.rds",
    "demographics": "demographics.rda",
    "products": "products.rda",
    "campaigns": "campaigns.rda",
    "campaign_descriptions": "campaign_descriptions.rda",
    "coupons": "coupons.rda",
    "coupon_redemptions": "coupon_redemptions.rda",
}
TWIN_FILES = (
    "question_catalog.json",
    "question_catalog_README.md",
    "wave1_3_response.csv",
    "wave4_response.csv",
)
ASSETS = {
    asset.key: asset
    for asset in [
        *[
            SourceAsset(
                "completejourney",
                filename,
                CJ_BASE + "data/" + filename,
                CJ_PAGE,
                "CC0 (completejourney package); retain upstream attribution",
            )
            for filename in CJ_TABLES.values()
        ],
        SourceAsset("completejourney", "DESCRIPTION", CJ_BASE + "DESCRIPTION", CJ_PAGE, "CC0"),
        *[
            SourceAsset(
                "twin2k",
                filename,
                TWIN_BASE + "question_catalog_and_human_response_csv/" + filename,
                TWIN_PAGE,
                "CC-BY-4.0",
            )
            for filename in TWIN_FILES
        ],
        SourceAsset("twin2k", "README.md", TWIN_BASE + "README.md", TWIN_PAGE, "CC-BY-4.0"),
        SourceAsset(
            "hillstrom",
            "hillstrom.csv.gz",
            "https://hillstorm1.s3.us-east-2.amazonaws.com/hillstorm_no_indices.csv.gz",
            "https://www.uplift-modeling.com/en/latest/api/datasets/fetch_hillstrom.html",
            "Not specified on the author's page; retain Kevin Hillstrom attribution",
            "a68a81291f53a14f4e29002629803ba3",
        ),
        SourceAsset(
            "breakfast",
            "breakfast_at_the_frat.zip",
            "https://downloads.ctfassets.net/psj0p18eh7z1/7B4xI3FKcgMLak9Wm95IHC/"
            "3b63fe3d37bc7122d13524514be4ab51/dunnhumby_Breakfast-at-the-Frat.zip",
            "https://www.dunnhumby.com/source-files/",
            "Not specified on source page; retain dunnhumby attribution",
        ),
    ]
}
