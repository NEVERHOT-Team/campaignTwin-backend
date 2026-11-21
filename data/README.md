# CampaignTwin data

## Source

The `completejourney_py` documentation states:

> This dataset contains simulated grocery shopping data created for educational purposes. While based on realistic shopping patterns, it is not real consumer data.

Here, `data` means an estimate or observation from the packaged inputs. `default`
marks assumptions such as the exchange rate and marketplace fees. Backstories
are short templates with `llm_estimate` as a placeholder for Big's later integration.
[completejourney_py data notice](https://pypi.org/project/completejourney-py/#data-source-important-notice)

## Selected category

The catalogue path is `GROCERY / BATH TISSUES / TOILET TISSUE`. The shortlist
found 1,514 buying households, 28 redemptions matched to same-day eligible
coupon purchases, a median repeat interval of 4 weeks, and 74.0% of retailer
sales on lines with a retail or coupon discount. Six manufacturers each
accounted for at least 5% of category sales. Shares use transaction
`sales_value`, the amount received by the retailer.
[Complete Journey transaction and product definitions](https://bradleyboehmke.github.io/completejourney/articles/completejourney.html)

| Manufacturer ID | Demo role | Retailer sales share |
|---|---|---:|
| 751 | Our brand | 18.1% |
| 764 | Leader | 26.9% |
| 595 | Peer | 19.2% |
| 69 | Private label | 16.4% |

The other material manufacturers are 1011 (13.6%) and 1315 (5.9%). IDs and
sales shares come from the packaged catalogue and transactions. Demo roles
come from the team selection.

## Build

From the backend root, with the backend environment active:

```sh
pip install completejourney_py==0.1.0
python -m data.etl.download_raw
python -m data.etl.pick_category
python -m data.etl.build_tissue
```

The builder regenerates `personas.json`, `market.json`, `knob_ranges.json`,
`segments.parquet`, and `unit_economics.json` under `data/processed/`. Use
`data/etl/config.py` to set the USD-to-THB rate and the shipping, cost, margin,
and marketplace fee assumptions. The current reference price is the median
regular shelf price per purchased unit for manufacturer 751, converted at the
default rate. The knob percentiles use the observed positive discounts from
toilet-tissue purchase lines. The 100% maximum includes fully discounted
lines in the source data. Fields ending in `_pct` use percentage points.
Shares, probabilities, and trait values use fractions.

The 60 personas represent five fixed-seed KMeans archetypes derived from
recency, purchase-week frequency, spend, coupon usage, and promo share. The
observed joint age–income distribution comes from the 646 buying households
with demographic coverage. Each cell with at least 2% share appears, and each
archetype has at least ten profiles. Behavior profiles resample averages from
groups of households. Six `tt-edge-` profiles use the empirical tails of their
archetypes. Weights preserve archetype prevalence and the relative
demographic frequencies of the sampled strata. The legacy segment
`promo_response` field measures the share of purchase weeks with a price
promotion. Elasticity is an observational slope of log units per store visit
against log paid price within product-store pairs with positive tracked sales.
Archetypes without a supported negative slope use the pooled estimate.

For optional survey traits, place the public numeric `wave1_3_response.csv`
export in `data/raw/`, in its original CSV directory under `data/raw/`, or in
a `data/raw/Twin-2K-500/` checkout. The builder averages responses for
difficulty limiting spending (QID32), preference for earlier payment (QID84), and
preference for a sure payout (QID250). These are proxies for the three
traits. The checked-in personas use archetype defaults.
[Twin-2K-500 question catalog and numeric CSV](https://huggingface.co/datasets/LLM-Digital-Twin/Twin-2K-500/tree/main/question_catalog_and_human_response_csv)

## Read

Use `data/loaders.py` to read the generated inputs. The campaign library,
rival priors, purchase-intent anchors, and example back-test remain demo fixtures.

The builder also writes `data/calib/household_behavior.parquet`, a local
calibration panel with one row per category-buying household per study week.
The price columns are per-unit THB amounts. The brand field is the manufacturer
with the most purchased units in that household-week. Zero-purchase weeks
have zero units and null prices, discount, and brand. Mailer and display flags
describe purchased items, with no exposure inferred in zero-purchase weeks.
Panel values derive from `data` with a `default` exchange rate.
Keep `data/raw/` and `data/calib/` local and out of Git and model prompts.
