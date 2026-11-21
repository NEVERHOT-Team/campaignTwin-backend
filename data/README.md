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
python -m data.etl.build_rival_prior
```

The builder regenerates `personas.json`, `market.json`, `knob_ranges.json`,
`segments.parquet`, and `unit_economics.json` under `data/processed/`. Use
`data/etl/config.py` to set the USD-to-THB rate and the shipping, cost, margin,
and marketplace fee assumptions. The current reference price is the median
regular shelf price per purchased unit for manufacturer 751, converted at the
default rate. Fee collection has no date (`collected_at` is null) while the
fees use defaults. The knob percentiles use positive discounts from
toilet-tissue purchase lines. `discount_pct.raw_max` captures the observed
100% discount. The 50% planning cap in `discount_pct.max` excludes fully
discounted purchases from campaign proposals. Fields ending in `_pct` use percentage points.
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
promotion. Each design archetype retains its original KMeans name in
`cluster_label`. The deal hunter leads on promo share plus coupon rate.
Among the remaining clusters, the loyal group leads on brand loyalty minus
promo share. The waiter leads on the combined ranks of promo share and
purchase cycle length. The explorer has the lowest brand loyalty among the
last two clusters, and the other cluster receives the indifferent label.

`price_elasticity_observed` carries the `data` label and measures the slope
of log units per store visit against log paid price within product-store pairs
with positive tracked sales. The model uses `price_elasticity`, a `default`
prior ranked by mean promo share plus mean coupon rate, from −0.8 to −2.8 in
steps of 0.5. The pooled default is the midpoint, −1.8. The rationale is
“observational slope biased toward 0 by promo endogeneity.”

For optional survey traits, place the public numeric `wave1_3_response.csv`
export in `data/raw/`, in its original CSV directory under `data/raw/`, or in
a `data/raw/Twin-2K-500/` checkout. The builder averages responses for
difficulty limiting spending (QID32), preference for earlier payment (QID84), and
preference for a sure payout (QID250). These are proxies for the three
traits. The checked-in personas use archetype defaults.
[Twin-2K-500 question catalog and numeric CSV](https://huggingface.co/datasets/LLM-Digital-Twin/Twin-2K-500/tree/main/question_catalog_and_human_response_csv)

## Rival data

`build_rival_prior` writes four aggregate files under `data/processed/`.
`brands.csv` covers manufacturers 751, 764, 595, 69, 1011, and 1315. Average
price is THB paid per purchased unit. Price tiers divide those six prices into
thirds. Market share uses all toilet-tissue sales as its denominator. Typical
discount is the median positive line discount in percentage points. The two
archetypes with the most distinct buying households form each brand's target
segments. `brands_sources.json` records provenance by column. Brand ownership,
channels, geography, margin floor, and the USD-to-THB rate are team defaults.

`reaction_events.csv` has one row per triggering brand-week and rival. The
reference price is the median of a brand's available weekly median paid unit
prices in the preceding eight calendar weeks. A drop of at least 10% or a
mailer/display appearance in any store qualifies the week. Consecutive
qualifying weeks each count. For each rival, the closest qualifying move within
three weeks either side wins. Ties favor a same-week or later move, then a price
move. A price cut of at least 5% qualifies as a match within five percentage
points of the trigger, or an undercut if it exceeds that band. A placement
without an observed 5% cut qualifies as a non-price move. A placement trigger
with no observed discount uses zero depth.

`reaction_base_rate.json` pools 294 triggering brand-weeks and 1,470 rival
comparisons. The observed shares are 0.14% none, 3.20% match, 28.10% undercut,
and 68.57% non-price. Frequent placements across stores contribute to the
non-price share. These are associations in the simulated grocery panel. The
audit retains signed lags, including preceding moves. Scenario delay uses the
absolute lag in weeks. With fewer than 20 triggers, the observed probability
weight is `n_events / 20`. The remainder uses the default distribution
(45% none, 30% match, 15% undercut, 10% non-price), and the prior is labeled
`default`. `load_rival_prior()` exposes the probabilities under `p`.

## Rival retrieval

`data.rival.get_rival_scenarios(spec)` selects up to three competitors scoring
at least 0.5. The score weights category (35%), price tier (25%), target segments
(20%), channels (15%), and geography (5%). Adjacent tiers receive half credit.
Set overlap uses intersection divided by union. Market share breaks score ties.

For hand-entered rows in `campaign_library.csv`, put the manufacturer's ID in
`brand`. Set `reaction_to_campaign_id` to the triggering campaign ID when known.
The trigger supplies the features and description, and the response row supplies
the outcome, discount, dates, and delay. Leave `reaction_type` blank for an
unknown outcome. Use `none` for an observed absence of reaction. Enter
`discount_depth` as a fraction and `reaction_delay_days` in days.

The ten nearest campaigns are ranked by an equal mix of structured similarity
(mechanic, discount depth, duration, season, category) and TF-IDF cosine between
the brief's goal and the historical description. Retrieval uses a rival's own
rows when available. The eight checked-in coffee fixtures provide shared
analogues until brand-specific campaigns are added. Each known outcome votes
with its similarity as weight. Five prior observations supply Dirichlet smoothing.

Undercut risk is calibrated on a one-percentage-point grid. If a deeper offer's
raw estimate falls, it takes the highest risk at a shallower offer. The other
outcomes keep their relative odds and the posterior keeps its total weight.
The 90% intervals use the resulting marginal beta distributions. Evidence IDs
cover the voting campaigns, including any that set the undercut floor.

S1 describes no response, S2 the most likely outcome, and S3 an undercut.
S2 can repeat either of the other outcomes. Depth, delay, and duration use
similarity-weighted observations for the selected reaction, with category
estimates filling gaps. S2 uses median terms. S3 uses the 90th-percentile
depth and earliest delay. Depths are fractions capped at the rival's margin
floor allowance. A missing or zero discount produces three certain no-response
scenarios per selected rival.

## Read

Use `data/loaders.py` to read the generated inputs. The campaign library,
purchase-intent anchors, and example back-test remain demo fixtures.

The builder also writes `data/calib/household_behavior.parquet`, a local
calibration panel with one row per category-buying household per study week.
The price columns are per-unit THB amounts. The brand field is the manufacturer
with the most purchased units in that household-week. Zero-purchase weeks
have zero units and null prices, discount, and brand. Mailer and display flags
describe purchased items, with no exposure inferred in zero-purchase weeks.
Panel values derive from `data` with a `default` exchange rate.
Keep `data/raw/` and `data/calib/` local and out of Git and model prompts.
