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
price is THB paid per purchased pack. Price tiers use paid THB per square foot
on packs with a measured area. Each brand receives a low, mid, or high tier
from its midpoint in the cumulative sales distribution ordered by this price.
Packs labeled only in rolls, sheets, or counts have no measured area. Coverage
among the three configured rivals is 40.3% of purchased units for 764, 99.8%
for 595, and 74.6% for 69. These give tiers high, low, and low respectively.
A brand without usable area gets its configured role tier, labeled `default`.
`brands_sources.json` records the comparable price, unit coverage, tier and
source for each brand.

Market share uses all toilet-tissue sales as its denominator. Typical discount
is the median positive line discount in percentage points. Target segments are
the two archetypes with the highest buyer-share lift. An archetype's share of
distinct households buying the brand is divided by its share of distinct
category buyers. Brand ownership, channels, geography, margin floor, and the
USD-to-THB rate are defaults.

### Event and background rates

The price reference is the median of the available weekly median paid unit
prices in the preceding eight weeks. A brand-week with at least a 10% cut or
a mailer/display placement in any store is a trigger. The audit covers only
weeks with two preceding weeks and three following weeks in the panel.

A reaction must start one, two, or three weeks after the anchor. The rival must
have two observed preceding weeks with a cut below 5% and no placement.
The earliest qualifying onset wins. A new cut of at least 5% matches within
five percentage points of the trigger depth and undercuts above that band.
A non-price reaction requires a new placement with an observed zero discount.
Ongoing cuts and placements cannot supply an onset. Missing prior prices
cannot establish that the rival was inactive.

For each trigger, the builder samples one non-event week from the same
manufacturer with replacement using seed 42. It evaluates the same rivals
and uses the trigger's depth and the same response rules in both windows.
When the rival fails the pre-event inactivity check, the comparison contributes
no new reaction. `reaction_events.csv` records both windows, their outcomes,
and the inactivity checks. Manufacturers 69 and 595 had no eligible non-event
week as triggering brands. Their trigger rows have empty background fields.
Both can still contribute as responding rivals.

Of 268 trigger weeks, 172 had same-manufacturer controls, yielding 860 paired
comparisons from 19 distinct sampled control brand-weeks. Fifteen matched
trigger weeks had at least one rival with a clear two-week history
(`n_events`). The event arm contained three matches and six undercuts. The
background arm had no qualifying responses. Each reaction's lift is its
event rate minus its background rate, floored at zero. The remaining
probability goes to `none`. When that probability is below 0.20 or fewer
than 20 informative trigger weeks remain, the builder averages the adjusted
rates with the defaults (45% none, 30% match, 15% undercut, 10% non-price).

The resulting `data+default` prior is 71.98% none, 15.17% match, 7.85%
undercut, and 5.00% non-price. Among observed matches and undercuts, the P50
depth is 12.85% and the P90 is 30.16%. The median observed delay is two
weeks. `load_rival_prior()` exposes the final probabilities under `p`, along
with both measured rates, the unblended lifts, and the sample counts.

## Rival retrieval

`data.rival.get_rival_scenarios(spec)` returns scenarios for the three rivals
in `data/etl/config.py` (764, 595, 69). Their score orders the results and is
available for display. It weights category (35%), price tier (25%), target
segments (20%), channels (15%), and geography (5%). Adjacent tiers receive
half credit. Set overlap uses intersection divided by union. Market share
breaks score ties.

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
depth and earliest delay. Rival scenario depths, audit depths and prior depth
percentiles use percent (0–100). Retrieval converts the library's fractional
discount once. All scenarios respect the margin-floor allowance. S3 also caps
the response at ten percentage points above the proposed discount. A missing or zero discount produces three certain no-response
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
