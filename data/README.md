# CampaignTwin data

## Source

The `completejourney_py` documentation states:

> This dataset contains simulated grocery shopping data created for educational purposes. While based on realistic shopping patterns, it is not real consumer data.

Here, `data` means an estimate or observation from the packaged inputs. `default`
marks assumptions such as the exchange rate and marketplace fees. Backstories
are short templates with `llm_estimate` as a placeholder for Big's later integration.
[completejourney_py data notice](https://pypi.org/project/completejourney-py/#data-source-important-notice)

## Selected category

The catalogue path is `GROCERY / BATH TISSUES / TOILET TISSUE`. The original full-panel shortlist
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
python -m data.etl.build_engine_inputs
python -m data.quality
```

The builder regenerates `personas.json`, `market.json`, `knob_ranges.json`,
`segments.parquet`, `unit_economics.json`, and `backtest_campaign.json` under `data/processed/`. Use
`data/etl/config.py` to set the USD-to-THB rate and the shipping, cost, margin,
and marketplace fee assumptions. The current reference price is the median
regular shelf price per purchased unit for manufacturer 751, converted at the
default rate. Fee collection has no date (`collected_at` is null) while the
fees use defaults. The knob percentiles use positive discounts from
toilet-tissue purchase lines. `discount_pct.raw_max` captures the observed
100% discount. The 50% planning cap in `discount_pct.max` excludes fully
discounted purchases from campaign proposals. Fields ending in `_pct` use percentage points.
Shares, probabilities, and trait values use fractions.

The 150 personas come from 150 distinct toilet-tissue buyers in weeks 1–44.
Each profile copies one buyer's observed purchase behaviour. The five fixed-seed
KMeans archetypes use recency, purchase-week frequency, spend, coupon usage,
and promo share. Sampling uses seed 42, with at least 20 profiles per archetype
and 15 edge profiles. Every age × income cell with at least 2% of buyers with
known demographics has a profile. All 1,417 buyers are eligible, including
the 796 who lack demographics. The 646 buyers with demographics across the
full 52 weeks include 25 who first bought tissue during the holdout.

The draw is without replacement within each archetype, demographic stratum
and edge group. Edges are the most distant tenth of each archetype from its
KMeans center. Common age × income cells, pooled rarer cells, and buyers
with missing demographics have separate base weights within each archetype.
Each sampled household starts with its stratum's population share divided
by its number of profiles. Age and income are `Unknown` when absent, which
gives unknown income a neutral elasticity factor. Missing size, child count,
and ownership use a seeded draw from a compatible household with complete
structure. The per-field `source` label distinguishes copied and imputed values.

Use these profiles locally for simulation. Behaviour and available demographics
come from the same source household, even though published persona IDs are
synthetic. For messaging scores, `load_appeal_fallback()` supplies an
archetype-level lookup that can be used without sending a profile to an LLM.

The legacy segment `promo_response` field measures the share of purchase weeks with a price
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
steps of 0.5 before income adjustment. The pooled base default is the midpoint,
−1.8. The rationale is “observational slope biased toward 0 by promo endogeneity.”

Multiply the archetype base prior by 1.2 for annual household income below
USD 35,000, by 1.0 from USD 35,000 to below USD 75,000, and by 0.85 from
USD 75,000 upwards. The segment prior is the within-segment weighted mean of
the adjusted persona priors. The three multipliers are modelling defaults.
Maier et al. found lower purchase intent among respondents reporting budget
pressure. That finding motivates the direction of this assumption.
[Maier et al. (2025), §4.3](https://arxiv.org/html/2510.08338v2#S4.SS3)

For optional survey traits, place the public numeric `wave1_3_response.csv`
export in `data/raw/`, in its original CSV directory under `data/raw/`, or in
a `data/raw/Twin-2K-500/` checkout. The builder averages responses for
difficulty limiting spending (QID32), preference for earlier payment (QID84), and
preference for a sure payout (QID250). These are proxies for the three
traits. The checked-in personas use archetype defaults.
[Twin-2K-500 question catalog and numeric CSV](https://huggingface.co/datasets/LLM-Digital-Twin/Twin-2K-500/tree/main/question_catalog_and_human_response_csv)

## Persona quality report

`python -m data.quality` writes `eval_report.json` and prints the checks.
Consumers can read the published aggregates through `load_eval_report()` without
the private calibration panel. A local rebuild without the panel marks the
reference-dependent checks as skipped and still measures directions, evidence
and assumption coverage.

Each reference household counts once. Buy rate uses all panel weeks. Promo and
coupon rates use purchase weeks, paid price is weighted by units, and purchase
cycle is the median gap between purchase weeks. For a single purchase week the
cycle is the observation length. Loyalty is the leading manufacturer's share of
all units across purchase lines. Rebuild older panels with
`python -m data.etl.build_tissue` to get that exact field.

The evaluator calls `engine.shopper.build_population(n_agents=5000, seed=42)`
once and reads its arrays directly. Each agent and reference household counts
once for KS (one minus the empirical CDF distance at six-decimal precision).
The engine adds Gaussian jitter to buy rate, purchase cycle and elasticity.
It takes units, promo share and loyalty directly from the sampled profiles,
subject to the engine's bounds. The PoC KS target is 0.85 for purchase
behaviour and 0.75 for brand loyalty.

In `marginal_fit`, `value` is the engine KS and `profile_value` is the
persona-weighted KS against households. Coupon rate and average price paid
appear only at profile level. The engine uses one market reference price.
The elasticity row compares the engine distribution with its weighted input
prior. This diagnostic has no target because the panel provides no per-household
elasticity measurements. Its observed slopes were fitted by archetype.

The buy-rate SD ratio uses the engine population. The correlation gap compares
the ten pairs of the five behaviours shared by the engine and reference households.
The prior direction checks group agents by the archetype and income of their
sampled personas. The observed income check compares weighted profile slopes
for annual income below USD 35,000 and at least USD 75,000. All household
comparisons use the same pre-holdout panel from the packaged simulated data.

`selection_bias` compares households with and without observed age and income
on all seven purchase attributes. Mean differences are with demographics minus
without, in the original units. A KS similarity below 0.90 triggers calibration.
For each flagged attribute, the builder groups all buyers into bins near quartiles
while keeping ties together. It fits positive weight multipliers to match the
population shares and preserves each original demographic stratum's mass within
its archetype. The audit records the cuts, multipliers, and target, initial and
achieved shares. The tissue build saves this aggregate audit in `data/calib/`
and the quality command publishes it in `eval_report.json`.

Five attributes triggered reweighting. The largest absolute gap between a
sampled bin share and its population target fell from 8.57 percentage points
to below 0.000001 percentage points. Agreement on these bins still leaves
room for differences within a bin and after jitter.

| KS similarity, seed 42 | 60 averaged | 150 own-jitter | 150 engine population | Target |
|---|---:|---:|---:|---:|
| Weekly buy rate | 0.6995 | 0.8384 | 0.8352 | 0.85 |
| Purchase cycle | 0.6977 | 0.8372 | 0.8368 | 0.85 |
| Average units | 0.4323 | 0.6317 | 0.9280 | 0.85 |
| Average price paid | 0.7719 | 0.9552 | — | — |
| Promo share | 0.6390 | 0.7786 | 0.9715 | 0.85 |
| Coupon rate | 0.6107 | 0.9779 | — | — |
| Brand loyalty | 0.5358 | 0.7472 | 0.9334 | 0.75 |
| Elasticity vs. configured prior | — | — | 0.8754 | — |

The first two columns preserve the published uniform-jitter comparisons for
commits `26e5e96` and `1d58e2a`, including the retrospective average-units
score for 60 profiles. All three use 5,000 agents. The profile KS is 0.9778
for coupon rate and 0.9607 for average price paid. The engine buy-rate SD ratio
is 1.0515 and the correlation gap is 0.0299. Both prior direction checks pass.
The observed income direction fails (low −0.2143, high −0.2475).

## Historical backtest

Complete Journey campaign 18 (Type A) ran from October 30 through December 24,
2017, covering retail weeks 45–52. Two of its coupon UPCs included ten
toilet-tissue products. We selected it for its eligible coupons and two
complete eight-week comparison windows. All model fitting uses weeks 1–44.
The cutoff applies before computing visits, clustering, elasticity, personas,
price and discount summaries, and rival priors. The local calibration panel
contains 1,417 households and 62,348 household-weeks. Rebuild it with
`python -m data.etl.build_tissue` when upgrading an existing checkout.

| Category measure | Baseline, weeks 37–44 | Campaign, weeks 45–52 | Change |
|---|---:|---:|---:|
| Retailer sales across all households (USD) | 5,221.50 | 5,669.07 | +447.57 (+8.57%) |
| Matched category coupon redemptions across campaigns | 2 | 8 | +6 (+300%) |

`actual.lift_pct_raw` is the 8.57% change in total toilet-tissue sales across
the two periods. For the seasonality control, we selected four comparable
household paper and hygiene product types in the GROCERY department. The ETL
checks every product in each type against campaign 18's coupon mappings.
Paper towels had nine coupon-eligible products, so they were excluded.

| Control product type | Weeks 37–44 (USD) | Weeks 45–52 (USD) | Sales change |
|---|---:|---:|---:|
| FACIAL TISSUE & PAPER HANDKE | 1,190.60 | 1,240.76 | +4.21% |
| PAPER NAPKINS | 436.91 | 558.75 | +27.89% |
| CLEANSING WIPES | 199.31 | 189.17 | −5.09% |
| BASIC PAPER PLATES | 294.57 | 295.88 | +0.44% |

Each change uses all households' retailer `sales_value`, including returns.
`actual.control_change_pct` is the equal-weight mean of the four unrounded
percentage changes (+6.86%). Mic reads `actual.lift_pct`, the difference
between raw tissue growth and control growth **(+1.71 percentage points)**.
Paper napkins (+27.89%) are a holiday outlier. The median change across the
four controls was +2.33%, giving a median-adjusted lift of +6.24 percentage
points. Report both estimates as the range `actual.lift_range_pct`, **[1.71,
6.24] percentage points**. The backtest passes when the simulated lift falls
inside this range, including either endpoint.
`seasonality_control.categories` in the JSON records each category's sales
and change. `actual.incremental_sales` is the raw sales increase of USD 447.57
converted to THB 15,709.71 at the configured default rate of 35.1.
The comparison is observational and other promotions can affect either group.

`data.backtest.naive_benchmark()` reads the published historical prediction of
**+19.02 percentage points**. Campaign 8 contributed +18.51 points using three
coupon-free controls. Campaign 13 contributed +19.54 points using basic paper
plates, its only eligible control. Campaign 27 lacked eight pre-launch weeks.
Each comparison uses mean weekly net sales to account for the shorter campaign
windows. Only complete active weeks count. The JSON records the windows, control
changes and exclusions. The benchmark is zero when no campaign qualifies.

`data.backtest.evaluate_backtest(simulated_lift_pct)` accepts percentage points.
Both endpoints of [1.71, 6.24] count as calibrated. Outside the interval, a
prediction is partially calibrated if its absolute error from the midpoint
(3.975) is at most 9.06 points, twice the interval width. Larger errors are
uncalibrated. Beating the naive planner requires a strictly smaller absolute
error from the same midpoint. Both functions work with the checked-in data.

`load_appeal_fallback()` returns 150 rows keyed by archetype, mechanic and message
style. The rule weights persona profiles within each archetype and combines
message affinity (60%) with mechanic affinity (40%) on a −1 to +1 scale. Every
row is labelled `default`. `load_assumptions()` returns the values, units,
provenance and consumers for the economic defaults, elasticity priors, rival
blend and persona trait fallbacks, alongside the simulated-data notice.

Each redemption must match a coupon-discounted purchase by the same household
on the same recorded date for a product eligible under that campaign and UPC.
Multiple eligible lines count as one redemption event. `treated_households`
is the number of assigned households (1,133). `actual.redemption_rate` is the
share of those households with a matched campaign-18 tissue redemption
(8/1,133, or 0.71%). Individual coupon receipt is unobserved.
`discount_depth` is the median realized shelf-to-paid discount on the matched
campaign lines, expressed as a fraction (0.222469). `target_segments` is empty
because the assignments do not record a design-archetype targeting rule.
`calibration_weeks` lists the entire fit period. `source` is `data` for the
observations from the packaged simulated panel.

## Purchase-intent anchors

`load_anchors()` returns two Thai sets and one English set about buying toilet
tissue. Each set has five strings in ascending order of purchase intent, from
1 (definitely skip) to 5 (definitely buy). The ten labeled test responses give
one independent example per score in each language. These are authored
reference examples with `default` provenance.

## Rival data

`build_rival_prior` writes four aggregate files under `data/processed/`.
`brands.csv` covers manufacturers 751, 764, 595, 69, 1011, and 1315. Average
price is THB paid per purchased pack. Price tiers use paid THB per square foot
on packs with a measured area. Each brand receives a low, mid, or high tier
from its midpoint in the cumulative sales distribution ordered by this price.
Packs labeled only in rolls, sheets, or counts have no measured area. Coverage
among the three configured rivals is 40.1% of purchased units for 764, 99.8%
for 595, and 73.9% for 69. These give tiers high, low, and low respectively.
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

Of 219 pre-campaign trigger weeks, 141 had same-manufacturer controls, yielding
705 paired comparisons from 15 distinct sampled control brand-weeks. Nine matched
trigger weeks had at least one rival with a clear two-week history
(`n_events`). The event arm contained one match and two undercuts. The
background arm had no qualifying responses. Each reaction's lift is its
event rate minus its background rate, floored at zero. The remaining
probability goes to `none`. When that probability is below 0.20 or fewer
than 20 informative trigger weeks remain, the builder averages the adjusted
rates with the defaults (45% none, 30% match, 15% undercut, 10% non-price).

The resulting `data+default` prior is 72.29% none, 15.07% match, 7.64%
undercut, and 5.00% non-price. Among observed matches and undercuts, both the P50
and P90 depth are 30.16%. The median observed delay is three
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

For a positive discount, S1 reports the probability of no response. S2 selects
the most likely of match, undercut, and non-price. Its likelihood uses the
total posterior weight across all four outcomes. S3 describes an undercut.
Depth, delay, and duration use
similarity-weighted observations for the selected reaction, with category
estimates filling gaps. S2 uses median terms. S3 uses the 90th-percentile
depth and earliest delay. Rival scenario depths, audit depths and prior depth
percentiles use percent (0–100). Retrieval converts the library's fractional
discount once. All scenarios respect the margin-floor allowance. S3 also caps
the response at ten percentage points above the proposed discount. A missing or zero discount produces three certain no-response
scenarios per selected rival.

## Read

Use `data/loaders.py` to read the generated inputs. The campaign library retains its demo rows.

The builder also writes `data/calib/household_behavior.parquet`, a local
calibration panel with one row per pre-campaign category-buying household per fit week.
The price columns are per-unit THB amounts. The brand field is the manufacturer
with the most purchased units in that household-week. Zero-purchase weeks
have zero units and null prices, discount, and brand. Mailer and display flags
describe purchased items, with no exposure inferred in zero-purchase weeks.
Panel values derive from `data` with a `default` exchange rate.
Keep `data/raw/` and `data/calib/` local and out of Git and model prompts.

## Downloaded public sources

The source downloader uses `data/sources.py` and records actual URLs, retrieval
times, byte counts, SHA256 checksums, attribution, and table metadata in
`processed/source_manifest.json`. See `processed/source_quality.json` for the
observed counts and limitations from the latest validation.

```powershell
.venv/Scripts/python.exe -m pip install -e ".[data]"
.venv/Scripts/python.exe -m data.etl.download_sources
.venv/Scripts/python.exe -m data.etl.validate_sources
```

Downloads are resumable by asset: reruns skip files matching their recorded
checksum and repair corrupted local files. Unexpected remote changes to a
recorded source checksum are rejected. Hillstrom is additionally checked against
the checksum published by scikit-uplift. Download errors preserve existing files.

| Source | Local preparation | Attribution |
|---|---|---|
| Complete Journey | Eight original R assets; seven normalized Parquet tables | [completejourney](https://bradleyboehmke.github.io/completejourney/), Brad Boehmke / Steven Mortimer, original study 84.51° |
| Twin-2K-500 | Numeric responses for waves 1–3 and wave 4, question catalog and documentation | [LLM-Digital-Twin](https://huggingface.co/datasets/LLM-Digital-Twin/Twin-2K-500), CC-BY-4.0 |
| Hillstrom | Compressed CSV with 64,000 customers | Kevin Hillstrom / MineThatData; [scikit-uplift mirror](https://www.uplift-modeling.com/en/latest/api/datasets/fetch_hillstrom.html) because the author's download host was unavailable |
| Breakfast at the Frat | Original ZIP containing workbook and user guide | [dunnhumby Source Files](https://www.dunnhumby.com/source-files/) |

Originals and normalized household/respondent tables stay in ignored `raw/`.
The full promotions RDS is downloaded but bulk conversion is opt-in because it
contains about 20.9 million rows and R parsing is expensive on a small laptop.
Run the downloader with `--datasets completejourney --normalize-promotions`
to explicitly convert it. `load_completejourney` requires
an explicit `limit` for unnormalized promotions to prevent accidental bulk reads.
Even a limited R read may scan the original compressed object; it is an ETL
operation, not something to perform inside a simulation loop.

Read all source data through the loaders:

```python
from data.loaders import (
    list_breakfast_files,
    load_breakfast_table,
    load_completejourney,
    load_hillstrom,
    load_twin_catalog,
    load_twin_responses,
)

transactions = load_completejourney("transactions")
demographics = load_completejourney("demographics")
products = load_completejourney("products")
survey = load_twin_responses("1_3")
repeat_survey = load_twin_responses("4")
questions = load_twin_catalog()
randomized_evaluation = load_hillstrom()
workbook = next(name for name in list_breakfast_files() if name.endswith(".xlsx"))
weekly_sample = load_breakfast_table(
    workbook, sheet_name="dh Transaction Data", nrows=1000,
)
```

`load_personas()`, `load_segments()`, and `load_unit_econ()` read the processed
tissue inputs built above. `load_library()` reads the demo campaign library.
The public-source download and validation commands prepare local source files.

Important source limitations:

- Complete Journey prices are USD. Thai prices, incomes, channels, costs, and fees
  have not been verified or inferred by this downloader.
- Transactions include product IDs without lookup metadata; the quality report
  records their count. Handle these explicitly before fitting a category model.
- The product `brand` field distinguishes National/Private labels. Manufacturer
  IDs are available, but are not named Thai competitors.
- Twin survey respondents are a separate US sample. Never join their participant
  IDs to grocery household IDs or describe the result as observed joint data.
- Keep a historical campaign held out before calibration. Hillstrom is an
  independent evaluation experiment, not a grocery campaign holdout.
- Before creating personas, publish only aggregates with at least 50 households;
  retain provenance for observed behavior versus assumed psychographics/context.
- Worksheet dimensions in the report include headers/formatting, so they are not
  claimed as verified data-row counts. The ZIP is never extracted to disk.
