# Reproduce the offline Layer A audit

Run `python scripts/e2e_layer_a.py` from the backend checkout. The command prints the A1–A13 verdicts and a ranked plan-by-scenario table. It writes the same text to `runs/e2e_layer_a.log` and the inputs, raw KPI percentiles, ranking weights, rival policies, check evidence and timings to `runs/e2e_layer_a.json`. Each invocation replaces those two local files. Git ignores them.

The reference run used dev commit `22ca47c3da1bf95bc400b47ec1bec65d25464625`, Python 3.11.14, NumPy 2.4.6 and pandas 3.1.1 on macOS arm64. Its expected fingerprint is **8 PASS, 5 FAIL, 0 BLOCKED**. The S1 median incremental profits round to **−฿9,721** for `g0-flash` and **−฿3,140,813** for `user-plan`. Campaign 18 category-sales lift rounds to **4.1463%**. Compare full-precision values from JSON before rounding for display. Wall-clock times vary by machine.

## Checkout, install and run

The original audit used a clean clone and an existing backend virtual environment. These are the checkout and test commands, with the virtual environment active. The separate Ruff invocation also runs when pytest stops during collection.

```sh
git clone --no-local --single-branch --branch dev https://github.com/NEVERHOT-Team/campaignTwin-backend.git current
cd current
git fetch origin && git checkout dev && git pull
git rev-parse HEAD origin/dev
export DB_MODE=json OPENAI_API_KEY=''
pytest -q
ruff check .
pytest -q --import-mode=importlib -rs
pytest -q tests/data/test_quality.py::test_published_report_thresholds_and_source_counts
python -m engine.backtest
```

For this reproduction we installed the backend in a fresh worktree virtual environment. With Python 3.11.14 available, the portable setup and run are:

```sh
uv venv --python 3.11.14 .venv
uv pip install --python .venv/bin/python -e '.[dev]' 'numpy==2.4.6' 'pandas==3.1.1' 'pytest==9.2.0' 'ruff==0.17.6'
source .venv/bin/activate
export DB_MODE=json OPENAI_API_KEY=''
python scripts/e2e_layer_a.py
```

To run the published script against the **exact historical engine**, start in a checkout containing the script. The following creates a separate worktree, installs only the audit file there, and uses the same interpreter as the active checkout. The script resolves imports and output paths against its own checkout.

```sh
audit_python="$PWD/.venv/bin/python"
git worktree add --detach ../layer-a-22ca47c 22ca47c
mkdir -p ../layer-a-22ca47c/scripts
cp scripts/e2e_layer_a.py ../layer-a-22ca47c/scripts/
(cd ../layer-a-22ca47c && "$audit_python" scripts/e2e_layer_a.py)
```

The JSON records both the executed checkout's `HEAD` and its locally fetched `origin/dev` ref. Read `head` as the revision that produced the numbers. Neither output requires Postgres, an API server, a model key, raw transactions, nor the local household calibration panel. The script returns exit code 0 when all thirteen checks pass and 1 when any check fails or is blocked. A recorded failure is a completed measurement. A blocked check lacks a usable prerequisite. Independent checks such as A12 and A13 still run. A failed P1 check affects the audit exit code even though the demo plan's release criterion is all P0s passing.

At the reference revision, ordinary pytest imports both `tests/data/test_backtest.py` and `tests/engine/test_backtest.py` as `test_backtest` and stops with one collection error. Importlib mode gives them distinct import names and exercises the suite. The clean macOS run had 336 passes and two skips for local calibration/raw files. Both skipped tests ran with those files available, yielding 338 passes. On Linux the clean importlib run had 335 passes, one failure and two skips. The failure at `tests/data/test_quality.py:337` compares an entire dictionary for exact equality. Linux recomputed `income_prior.low` as −1.7605819389331068 and the published value is −1.760581938933107. The result reproduced with and without the local calibration panel.

## Inputs and accounting

`load_demo_brief()["user_plan_spec"]` supplies the tissue 11.11 campaign. The context comes from `ContextPack.model_validate(load_mock("context_pack"))`. Calling `agents.strategist.propose(spec, context)` without a proposer uses its deterministic generation-0 templates. `arena.run(plan, base_seed=0)` receives no parameter or appeal override. The shopper loads its published parameters and the arena uses its offline archetype/mechanic/message-style appeal table. Restricting the scenario list to S1–S3 keeps the S4 red-team path out of the run.

One population of 5,000 agents uses seed 0. Ten paired simulation seeds, 0 through 9, cover four campaign weeks and two post-campaign weeks. For each seed all seven campaigns use the same no-campaign baseline. This yields 210 campaign simulations plus ten baseline simulations and 21 reported campaign/scenario rows. The baseline is a shared comparator with zero incremental profit by definition. It has no rank, spend-based return, leakage or breakpoint row in the arena output.

The evaluator calculates each seed's weekly gross profit as own-brand revenue after platform/payment fees, less unit cost and shipping. It rounds weekly gross profit to satang, subtracts the paired baseline, and subtracts the campaign's media allocation once over the active offer weeks. The final cumulative difference is six-week incremental profit. Discounted selling prices already reduce revenue. Campaign cost for the return denominator is redeemed discounted units × reference price × discount fraction, plus media. **iROI = incremental profit / campaign cost**, calculated per seed. The evaluator uses zero for this ratio when campaign cost is zero. Every reported P10, P50 and P90 is a NumPy percentile across the ten seed values for that metric. The displayed iROI is the median of ten ratios.

For each plan, `evaluator.scenario_weights` selects the highest-scoring rival in each scenario, takes its likelihood, then divides by the sum across S1–S3. In the reference run the raw likelihoods were 0.722872, 0.150709 and 0.076418. The normalized weights were 0.760918, 0.158641 and 0.080440. `evaluator.summarize` ranks descending by the weighted sum of the three scenario P50 profits, rounded to satang. Ties use descending S1 P10 profit and then ascending strategy ID. Rank one is a relative position and can still lose money.

## Acceptance checks

| Check | Computation and pass threshold | Why |
|---|---|---|
| A1 · Brief (Gan) | Load and validate the demo spec. Require `ready`, one 40% `percent_off` mechanic, event `11.11` and the channel set `shopee`, `line`. Record the budget. | Catch a wrong category-demo input or a draft before interpreting its results. |
| A2 · Parameters (Mic) | Compare every `load_shopper_params()` value with the matching field on `shopper.default_params()`. Require at least one fitted field and the `calibrated` label on both the active params and the seed-0 population. Log all per-field labels and values in JSON. | A silent fallback to class defaults can change every campaign's economics. |
| A3 · Strategies and knobs (Mic) | Require eight unique IDs comprising six `g0-*`, `user-plan` and `baseline`. All discounts must be ≤50%. For each campaign check `load_knobs()`: discount range, allowed mechanic, case-normalized channels, calendar, duration and any explicit cap. Treat `percent_off` as `percentage_discount` and `voucher` as either published voucher type. Use `flash_days` when present, otherwise end date minus start date. | The original audit checked the published knobs as well as the strategist's cap. At the reference commit the calendar omits 11.11, and the 2-day flash offer is below the 7-day minimum and uses an unlisted mechanic. |
| A4 · Rivals (Gan) | Directly call `get_rival_scenarios(spec, strategy)` for each of the eight plans. Require nine distinct rival/scenario pairs for IDs 764, 595 and 69 across S1–S3. Depth is 0–100 percentage points. S1 is passive at depth zero and must be within 0.01 of 0.72 for campaigns. The baseline must be passive with likelihood 1 throughout. S3 may add at most ten percentage points to the offer. | Verify the empirical scenario inputs actually used in the price comparison and keep raw likelihoods distinct from ranking weights. |
| A5 · Full arena (Mic) | Time one ordinary `arena.run` and require <60 seconds, 21 distinct expected result rows, offline fallback, 5,000 agents, seeds 0–9 and 4+2 weeks. A second run wraps the real `shopper.simulate` to observe all 220 calls and 1,320 weekly records. All numeric weekly fields, result fields and ranking values must be finite. Require 0 ≤ discounted units ≤ own units, and check the same bounds on measured overlap when present. | Catch missing scenarios, invalid quantities or a run too slow for the demo. Only the first run counts toward the 60-second limit. A5's logged step time includes the replay and observation. |
| A6 · Determinism (Mic) | Compare the complete primary and replay `ArenaResult` dataclasses for exact equality, including KPIs, weekly rows, seeds, rival policies and ranks. | Paired replays with identical inputs must give identical decisions. |
| A7 · A/A (Mic) | Reuse A5's ten no-campaign baselines as set A. Rebuild the same population with seed 0 and simulate the same baseline with seeds 10–19 as set B. Give `evaluator.aa_lift` matching replicate-slot keys 0–9 for both sets. It sums **own-brand sales over every seed and all six weeks**. Pass when `abs(100 × (sales_B − sales_A) / sales_A) ≤ 2`. | Measure the sales variation between independent seed sets when the offer is identical. Using the same random draws on both sides would hide that variation. |
| A8 · S2/S3 separation (Gan) | For `user-plan`, compare the full-precision P50 incremental profits in S2 and S3. Require different profits and a larger effective rival depth (the maximum rival discount) in S3. | An undercut should produce a distinct stress case. At the reference commit both scenarios use 30.155642% at the same time. |
| A9 · Demo story (Mic) | Require the user discount to be 40%, user P50 profit <0 in both S1 and S3, and at least one generated plan with S1 P50 profit >0 and a better arena rank than the user. | An AI plan that merely loses less cannot support a claim of positive incremental profit. |
| A10 · Leakage (Gan) | Require all 21 rows to have a `data` label and a leakage P50 in [0, 1]. Within each seed the evaluator divides the summed `discounted_units_would_buy` by all redeemed discounted units. The shopper derives the numerator from the same-week no-offer purchase counterfactual. No redemptions give an unavailable ratio. | Separate sales subsidized for shoppers who would already have bought from genuinely incremental sales. Zero is a valid measured result. |
| A11 · Breakpoint (Gan) | In every scenario require the user breakpoint to be `never_profitable`, or to have both a week and a cause. Take the top-ranked generated plan and require a non-null `peak_week` in each scenario. The evaluator takes the median cumulative incremental-profit curve across seeds, tracks its positive peak, and reports the first negative week after a positive peak. | The demo needs an interpretable loss or turning point. A highest-ranked plan whose curve never becomes positive has no peak week. |
| A12 · Campaign 18 (Mic) | Run `backtest.run_backtest(seeds=range(5), n_agents=5000)`, the same call as `python -m engine.backtest` with no flags. Use population seed 0, the eight historical campaign weeks, and the measured shelf/coupon offer from the published record. With holdout disabled, calculate each paired seed's **category** sales lift across our, rival and other brands. Average the five lifts and pass that unrounded mean to `data.backtest.evaluate_backtest` inside the runner. Require campaign 18, a finite lift, the published [1.71%, 6.24%] interval, the 19.02% naive benchmark and a recognized label. | Test whether the historical comparison runs and reports its calibration. Inside the inclusive interval is Calibrated. Outside it, absolute error from the midpoint up to twice the interval width is Partially calibrated. Larger errors are Uncalibrated. The label is reported independently of this execution check. |
| A13 · Unit economics (Gan) | From `load_market()`, calculate `ref_price × (1 − 0.40) × (1 − (platform_fee_pct + payment_fee_pct)/100) − unit_cost`. Require the result rounded to two decimals to equal −฿18.93. The reference inputs give `315.55 × 0.60 × 0.90 − 189.33 = −18.933`. | Verify the loss on one discounted unit before adding order-level shipping or media. |
