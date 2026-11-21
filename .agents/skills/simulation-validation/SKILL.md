---
name: simulation-validation
description: Verify Campaign Rehearsal's seeded shopper engine, profit and brand metrics, scenario ranking, and held-out campaign back-test.
---

Start with the [frozen contract summary](../../../docs/contract-brief.md), the assumptions config once available, and the actual engine/data outputs. Keep tuned parameters and evidence for the historical campaign separate from assumed Thai market mechanics.

- **Comparisons:** Keep the 10% shopper holdout distinct from the no-campaign counterfactual and the historical campaign held out for back-testing. Match scenarios and random streams across Plan A, proposed plans, and their baselines. Check that a plan change leaves unrelated random draws paired.
- **Metrics:** Verify the profit ledger against the agreed margin, discount, fee, and spend definitions. Check that discounts and costs are counted once. Verify P10 ≤ P50 ≤ P90 and trace each summary statistic to the documented sample set and unit. Keep brand measures separate from sales.
- **Ranking:** Verify the agreed 6-variant, 8-seed screen and 20-seed rerun of the top 3. Check that hard guardrails eliminate failing plans and that ranking uses the agreed definition of P10 profit across scenarios. Report if no eligible plan improves on Plan A.
- **Back-test:** Identify the held-out historical campaign before fitting and exclude it from tuning. Compare observed and simulated lift using the same window and metric. Record both lift values, the agreed error calculation, and the confidence label. If the denominator makes percentage error undefined, report that and the raw gap. Apply the agreed ±25% tolerance when percent error is defined.
- **Runtime and traces:** Verify the NumPy daily loop with a small seeded case and measure the full 4-plan × 4-scenario × 20-seed target when available. Check that sampled event logs and shopper details agree with the simulated run.
