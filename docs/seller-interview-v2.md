# Seller interview API v2

A seller can start with “I want to promote rice next week” and a spending limit. The client can ask one question at a time, offer the selected listing’s voucher, and show what must be true for that voucher to pay off. The same review can accept a complete structured brief from a finance team.

## Try it alongside the team server

From the backend checkout, with the project environment activated, start a second local process on an available port.

```sh
DB_MODE=json python -m uvicorn api.v2.app:app --host 127.0.0.1 --port 8002
```

Open the [v2 directory](http://127.0.0.1:8002/api/v2/) or [interactive API docs](http://127.0.0.1:8002/api/v2/docs). The opt-in process also serves the existing client’s routes. Reviews, approvals and run identifiers belong to the API version that issued them. The standard team launcher continues to use its configured entry point. The examples here use the packaged synthetic catalog.

JSON mode needs no database migration. For an existing PostgreSQL backend, a team choosing v2 can apply [the dedicated v2 table](../db/seller_v2.sql) once in that database. The standard backend bootstrap and deployment command remain available for v1 only.

```sh
curl -sS http://127.0.0.1:8002/api/v2/campaign-reviews \
  -H 'Content-Type: application/json' \
  -d '{"fields":{"budget_thb":2500},"selling_period":{"first_day":"2026-12-01","last_day":"2026-12-07"}}'
```

The first response asks for the listing. These dates mean seven selling days in the listing’s market. The canonical end is December 8, exclusive. Sending `{}` starts a blank interview. Sending `brief`, existing planner `fields`, or both starts with the facts already available.

| Operation | How a client uses it |
| --- | --- |
| `GET /api/v2` | Discover the start URL, supported input responsibilities, example, and schema. |
| `POST /api/v2/campaign-reviews` | Save a first revision. Render `next_step.prompt`, its reason, and any choices. |
| `GET /api/v2/campaign-reviews/{id}` | Resume the saved session. Keep the ID across a page reload. |
| `POST /api/v2/campaign-reviews/{id}/turns` | Send `base_revision` and one reply, choice ID, or group of structured edits and confirmations. Send a replacement offer with all intended restrictions when editing it directly. |
| `POST /api/v2/campaign-reviews/{id}/what-if` | Enter hypothetical redeemed and additional orders for a resolved review. Returns seller arithmetic for that revision and leaves the review untouched. |
| `POST /api/v2/campaign-reviews/{id}/approvals` | Show the whole review, including material defaults. Submit the exact revision and explicit confirmation. |
| `POST /api/v2/runs` | Send the approval ID and a client-generated UUID in `submission_id`. Poll the returned URL and open the saved report or evaluation. |

For the rice example, send `{"base_revision":1,"answer":"Rice"}` to the returned `links.turn`. The next response quotes “Rice” and proposes one catalog ID. Confirm the candidate, select the eligible audience and goal, then select the catalog voucher and a profit target. The dated fixture supplies a ฿20 voucher on a ฿120 minimum order. The verified voucher value is fixed. On the adjustable tissue example, the guided flow also asks whether a smaller discount is permitted. A form can supply these decisions at creation. Use the current response’s choice IDs and revision on every turn.

On a stale write, reload `links.self` and show the current value before submitting a correction. A failed interpretation leaves the saved answers available for structured editing. The approval action is separate from a conversational “yes.” A direct structured client can approve a ready revision including its displayed policy defaults. A guided client should finish `next_step` and present the final check first.

## What belongs in the input

| Business question | Who supplies the answer | What the client can show |
| --- | --- | --- |
| Which item and selling unit will be promoted? | Seller selects or confirms one supported listing. | Product label, pack and channel together. A generic description can remain unresolved. |
| What has already been promised? | Seller supplies dates, eligible buyers, total spend, offer restrictions and permission to vary the offer. | A dated fixed voucher can be offered as a single choice, including its minimum spend. |
| What would count as success? | Seller chooses the outcome and target. Finance can set a floor per redeemed pack and an acceptable loss probability. | Incremental contribution in THB over the campaign and follow-up, against ordinary trading. Break-even is an explicit target choice. |
| What does an ordinary order cost? | Retrieve the dated listing evidence where available. Ask for corrections, missing fees, or seller-specific costs. | Each cost in `cost_inputs` carries value, basis, unit, origin and reference. An explicit zero is meaningful. |
| How many buyers are eligible and how might they respond? | Use the released aggregate population. The model proposes response, carryover, follow-up and rival-stress assumptions. | Show `policy_inputs`, `review.segments`, and `review.evaluation.assumptions`. A stress magnitude does not establish its likelihood. |
| What can the system calculate? | The service normalizes dates and uses the evaluator’s ledger to calculate ordinary and redeemed contribution, available voucher budget and the redemption ceiling. | Show `unit_economics` as a draft preview. Use `review.blockers` for unresolved terms. A budget ceiling measures capacity rather than expected demand. |

The brief’s strategist, shopper and rival roles motivate the later simulation. R2 places the commercial agreement first. The live single-pack path can demonstrate budget settlement, paired comparisons and risk checks. Persona-specific creative effects, calibrated rival likelihood, multi-product campaigns and new market mechanics need additional evidence and engine support. The interview leaves these outside the set of executable choices. SRS r2, §§2, 5 and 8 (local design dossier) · Original product brief (local design dossier) · Agent design (local design dossier)

## Why these choices

- **Ask for the fact that unlocks the next calculation.** GOV.UK’s question-page and check-answers patterns recommend focused questions followed by an editable summary. Here, the product and dates unlock the catalog terms. Asking for every fee first would make the seller reconstruct information the service can retrieve. The existing frontend’s controls also include outcomes and mechanics that its current evaluator cannot score. The server-provided choices let a new client offer the runnable subset. [Question pages](https://design-system.service.gov.uk/patterns/question-pages/) · [Check answers](https://design-system.service.gov.uk/patterns/check-answers/)
- **Make an interpretation cheap to correct.** Amershi and colleagues’ human–AI interaction guidelines emphasize disambiguation, recovery and the consequences of changes. V2 binds a short answer such as “1,000 THB” to the question actually displayed. It returns the quoted suggestion for confirmation, keeps a manual edit path, and rejects stale revisions. Free-text extraction can still miss restrictions. The review should display the normalized offer, audience and change permissions before approval. [Guidelines for Human–AI Interaction](https://www.microsoft.com/en-us/research/publication/guidelines-for-human-ai-interaction/)
- **Keep business commitments and behavioral assumptions visible.** Meridian’s scenario-planning guidance identifies cost, timing and value per outcome as quantities that can change between historical and planned campaigns. For this product, a seller can correct price, COGS and fees, and inspect the separate assumed purchase response. Changing a demand multiplier represents a different scenario and does not supply new evidence about actual buyers. [Meridian scenario planning](https://developers.google.com/meridian/docs/post-modeling/scenario-planning-and-future-budget-optimization)
- **Ask how many purchases the promotion adds.** Google’s Conversion Lift documentation defines incremental outcomes through a treatment/control comparison. For the pre-run arithmetic, the seller can instead posit how many voucher orders would have occurred at ordinary price. The SBA’s break-even calculation provides the fixed-cost and unit-contribution basis. The voucher calculation must additionally account for subsidies paid on existing demand. The seeded simulation later estimates the paired difference under its declared assumptions. [Conversion Lift measurement](https://support.google.com/google-ads/answer/14102450?hl=en) · [SBA break-even calculation](https://legacy.sba.gov/es/node/56191)

## What the what-if calculation answers

Suppose the price is ฿100, COGS is ฿55, the fee is ฿5 per kept order, fixed media is ฿100, and the voucher is ฿10. All other charges are zero. Ordinary contribution is ฿40 per order. Redeemed contribution is ฿30 before media.

With 120 redeemed orders, tell the service how many would be additional. For 20 additional orders, 100 of those orders would have happened anyway. Campaign contribution on the affected orders is 120 × ฿30 − ฿100 = ฿3,500. The ordinary comparator is 100 × ฿40 = ฿4,000. Incremental profit is **−฿500**. With a ฿25 target and 120 redemptions, at least **34 additional orders** are needed. With a ฿5 voucher, 118 redemptions and 18 additional orders, incremental profit becomes **+฿30**. These reproduce the SRS hand fixture. SRS hand calculation (local design dossier)

```json
{"base_revision":1,"redeemed_orders":120,"additional_orders":20}
```

The calculator holds other orders constant, charges committed media once, and assumes every additional order redeems one voucher. The result reports whether the target is reachable at that volume. Undefined or unreachable thresholds are returned as null. The full simulation applies its behavioral assumptions, paired seeds and decision guardrails to produce a recommendation.

## Extending the business model

For a new mechanic, first define the selling unit, eligibility, payer, settlement formula, evidence window and inputs that a seller may vary. Expose a catalog choice once the review resolver and evaluator accept the same terms. The interview already builds product choices from the supported catalog.

For a new goal, define its unit, direction, target, comparison population and measurement window. Implement and reconcile the paired measure before adding it to the goal choices. For example, an acquisition target needs an observable definition of a new buyer and a treatment of zero incremental buyers.

Keep an independent hand fixture beside each new calculation. Good extension tests cover zero versus unknown costs, date changes that invalidate evidence, a fixed offer, a stale edit, an unreachable target and a pinned run that survives a subsequent edit. The walkthrough above and its HTTP regression tests exercise the existing entry points. [Seller interview tests](../tests/test_v2_seller_interview.py)
