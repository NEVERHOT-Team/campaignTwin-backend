# CampaignSpec v0.1

Implemented from `temp/campaign_spec.md` in the shared workspace. The canonical
models are in `contracts/campaign_spec.py`, re-exported as
`contracts.models.CampaignSpec`. FastAPI publishes the schema at `/openapi.json`.
The complete synthetic example is `mocks/campaign_spec.json` under `spec`.

## Ownership and flow

```text
Brief -> P1 CampaignSpec -> P2 ContextPack -> P3 human review -> SimulationPlan
```

CampaignSpec describes the requested campaign. Matched segment IDs, persona
weights, data-derived economics, and assumptions belong in ContextPack and the
reviewed plan. Audience hints are not resolved segment IDs.

`POST /api/brief/parse` still returns a fixed synthetic example regardless of
the submitted brief. Its values are labelled `default`, not `extracted`.
`POST /api/plan/context` still returns fixture context. This migration does not
implement either agent or replace the fixture simulation with the engine.

## Nested fields

| Group | Main fields |
|---|---|
| Identity | `spec_version`, `campaign_id`, `name`, `status` |
| Brand | `brand.name`, `brand.category`, `brand.price_tier` |
| Objective | `objective.primary`, `secondary[]`, `smart_goals[]` |
| Audience | `audience.description`, `include_hints[]`, `exclude_hints[]`, `geo[]` |
| Offer | `offer.mechanics[]`, `offer.message{headline,key_message,tone}` |
| Budget | `budget.total_thb`, `split{discount,media,fees}`, `max_discount_pct` |
| Delivery | `channels[]`, `timing{start_date,end_date,event,post_window_weeks}` |
| Economics | `unit_economics{avg_price_thb,margin_pct,fee_pct}` or `null` |
| Constraints | `constraints{margin_floor_pct,max_loss_prob,must_avoid,must_include}` |
| AI bounds | `strategist_bounds{discount_pct,mechanics_allowed,budget_flex_pct,timing_shift_days,segments_locked}` or `null` |
| Evidence | `competitor_hints[]`, `field_meta{}`, `open_questions[]` |

Channels use lowercase enum values: `shopee`, `lazada`, `tiktok_shop`, `line`,
`meta`, `google`, `offline`. Multiple mechanics and one primary objective with
optional secondary objectives are supported. Objective names express the user's
intent; the engine/evaluator must separately check which objectives they support.

## Units and validation

- Money is THB. Percentages use 0–100; `constraints.max_loss_prob` uses 0–1.
- `budget.split` contains THB allocations, not fractions; their sum must be at
  most `budget.total_thb`. This does not change `Strategy.budget_split`.
- Dates serialize as ISO 8601. End must follow start by at most 56 days.
- `post_window_weeks` defaults to 2. A short campaign is not automatically
  expanded to four weeks; the engine must map actual dates to its time steps.
- `percent_off.value` is a percentage; `fixed_off.value` and
  `free_shipping.value` are THB. For free shipping this is the shipping cap.
- Other mechanics with a numeric `value`, including vouchers, require
  `value_unit: "pct" | "thb"`. This resolves the draft's ambiguous voucher units.
  A voucher of 200 THB is not compared directly with a 25% discount cap.
- Each percentage mechanic is checked against `budget.max_discount_pct` and
  the supplied `strategist_bounds.discount_pct` range. The engine must also
  enforce effective stacked discounts, margin, eligibility, and actual spend.
- Numeric inputs must be finite; unknown fields and legacy flat fields are
  rejected instead of being silently ignored.

## Missing values and provenance

Use `null` for unknown values. Missing brand name/category, audience description,
budget, dates, channels, mechanics, or required mechanic values generate blocking
`open_questions` and set status to `needs_info`. Missing numeric targets can
remain a nonblocking question. `ready` with blocking questions is rejected.
`POST /api/runs` accepts only a `SimulationPlan` whose spec has status `ready`.
P3 remains responsible for human review; `ready` alone is not an approval record.

Every populated business field needs `field_meta` with a source of `user`,
`extracted`, or `default`. Metadata can cover a whole subtree when its source
is uniform (`budget`), or a specific path (`budget.total_thb`,
`offer.mechanics.0.value`) when sources differ. More specific entries take
precedence for consumers resolving provenance. Metadata paths must exist.
`extracted` entries require a nonempty evidence quote; P1 must additionally
check that the quote occurs in the original brief and supports the value.

Schema-supplied defaults receive `default` metadata automatically. Bookkeeping
fields `spec_version`, `status`, `field_meta`, and `open_questions` are excluded
from recursive source coverage. The IDs and campaign name still need metadata.
Economics and AI bounds remain `null` until provided or proposed in P2/P3;
there is no automatic +/-50% change to user-approved limits. If a bounds object
is supplied, unspecified budget/timing flexibility defaults to zero.

## Migration from the scaffold

| Old field | v0.1 |
|---|---|
| `goal` | `objective.primary` plus `objective.smart_goals` |
| `target_segments` | `audience` hints; resolved segments remain in context/strategies |
| `budget_thb` | `budget.total_thb` |
| `discount_pct`, `mechanic` | `offer.mechanics[]` |
| `channel` | `channels` using lowercase enum values |
| `start`, `end` | `timing.start_date`, `timing.end_date` |
| `kpi_target` | `objective.smart_goals[]` |
| `missing_fields` | `open_questions[]` with a `blocking` flag |

`BriefParseResponse` keeps its `{spec, questions}` envelope. Send its `spec` to
`POST /api/plan/context`. Python callers should use `model_dump(mode="json")`
when passing payloads to an HTTP client, because models now contain typed dates.
Existing API clients and saved plans containing the old flat shape must migrate.
Strategy, ContextPack, result/report endpoints and replay payloads are unchanged.
