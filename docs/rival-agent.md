# Rival S4 and the shared LLM wrapper

## Scope and units

S1-S3 come from the evidence model `data.rival.get_rival_scenarios(spec, strategy=None)`.
It reads a CampaignSpec v0.1 through `spec_inputs()` and takes an optional Strategy
whose offer replaces the brief's offer, so each plan gets its own rival response. The
baseline strategy yields `none` for every rival. When the evidence model is unavailable,
`red_team` falls back to the rivals in `mocks/context_pack.json`.

**Every rival `depth` and `depth_bound` is in percentage points (0-100)**, matching
`data/rival.py`, `reaction_base_rate.json` and `knob_ranges.json`. The campaign library
stores `discount_depth` as a fraction; `red_team` converts it to percent.

## Call before Arena simulation

```python
from agents.rival_policy import red_team

# strategies are already ranked by the caller; baseline entries are skipped.
s4_policies = red_team(spec, strategies, context.rivals)
payloads = [policy.model_dump(mode="json") for policy in s4_policies]
```

The highest-scoring eligible rival is used. Its rival-specific library history
sets the maximum depth, capped at `max_observed_depth * 1.1` and at 100%. The rival's
`margin_floor_pct` in `brands.csv` caps it at `100 - margin_floor_pct`. The first three
unique non-baseline strategies each receive their own S4 policy. Apply an S4
policy only to its `strategy_id`; these are not three simultaneous competitors.

`RedTeamScenario` is a local subclass of RivalScenario with `strategy_id`,
`rationale`, `label`, `depth_bound`, `bound_source`, and `evidence_label`. Serialize
each subclass object directly to retain those fields. Wrapping it in a Pydantic
container typed only as RivalScenario can omit subclass fields. The existing
wire contracts and engine have deliberately not been changed.

- `likelihood`, `ci_low`, and `ci_high` are 0.0: S4 is an unweighted stress test.
- Successful typed proposals are labelled `llm_estimate`; unavailable, refused,
  or invalid proposals use bounded S3 values and are labelled `default`.
- Missing S3 values use the prior's depth P90, minimum delay, and duration.
- Empty/missing relevant history bypasses the model and uses a clearly labelled
  prior-based cap. It is not described as an observed-depth estimate.
- Evidence IDs are selected from existing records for the chosen rival and
  category. Unknown and unrelated IDs are dropped.
- Synthetic library evidence retains `evidence_label=default`, even for an
  LLM-generated policy. Current fixtures are not evidence of real Thai reactions.
- This function proposes a counter-move; whether it damages a strategy more than
  S3 must be measured by the Arena. No simulated profit is invented here.

Optional bound fields can live on rival history records, or on supplied rival
records carrying additional metadata. `rival_margin_floor`, `margin_floor`, and
`max_discount_depth` use fractions. `rival_margin_floor_pct`, `margin_floor_pct`,
and `max_discount_pct` use percentage points. Margin floors follow `data/rival.py`:
the rival keeps that margin, so its discount ceiling is `100 - margin_floor`.
`max_discount_*` fields are ceilings used directly. Our own campaign's margin is
never used as a rival's margin.

## Shared LLM contract

`call_llm(model, messages, schema) -> schema | None` keeps the existing signature.
`strong` and `fast` resolve MODEL_STRONG and MODEL_FAST; explicit model IDs also
work. `.env` is loaded without overriding existing process environment variables.

The order is validated OpenAI cache, a typed OpenAI Chat Completions parse call
using `response_format=schema`, existing fixture cache/fixture, then None. The SDK
uses a 30-second timeout per request attempt and one retry. A refusal, malformed
response or timeout follows the same fallback path. `EMBED_MODEL` remains
available in the environment for SSR; this module does not make embedding calls.

Cache identities include provider, resolved model, prompt/schema version, full
schema, and messages. Fixture fallbacks use a separate provider identity and do
not poison the OpenAI cache or prevent a later provider retry. Replaying a valid
cached result is deterministic; a fresh uncached model generation is not claimed
to be deterministic. Cache failures are logged and do not discard valid output.

Enable INFO logging to see `path=cache`, `path=openai`, `path=fixture`, and
`path=none`. Errors redact API keys. The wrapper's SDK call follows the official
[Structured Outputs guide](https://developers.openai.com/api/docs/guides/structured-outputs).

```powershell
.venv/Scripts/python.exe -m agents.llm --check
```

The probe bypasses cache and fixtures, makes one small structured call for each
configured model, prints OK/FAIL with a sanitized error, and exits nonzero if
either fails. It does not change model configuration. Offline tests monkeypatch
the OpenAI client and isolate cache storage.
