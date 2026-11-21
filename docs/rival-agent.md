# Rival S4 and the shared LLM wrapper

## Scope

`data/rival.py` remains unchanged. S1-S3 continue to come from
`ContextPack.model_validate(load_mock("context_pack")).rivals`. This change does
not implement the deferred evidence likelihood model or recalibrate its weights.
The S1-S3 interval tests check the current fixture, not statistical calibration.

The current checkout has `CampaignSpec` in `contracts/models.py`; it does not
contain `contracts/campaign_spec.py`. Rival helpers also accept a dictionary
containing the proposed nested `brand.name` and `brand.category`. Explicit brand
data enables exclusion of our own brand. The old flat CampaignSpec cannot carry
a brand name, so the helper cannot infer it from that object alone.

## Call before Arena simulation

```python
from agents.rival_policy import red_team

# strategies are already ranked by the caller; baseline entries are skipped.
s4_policies = red_team(spec, strategies, context.rivals)
payloads = [policy.model_dump(mode="json") for policy in s4_policies]
```

The highest-scoring eligible rival is used. Its rival-specific library history
sets the maximum depth, capped at `max_observed_depth * 1.1` and at 1.0. Optional
rival-specific affordability bounds further reduce this cap. The first three
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
and `max_discount_pct` use percentage points. Following the POC request, the
margin fields here are interpreted as maximum affordable **discount ceilings**.
If upstream data describes minimum retained profit margin instead, convert that
quantity to a discount ceiling using the agreed unit economics first. Our own
campaign's margin is never used as a rival's margin.

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
