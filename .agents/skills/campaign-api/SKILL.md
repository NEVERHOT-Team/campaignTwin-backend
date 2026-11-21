---
name: campaign-api
description: Implement or verify Campaign Rehearsal's FastAPI endpoints, brief extraction, strategist proposals, and OpenAI Responses API integration.
---

Read the [frozen contract summary](../../../docs/contract-brief.md) and the checked-in schemas or OpenAPI if available. Resolve field names, units, ranges, and request ownership before editing a payload. If adding an integration example, take its shape from the agreed schema.

- **Extraction:** Use Pydantic-backed Structured Outputs. Check that the quoted text exists in the pasted brief and supports the proposed value. Surface missing or ambiguous values for confirmation. Use only human-confirmed inputs for simulation.
- **Strategist:** Validate model proposals against the plan schema and constraints before sending them to the engine. The engine computes the outcomes and ranks eligible plans. Tie change explanations to actual levers and results.
- **Competitor and shopper explanation:** Keep the R1 rule available independently of the optional model call. Give the explanation model only the relevant synthetic events and aggregate traits. Keep personal and raw household records out of LLM calls.
- **Failures and cache:** Handle missing, refused, or invalid model output explicitly. Include the inputs that affect the response in the disk cache key, including the model and prompt/schema version. Keep structurally invalid output out of the cache.

Verify a valid response and the relevant invalid-input or model-failure path for the endpoint touched. Use the installed SDK version and the [Responses API Structured Outputs guide](https://developers.openai.com/api/docs/guides/structured-outputs) for SDK call syntax.
