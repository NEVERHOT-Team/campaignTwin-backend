# Campaign Rehearsal backend

- Big owns FastAPI and OpenAI integration. Mic owns the NumPy engine and shopper traces. The team lead owns the data pipeline, back-test, and integration checkpoints.
- Before coding opens at 10:00 Bangkok time on 21 November 2026, prepare specs, decisions, and prompts only.
- Start integration work with [the frozen contract summary](docs/contract-brief.md). The backend owns the published schema and OpenAPI on build day. Bring any wire contract change to Phee and the affected backend owner before implementation.
- Use segment aggregates from groups of at least 50 households. Keep personal and household records out of LLM calls. Trace IDs identify simulated shoppers only.
- Route extraction, strategist, optional competitor decisions, and explanations through the OpenAI API. Use vectorized math for shopper decisions. Apply human confirmation before using extracted inputs.
- Report scenario ranges, profit after costs, brand outcomes, assumptions, and the confidence label. Keep data-derived results and assumed Thai market mechanics distinct.
- Use `$campaign-api` for API and AI work, `$simulation-validation` for the engine, strategist ranking, or back-test, and `$campaign-debug` and `$campaign-code-review` as needed.
- When delegation is requested, route work to `backend_api`, `simulation_engine`, or `data_integration`. Record what Codex built and the verification evidence for the team log.
