# Backend fixture data

`processed/` contains synthetic segments, personas, unit economics, rival examples,
purchase-intent anchors, and a held-out campaign example. Their source labels
are `default`. The campaign-library brands and outcomes are fictional, so their
`source_url` cells are empty.

Import the cached readers in `data/loaders.py` for all fixture reads. Treat the
returned tables and dictionaries as read-only. Keep household-level inputs in
the ignored `raw/` directory and out of model prompts.

Fields ending in `_pct` use percentage points. The `discount_depth` and rival
`depth` fields, probabilities, and shares use fractions between 0 and 1.
