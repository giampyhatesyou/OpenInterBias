# tests/

Pytest scaffolding. As of the `chore/repo-prep` branch:

- **No intersectional logic is tested here yet** — the increment is not yet built.
- The tests present validate **upstream invariants** that any future code must preserve:
  - the existing `proposed_biases/*.json` files parse and conform to the expected shape;
  - `utils/config.py` keeps the documented keys (`BIAS_PROPOSAL_SETTING`, `GEN_SETTING`, `VQA_SETTING`).

These are cheap sanity checks (no GPU, no model weights), suitable to run as a pre-commit hook or in CI later.

## Running

```bash
# from repo root
pip install pytest
pytest tests/ -v
```

If `pytest` is not installed, the test files are still readable Python — useful as living documentation of the upstream schema.

## Layout convention

```
tests/
  conftest.py                         # shared fixtures (paths)
  test_upstream_schema.py             # invariants on proposed_biases JSON
  test_config_shape.py                # invariants on utils/config.py
  intersectional/                     # placeholder — populated when increment lands
    .gitkeep
```

## What goes here later

Once the intersectional increment lands, add:

- `tests/intersectional/test_pairing.py` — pairing rules (refer_to match, same caption, valid bias filter).
- `tests/intersectional/test_scoring.py` — synthetic distributions: independent uniform (MI=0), perfectly skewed (intensity=1), perfectly dependent (MI=1).
- `tests/intersectional/test_schema.py` — JSON schema validation against canonical shape.

Do not add tests that require LLM/VQA/diffusion weights — those belong to integration tests outside this folder.
