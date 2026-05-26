# tests/intersectional/

Placeholder folder for tests that will land **with** the intersectional increment. Empty for now on purpose — no source code to test yet.

Once the increment is implemented, this folder should contain at least:

| Test file | What it asserts |
|---|---|
| `test_schema.py` | Pairwise candidate records conform to `intersectional/SCHEMA.md`. Round-trip JSON → object → JSON is lossless. |
| `test_pairing.py` | The pairing rules from `docs/SCHEMA_DECISION.md` are enforced: same `refer_to` (Q2), `present_in_prompt=false` on both attributes (Q3), `min_support_intersect` respected. |
| `test_scoring.py` | Joint-entropy and MI on **synthetic** distributions: independent-uniform (MI=0, joint_intensity=0), perfectly skewed (joint_intensity=1), perfectly dependent (MI=1). Run without any model — pure numpy. |
| `test_exclusions.py` | `unknown` / `other` / `non-binary` are excluded before counting (Q5), matching upstream `make_plots.py:99-103`. |
| `test_integration_mock.py` | End-to-end on a hand-crafted toy `vqa_answers.json` + `proposed_biases.json` → verify the final `intersectional_results.json` has the expected shape and known-good numbers. |

## Design constraints for these tests

- **No GPU**. No model weights, no torch beyond what's needed for shapes.
- **No network**. All fixtures live under `tests/intersectional/fixtures/`.
- **Synthetic data over real**. Reproducible across machines, fast to run, easy to debug.
- **One test = one invariant**. Don't combine pairing + scoring in a single test.

Until those files exist, this folder is intentionally empty (other than this README) so the directory is not lost in git.
