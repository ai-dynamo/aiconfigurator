# Model recipes — trace-extracted reference facts

Reference artifacts, NOT runtime inputs. A recipe (`aic-model-recipe/v0` YAML)
is machine-extracted from real framework execution traces, one per
(model, framework version, quant identity): the ordered per-layer op sequence
with module identities, quant-method classes, tensor shapes, kernels and
framework call paths, plus derived branch guards and tp-validated sharding
rules.

Models keep being built the existing way — config interpretation feeding a
hand- or AI-authored model class. Recipes serve the other half of that loop:

- **Authoring evidence**: when adding or updating a model class (human or AI),
  the recipe answers "what does the framework actually execute for this
  checkpoint" — fused projections, shared-expert placement, skip-indexer
  layers, branch thresholds, TP sharding — without re-reading framework code.
- **Drift detection**: `aiconfigurator_core.sdk.models.check_model_against_recipe`
  introspects a built model's op graph against the recipe and reports
  MATCH / TOLERATED / DIVERGENT / UNCHECKED per op family
  (`tests/unit/sdk/models/test_recipe_check.py` pins the GLM-5.2 findings).

Provenance: sglang 0.5.16 traces on SM90, tp ∈ {1, 2}; extraction tooling
currently lives in the opharness pilot workspace (`probe/recipe_probe.py`,
`probe/recipe_extractor.py`); see `docs/model_recipes.md`.
