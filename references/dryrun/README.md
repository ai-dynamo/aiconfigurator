# Dry-run evidence JSONs

Raw probe outputs from REAL framework loads of depth-cut dummy variants (one
JSON per layer-kind variant × quant identity; GLM-5.2 family, sglang 0.5.16,
SM90, tp1). These are the single evidence artifact for model facts — there is
no intermediate "recipe" format.

Each JSON records: per-module quant-method classes, full parameter shapes,
resolved server args (kv dtype, attention backends), and ordered per-op traces
with tensor shapes / kernels / framework call paths per (phase, batch, isl).

Consumed by `aiconfigurator_core.sdk.models.facts.check_facts_against_dryrun`
(validating the hf→config conversion) and readable as authoring evidence when
writing or updating a model class ("what does the framework actually execute
for this checkpoint"). Identity matters: a JSON validates facts for its own
(framework version, platform, quant identity) only.

Produced by the opharness probe (`probe/recipe_probe.py` there); regeneration
is one containerized dummy load per variant (~5 min bf16, +DeepGEMM JIT for
fp8). Candidate `collector/` stage if this direction is funded.
