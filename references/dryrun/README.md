# Dry-run evidence summaries

Distilled evidence from REAL framework loads, one small YAML per
(model, framework version, platform): per layer kind, the per-module
quant-method classes, the runtime MoE / dense-MLP shape, the kv identity, and
compact prefill-branch evidence (kernel switch + the config threshold
candidate). Provenance points back to the raw probe traces.

These are summarized material, not raw data: the probe's full trace JSONs
(ordered op sequences, kernel timings, call paths, weight tables;
~100-180KB per variant) stay in the facts archive (opharness workspace) and
are deliberately NOT checked in. `summarize_dryruns` in
`aiconfigurator_core.sdk.models.facts` is the single owner of this format —
re-run it over refreshed traces to regenerate a summary.

Consumed by `check_facts_against_dryrun` (validating the hf→config
conversion) and readable as authoring evidence when writing or updating a
model class. Identity matters: a summary validates facts for its own
(framework version, platform, quant identity) only.
