# Model facts — upstream-produced dry-run evidence

One small YAML per (model, framework version, platform), produced by the
collection side from REAL framework dry runs (dummy-weight loads of depth-cut
variants). `get_model` compares its own derivations against these and logs a
warning on divergence — never a failure; no file, no check.

Format (`aic-dryrun-summary/v1`, ~70 lines; see the GLM-5.2 files here):

- `kv_cache_dtype` — the kv identity the framework actually resolved.
- `layer_kinds.<kind>.quant_by_module` — per-module quant-method classes,
  layer-normalized and deduped.
- `layer_kinds.<kind>.moe_runtime` — `{num_experts, topk, inter, router_width}`
  as executed. Note frameworks may fuse the shared expert into the routed
  experts (runtime = routed+shared / topk+shared); the build-time check knows
  this deliberate modeling decomposition and does not warn on it.
- `layer_kinds.<kind>.dense_mlp` / `prefill_branch` — authoring evidence
  (dense inter size; the kernel switch across probed isl lengths with the
  config threshold candidate).
- `provenance` — probe + source traces (raw traces stay in the facts archive,
  not in this repo).

Consumed by `sdk/models/helpers.py::model_facts_divergences` (called from
`get_model`). Producer tooling currently lives in the opharness workspace.
